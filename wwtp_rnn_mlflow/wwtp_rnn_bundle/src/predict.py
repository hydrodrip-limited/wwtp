"""
Load the registered WWTP effluent model from MLflow and run inference.

Demonstrates two deployment paths:
    1. Direct in-process load via mlflow.pyfunc — for batch scoring, notebooks,
       embedded use in a Python service.
    2. CLI-friendly entry point that consumes a CSV of new sensor readings
       and writes predictions back out.

The same model can also be served as an HTTP endpoint with no code changes:
    mlflow models serve -m "models:/wwtp_effluent_predictor/latest" -p 5000

Then:
    curl -X POST http://localhost:5000/invocations \\
         -H "Content-Type: application/json" \\
         -d '{"inputs": [[[...window of features...]]]}'
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd

from data_prep import DEFAULT_INPUT_COLS, DEFAULT_TARGET_COL, VALID_TARGETS, load_online_dataset


def load_registered_model(model_name: str, version: str | int = "latest"):
    """Load a model + its scalers from MLflow.

    The scalers are stored as an artifact under the source run, so we walk
    from registered model -> source run -> artifact.
    """
    client = mlflow.tracking.MlflowClient()
    if version == "latest":
        versions = client.search_model_versions(f"name='{model_name}'")
        latest = max(versions, key=lambda mv: int(mv.version))
        version = latest.version
        run_id = latest.run_id
    else:
        mv = client.get_model_version(model_name, str(version))
        run_id = mv.run_id

    model_uri = f"models:/{model_name}/{version}"
    model = mlflow.pyfunc.load_model(model_uri)

    # Find and download the scalers artifact for this run
    artifacts = client.list_artifacts(run_id)
    scaler_artifact = next(
        (a for a in artifacts if a.path.endswith("_scalers.joblib")), None
    )
    if scaler_artifact is None:
        raise RuntimeError(f"No scalers artifact found for run {run_id}")
    local_path = client.download_artifacts(run_id, scaler_artifact.path)
    scalers = joblib.load(local_path)

    return model, scalers, {"version": version, "run_id": run_id}


def build_inference_windows(
    df: pd.DataFrame,
    input_cols: list[str],
    feature_scaler,
    lookback: int,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Convert a raw sensor DataFrame into the (n, lookback, n_features) tensor the model expects."""
    df = df[input_cols].dropna()
    scaled = feature_scaler.transform(df.values)
    n = len(scaled)
    if n <= lookback:
        raise ValueError(f"Need more than {lookback} rows for inference, got {n}")
    X = np.stack([scaled[i - lookback : i] for i in range(lookback, n)]).astype("float32")
    return X, df.index[lookback:]


def predict(
    data_path: str,
    model_name: str,
    version: str = "latest",
    lookback: int = 8,
    target_name: str = DEFAULT_TARGET_COL,
    output_csv: str | None = None,
) -> pd.DataFrame:
    """Run inference and optionally write predictions to CSV."""
    model, scalers, meta = load_registered_model(model_name, version)
    print(f"Loaded {model_name} v{meta['version']} from run {meta['run_id']}")

    df = load_online_dataset(data_path)
    X, idx = build_inference_windows(
        df, DEFAULT_INPUT_COLS, scalers["feature_scaler"], lookback=lookback
    )

    y_pred_scaled = np.asarray(model.predict(X)).ravel()
    y_pred = scalers["target_scaler"].inverse_transform(
        y_pred_scaled.reshape(-1, 1)
    ).ravel()

    out = pd.DataFrame(
        {"timestamp": idx, f"predicted_{target_name}": y_pred}
    )
    if output_csv:
        out.to_csv(output_csv, index=False)
        print(f"Wrote {len(out)} predictions to {output_csv}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, help="CSV of new sensor readings")
    parser.add_argument("--model-name", default="wwtp_effluent_predictor")
    parser.add_argument("--version", default="latest")
    parser.add_argument("--lookback", type=int, default=8)
    parser.add_argument("--target", default=DEFAULT_TARGET_COL, choices=VALID_TARGETS,
                        help="Target name (only affects the output column header)")
    parser.add_argument("--output", default=None, help="Path to write predictions CSV")
    parser.add_argument("--tracking-uri", default="file:./mlruns")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    out = predict(
        data_path=args.data_path,
        model_name=args.model_name,
        version=args.version,
        lookback=args.lookback,
        target_name=args.target,
        output_csv=args.output,
    )
    print(out.head().to_string(index=False))
    print(f"... ({len(out)} rows total)")


if __name__ == "__main__":
    main()
