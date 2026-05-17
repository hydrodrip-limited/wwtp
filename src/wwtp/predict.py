"""
Load the registered WWTP effluent model from MLflow and run batch inference.

Because scalers are now **bundled** inside the MLflow pyfunc artifact via
:class:`~wwtp.predictor.WwtpPredictor`, a single
``mlflow.pyfunc.load_model()`` call replaces the previous fragile pattern
of hunting for a ``*_scalers.joblib`` file by run_id.

The same model can also be served as an HTTP endpoint with no code changes::

    mlflow models serve -m "models:/wwtp_effluent_predictor/latest" -p 5000

Then send windows directly::

    curl -X POST http://localhost:5000/invocations \\
         -H "Content-Type: application/json" \\
         -d '{"inputs": [[[...8x7 window of scaled features...]]]}'

Usage
-----
    python -m wwtp.predict \\
        --data-path evaluation_online_dataset.csv \\
        --model-name wwtp_effluent_predictor \\
        --version latest \\
        --lookback 8 \\
        --output evaluation_predictions.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from wwtp.config import DEFAULT_INPUT_COLS, VALID_TARGETS, Settings
from wwtp.data_prep import load_online_dataset
from wwtp.logging_cfg import get_logger

logger = get_logger(__name__)


def predict(
    data_path: str | Path,
    model_name: str,
    version: str = "latest",
    lookback: int = 8,
    target_name: str = "NH4",
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Run inference and optionally write predictions to CSV.

    Args:
        data_path: Path to a tab-separated online sensor CSV.
        model_name: Name of the registered model in MLflow.
        version: Model version string or ``"latest"``.
        lookback: Lookback window size (must match the trained model).
        target_name: Label for the output column header in the result CSV.
        output_csv: Optional path to write a predictions CSV.

    Returns:
        DataFrame with columns ``[timestamp, predicted_{target_name}]``.
    """
    model_uri = f"models:/{model_name}/{version}"
    logger.info("Loading model from %s", model_uri)
    predictor = mlflow.pyfunc.load_model(model_uri)

    df = load_online_dataset(data_path)
    input_cols = list(DEFAULT_INPUT_COLS)
    df = df[input_cols].dropna()

    n = len(df)
    if n <= lookback:
        raise ValueError(f"Dataset has only {n} rows; need more than {lookback} for windowing.")

    # Build raw (unscaled) windows — WwtpPredictor.predict handles scaling.
    arr = df.values.astype("float32")
    X = np.stack([arr[i - lookback : i] for i in range(lookback, n)])
    timestamps = df.index[lookback:]

    logger.info("Running inference on %d windows…", len(X))
    nh4_preds: np.ndarray = np.asarray(predictor.predict(X), dtype="float32")

    out = pd.DataFrame({"timestamp": timestamps, f"predicted_{target_name}": nh4_preds})

    if output_csv:
        out.to_csv(output_csv, index=False)
        logger.info("Wrote %d predictions to %s", len(out), output_csv)

    return out


def main() -> None:
    """CLI entry point for batch inference."""
    settings = Settings()

    parser = argparse.ArgumentParser(
        description="Run batch inference with the registered WWTP effluent model."
    )
    parser.add_argument("--data-path", required=True, help="CSV of new sensor readings")
    parser.add_argument("--model-name", default=settings.inference.model_name)
    parser.add_argument("--version", default=settings.inference.model_version)
    parser.add_argument("--lookback", type=int, default=settings.model.lookback)
    parser.add_argument(
        "--target",
        default=settings.data.target_col,
        choices=VALID_TARGETS,
        help="Target name (only affects the output column header)",
    )
    parser.add_argument("--output", default=None, help="Path to write predictions CSV")
    parser.add_argument("--tracking-uri", default=settings.mlflow_tracking_uri)
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
    print(f"… ({len(out)} rows total)")


if __name__ == "__main__":
    main()
