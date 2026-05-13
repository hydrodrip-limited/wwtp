"""
WWTP RNN prediction client.

The /invocations endpoint speaks scaled tensors — raw sensor readings must be
MinMax-scaled before sending, and the response must be inverse-scaled to get
NH4 in mg/L.  This client handles both steps using the scaler saved alongside
the model.

Usage
-----
    # Single prediction from a rolling 8-step sensor window
    python api_client.py \
        --scalers path/to/RNN_e50_b100_lb8_scalers.joblib \
        --window  "3200,1500,1500,1500,1500,1500,12.5" \
                  "3100,1480,1490,1500,1510,1490,12.4" \
                  "3050,1460,1480,1490,1500,1480,12.3" \
                  "3000,1440,1470,1480,1490,1470,12.2" \
                  "3100,1450,1480,1490,1500,1480,12.3" \
                  "3200,1460,1490,1500,1510,1490,12.4" \
                  "3150,1470,1480,1490,1500,1480,12.3" \
                  "3100,1460,1470,1480,1490,1470,12.2"

    # Batch scoring from a CSV file (tab-separated, same format as training data)
    python api_client.py \
        --scalers path/to/RNN_e50_b100_lb8_scalers.joblib \
        --csv     path/to/evaluation_online_dataset.csv \
        --lookback 8 \
        --output  predictions.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests

API_URL = "http://localhost:5000/invocations"

INPUT_COLS = ["Q_inf", "Q_air_1", "Q_air_2", "Q_air_3", "Q_air_4", "Q_air_5", "Temp"]


def scale_window(raw_window: np.ndarray, feature_scaler) -> np.ndarray:
    """Scale a (lookback, n_features) array to [0,1] using the fitted scaler."""
    lookback, n_feat = raw_window.shape
    return feature_scaler.transform(raw_window.reshape(-1, n_feat)).reshape(lookback, n_feat)


def call_invocations(scaled_windows: np.ndarray) -> np.ndarray:
    """
    POST to /invocations.

    Parameters
    ----------
    scaled_windows : (n_samples, lookback, n_features) float32 array

    Returns
    -------
    raw_output : (n_samples,) scaled predictions
    """
    payload = {"inputs": scaled_windows.tolist()}
    resp = requests.post(API_URL, json=payload, timeout=30)
    resp.raise_for_status()
    result = resp.json()
    # MLflow returns {"predictions": [[val], [val], ...]} for tensor outputs
    preds = result.get("predictions", result)
    return np.array(preds, dtype="float32").ravel()


def predict_raw_window(raw_window: np.ndarray, scalers: dict) -> float:
    """
    Predict NH4 (mg/L) from one raw sensor window.

    Parameters
    ----------
    raw_window : (8, 7) array of raw sensor readings
                 columns: Q_inf, Q_air_1..5, Temp
    scalers    : dict with 'feature_scaler' and 'target_scaler'

    Returns
    -------
    nh4_mg_per_l : float
    """
    scaled = scale_window(raw_window.astype("float32"), scalers["feature_scaler"])
    scaled_batch = scaled[np.newaxis]  # (1, 8, 7)
    scaled_pred = call_invocations(scaled_batch)
    nh4 = scalers["target_scaler"].inverse_transform(
        scaled_pred.reshape(-1, 1)
    ).ravel()
    return float(nh4[0])


def predict_from_csv(
    csv_path: str,
    scalers: dict,
    lookback: int = 8,
    output_csv: str | None = None,
) -> pd.DataFrame:
    """
    Batch-score a tab-separated sensor CSV file.

    Reads the same format as the training data (DATETIME + INPUT_COLS columns).
    Returns a DataFrame with columns [timestamp, predicted_NH4].
    """
    df = pd.read_csv(csv_path, sep="\t")
    df["DATETIME"] = pd.to_datetime(df["DATETIME"])
    df = df.sort_values("DATETIME").set_index("DATETIME")
    df = df[INPUT_COLS].apply(pd.to_numeric, errors="coerce").dropna()

    scaled = scalers["feature_scaler"].transform(df.values).astype("float32")

    n = len(scaled)
    if n <= lookback:
        raise ValueError(f"CSV has only {n} rows; need more than {lookback}")

    # Build all windows at once
    X = np.stack([scaled[i - lookback : i] for i in range(lookback, n)])
    timestamps = df.index[lookback:]

    # Call the API in one batch (split into chunks of 512 if very large)
    chunk_size = 512
    all_preds = []
    for start in range(0, len(X), chunk_size):
        chunk = X[start : start + chunk_size]
        all_preds.append(call_invocations(chunk))

    scaled_preds = np.concatenate(all_preds)
    nh4_preds = scalers["target_scaler"].inverse_transform(
        scaled_preds.reshape(-1, 1)
    ).ravel()

    out = pd.DataFrame({"timestamp": timestamps, "predicted_NH4_mg_per_l": nh4_preds})

    if output_csv:
        out.to_csv(output_csv, index=False)
        print(f"Wrote {len(out)} predictions to {output_csv}")

    return out


def _parse_window_args(args_window: list[str]) -> np.ndarray:
    rows = []
    for row_str in args_window:
        vals = [float(v.strip()) for v in row_str.split(",")]
        if len(vals) != 7:
            raise ValueError(
                f"Each window row needs 7 values (Q_inf, Q_air_1..5, Temp); got {len(vals)}: {row_str}"
            )
        rows.append(vals)
    if len(rows) != 8:
        raise ValueError(f"Window must have exactly 8 rows (lookback=8); got {len(rows)}")
    return np.array(rows, dtype="float32")


def main() -> None:
    parser = argparse.ArgumentParser(description="WWTP RNN prediction API client")
    parser.add_argument("--scalers", required=True, help="Path to *_scalers.joblib artifact")
    parser.add_argument("--url", default=API_URL, help="API URL (default: http://localhost:5000/invocations)")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--window", nargs=8, metavar="ROW",
                      help='8 rows of "Q_inf,Q_air_1,Q_air_2,Q_air_3,Q_air_4,Q_air_5,Temp" (raw units)')
    mode.add_argument("--csv", metavar="CSV_PATH",
                      help="Tab-separated sensor CSV (same format as training data)")

    parser.add_argument("--lookback", type=int, default=8, help="Lookback window size (default: 8)")
    parser.add_argument("--output", default=None, help="Path to write predictions CSV (batch mode only)")
    args = parser.parse_args()

    global API_URL
    API_URL = args.url

    scalers = joblib.load(args.scalers)

    if args.window:
        raw_window = _parse_window_args(args.window)
        nh4 = predict_raw_window(raw_window, scalers)
        print(f"Predicted NH4: {nh4:.4f} mg/L")
    else:
        out = predict_from_csv(args.csv, scalers, lookback=args.lookback, output_csv=args.output)
        print(out.head(10).to_string(index=False))
        print(f"... ({len(out)} rows total)")


if __name__ == "__main__":
    main()
