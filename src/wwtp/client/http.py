"""HTTP client for the MLflow ``/invocations`` prediction endpoint.

The ``/invocations`` endpoint speaks scaled tensors — raw sensor readings
must be MinMax-scaled before sending, and the response must be
inverse-scaled to obtain NH₄ in mg/L.  This module handles both steps
using the fitted scalers saved alongside the model.

Design notes
------------
* ``api_url`` is always passed as an argument — there is no module-level
  mutable global, so this module is safe to import and use from concurrent
  threads or multiple processes.
* All predictions are clipped to ``[0, ∞)`` because concentrations are
  physically non-negative.

Usage
-----
    from wwtp.client.http import WwtpHttpClient

    client = WwtpHttpClient(
        api_url="http://localhost:5000/invocations",
        scalers_path="path/to/scalers.joblib",
    )

    # Single window (8 × 7 raw sensor readings)
    nh4 = client.predict_window(raw_window)

    # Batch from a CSV file
    df = client.predict_from_csv("evaluation_online_dataset.csv", lookback=8)
"""

from __future__ import annotations

from pathlib import Path

import httpx
import joblib
import numpy as np
import pandas as pd

from wwtp.config import DEFAULT_INPUT_COLS
from wwtp.logging_cfg import get_logger

logger = get_logger(__name__)

_INPUT_COLS = list(DEFAULT_INPUT_COLS)


class WwtpHttpClient:
    """Thin wrapper around the MLflow ``/invocations`` REST endpoint.

    Args:
        api_url: Full URL of the prediction endpoint, e.g.
            ``"http://localhost:5000/invocations"``.
        scalers_path: Path to the ``*_scalers.joblib`` artifact that
            contains ``feature_scaler`` and ``target_scaler``.
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        api_url: str,
        scalers_path: str | Path,
        timeout: int = 30,
    ) -> None:
        self._api_url = api_url
        self._timeout = timeout
        self._scalers: dict = joblib.load(scalers_path)
        logger.info("Loaded scalers from %s", scalers_path)

    # ── Low-level HTTP call ──────────────────────────────────────────────────

    def _call_invocations(self, scaled_windows: np.ndarray) -> np.ndarray:
        """POST a batch of scaled windows to ``/invocations``.

        Args:
            scaled_windows: Float32 array of shape
                ``(n_samples, lookback, n_features)`` already scaled to
                ``[0, 1]``.

        Returns:
            Raw scaled predictions of shape ``(n_samples,)``.

        Raises:
            httpx.HTTPStatusError: On non-2xx responses.
        """
        payload = {"inputs": scaled_windows.tolist()}
        resp = httpx.post(self._api_url, json=payload, timeout=self._timeout)
        resp.raise_for_status()
        result = resp.json()
        preds = result.get("predictions", result)
        return np.array(preds, dtype="float32").ravel()

    # ── Public prediction helpers ────────────────────────────────────────────

    def predict_window(self, raw_window: np.ndarray) -> float:
        """Predict NH₄ (mg/L) from one raw sensor window.

        Args:
            raw_window: Array of shape ``(lookback, 7)`` containing raw
                sensor readings in physical units (m³/h, °C).

        Returns:
            Predicted NH₄ concentration in mg/L, clipped to ``[0, ∞)``.
        """
        lb, nf = raw_window.shape
        scaled = (
            self._scalers["feature_scaler"]
            .transform(raw_window.astype("float32").reshape(-1, nf))
            .reshape(1, lb, nf)
        )
        raw_pred = self._call_invocations(scaled)
        nh4 = self._scalers["target_scaler"].inverse_transform(raw_pred.reshape(-1, 1)).ravel()
        return float(np.clip(nh4[0], 0.0, None))

    def predict_windows(self, raw_windows: np.ndarray) -> np.ndarray:
        """Predict NH₄ (mg/L) for a batch of raw sensor windows.

        Args:
            raw_windows: Array of shape ``(n, lookback, n_features)``
                containing raw sensor readings.

        Returns:
            Array of shape ``(n,)`` with predicted NH₄ values in mg/L,
            clipped to ``[0, ∞)``.
        """
        n, lb, nf = raw_windows.shape
        scaled = (
            self._scalers["feature_scaler"]
            .transform(raw_windows.reshape(-1, nf).astype("float32"))
            .reshape(n, lb, nf)
        )

        # Process in chunks to avoid oversized HTTP payloads.
        chunk_size = 512
        all_preds: list[np.ndarray] = []
        for start in range(0, n, chunk_size):
            chunk = scaled[start : start + chunk_size]
            all_preds.append(self._call_invocations(chunk))

        raw_preds = np.concatenate(all_preds)
        return np.clip(
            self._scalers["target_scaler"].inverse_transform(raw_preds.reshape(-1, 1)).ravel(),
            0.0,
            None,
        )

    def predict_from_csv(
        self,
        csv_path: str | Path,
        lookback: int = 8,
        output_csv: str | Path | None = None,
    ) -> pd.DataFrame:
        """Batch-score a tab-separated sensor CSV file.

        Reads the same format as the training data (``DATETIME`` + input
        columns).

        Args:
            csv_path: Path to the tab-separated sensor CSV.
            lookback: Lookback window size.
            output_csv: Optional path to write the predictions CSV.

        Returns:
            DataFrame with columns ``[timestamp, predicted_NH4_mg_per_l]``.
        """
        df = pd.read_csv(csv_path, sep="\t")
        df["DATETIME"] = pd.to_datetime(df["DATETIME"])
        df = df.sort_values("DATETIME").set_index("DATETIME")
        df = df[_INPUT_COLS].apply(pd.to_numeric, errors="coerce").dropna()

        n = len(df)
        if n <= lookback:
            raise ValueError(f"CSV has only {n} rows; need more than {lookback} for windowing.")

        scaled_all = self._scalers["feature_scaler"].transform(df.values.astype("float32"))
        X = np.stack([scaled_all[i - lookback : i] for i in range(lookback, n)])
        timestamps = df.index[lookback:]

        preds = self.predict_windows(X)
        out = pd.DataFrame({"timestamp": timestamps, "predicted_NH4_mg_per_l": preds})

        if output_csv:
            out.to_csv(output_csv, index=False)
            logger.info("Wrote %d predictions to %s", len(out), output_csv)

        return out

    # ── Connectivity check ───────────────────────────────────────────────────

    def ping(self) -> bool:
        """Return ``True`` if the prediction server is reachable.

        Returns:
            Boolean indicating whether ``/ping`` responded with HTTP 200.
        """
        ping_url = self._api_url.replace("/invocations", "/ping")
        try:
            r = httpx.get(ping_url, timeout=3.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False
