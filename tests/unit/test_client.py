"""Unit tests for ``wwtp.client.http.WwtpHttpClient``."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest

LOOKBACK = 8
N_FEATURES = 7


# ---------------------------------------------------------------------------
# Helpers to build a temporary scalers file
# ---------------------------------------------------------------------------


@pytest.fixture()
def scalers_file(tmp_path: Path, mock_scalers: dict) -> Path:
    """Write *mock_scalers* to a temporary .joblib file."""
    p = tmp_path / "scalers.joblib"
    joblib.dump(mock_scalers, p)
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWwtpHttpClientScaling:
    def test_predict_window_sends_scaled_payload(
        self,
        scalers_file: Path,
        rng: np.random.Generator,
        httpx_mock,
    ) -> None:
        """Client must scale the window before posting to /invocations."""
        from wwtp.client.http import WwtpHttpClient

        raw_window = rng.uniform(500, 3000, (LOOKBACK, N_FEATURES)).astype("float32")
        # Mock response: model returns a 2-D array [[value]]
        httpx_mock.add_response(
            method="POST",
            url="http://test-server/invocations",
            json={"predictions": [[2.34]]},
        )

        client = WwtpHttpClient(
            api_url="http://test-server/invocations",
            scalers_path=scalers_file,
        )
        result = client.predict_window(raw_window)
        assert isinstance(result, float)
        assert result >= 0.0  # clipping enforced

    def test_predict_window_result_is_non_negative(
        self,
        scalers_file: Path,
        rng: np.random.Generator,
        httpx_mock,
    ) -> None:
        """Client must clip predictions to [0, ∞) even if API returns negative."""
        from wwtp.client.http import WwtpHttpClient

        raw_window = rng.uniform(500, 3000, (LOOKBACK, N_FEATURES)).astype("float32")
        httpx_mock.add_response(
            method="POST",
            url="http://test-server/invocations",
            json={"predictions": [[-5.0]]},
        )

        client = WwtpHttpClient(
            api_url="http://test-server/invocations",
            scalers_path=scalers_file,
        )
        result = client.predict_window(raw_window)
        assert result == 0.0, f"Expected 0.0 after clipping, got {result}"

    def test_predict_windows_batch(
        self,
        scalers_file: Path,
        rng: np.random.Generator,
        httpx_mock,
    ) -> None:
        """Batch predict must return the same length as input windows."""
        from wwtp.client.http import WwtpHttpClient

        n = 5
        windows = rng.uniform(500, 3000, (n, LOOKBACK, N_FEATURES)).astype("float32")

        # API returns one prediction per window
        httpx_mock.add_response(
            method="POST",
            url="http://test-server/invocations",
            json={"predictions": [[float(i)] for i in range(n)]},
        )

        client = WwtpHttpClient(
            api_url="http://test-server/invocations",
            scalers_path=scalers_file,
        )
        results = client.predict_windows(windows)
        assert results.shape == (n,)


class TestWwtpHttpClientPredictFromCsv:
    def test_returns_dataframe_with_predictions(
        self,
        scalers_file: Path,
        tmp_csv: Path,
        httpx_mock,
    ) -> None:
        from wwtp.client.http import WwtpHttpClient

        # The CSV has 200 rows → 192 windows (200 - lookback=8)
        n_windows = 200 - LOOKBACK
        httpx_mock.add_response(
            method="POST",
            url="http://test-server/invocations",
            json={"predictions": [[1.0]] * n_windows},
        )
        client = WwtpHttpClient(
            api_url="http://test-server/invocations",
            scalers_path=scalers_file,
        )
        df = client.predict_from_csv(tmp_csv, lookback=LOOKBACK)
        assert "predicted_NH4_mg_per_l" in df.columns
        assert len(df) == n_windows

    def test_raises_when_csv_too_short(self, scalers_file: Path, tmp_path: Path, raw_df) -> None:
        from wwtp.client.http import WwtpHttpClient

        short_df = raw_df.head(LOOKBACK)  # exactly lookback rows — not enough
        p = tmp_path / "short.csv"
        short_df.to_csv(p, sep="\t", index=False)

        client = WwtpHttpClient(
            api_url="http://test-server/invocations",
            scalers_path=scalers_file,
        )
        with pytest.raises(ValueError, match="need more than"):
            client.predict_from_csv(p, lookback=LOOKBACK)

    def test_writes_output_csv_when_requested(
        self,
        scalers_file: Path,
        tmp_csv: Path,
        tmp_path: Path,
        httpx_mock,
    ) -> None:
        from wwtp.client.http import WwtpHttpClient

        n_windows = 200 - LOOKBACK
        httpx_mock.add_response(
            method="POST",
            url="http://test-server/invocations",
            json={"predictions": [[0.5]] * n_windows},
        )
        out_csv = tmp_path / "out.csv"
        client = WwtpHttpClient(
            api_url="http://test-server/invocations",
            scalers_path=scalers_file,
        )
        client.predict_from_csv(tmp_csv, lookback=LOOKBACK, output_csv=out_csv)
        assert out_csv.exists()


class TestWwtpHttpClientPing:
    def test_ping_returns_true_when_api_responds(self, scalers_file: Path, httpx_mock) -> None:
        from wwtp.client.http import WwtpHttpClient

        httpx_mock.add_response(
            method="GET",
            url="http://test-server/ping",
            status_code=200,
        )
        client = WwtpHttpClient(
            api_url="http://test-server/invocations",
            scalers_path=scalers_file,
        )
        assert client.ping() is True

    def test_ping_returns_false_on_connection_error(self, scalers_file: Path, httpx_mock) -> None:
        import httpx

        from wwtp.client.http import WwtpHttpClient

        httpx_mock.add_exception(httpx.ConnectError("refused"), url="http://unreachable/ping")
        client = WwtpHttpClient(
            api_url="http://unreachable/invocations",
            scalers_path=scalers_file,
        )
        assert client.ping() is False
