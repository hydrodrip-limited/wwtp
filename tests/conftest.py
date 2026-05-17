"""
Pytest fixtures shared across all test modules.

The fixtures create minimal, fully deterministic datasets and objects that
let unit tests run without real sensor data, a live MLflow server, or a
trained Keras model.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from wwtp.config import DEFAULT_INPUT_COLS

# ─── Constants ────────────────────────────────────────────────────────────────

LOOKBACK = 8
N_FEATURES = len(DEFAULT_INPUT_COLS)  # 7
N_ROWS = 200  # large enough for 70/10/20 split after windowing


# ─── Data fixtures ────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """Deterministic random number generator."""
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def raw_df(rng: np.random.Generator) -> pd.DataFrame:
    """200-row synthetic sensor DataFrame (all INPUT_COLS + NH4)."""
    timestamps = pd.date_range("2023-01-01", periods=N_ROWS, freq="15min")
    cols = list(DEFAULT_INPUT_COLS)
    data = {
        "DATETIME": timestamps,
        "Q_inf": rng.uniform(1500, 4500, N_ROWS),
        "Q_air_1": rng.uniform(600, 2500, N_ROWS),
        "Q_air_2": rng.uniform(600, 2500, N_ROWS),
        "Q_air_3": rng.uniform(600, 2500, N_ROWS),
        "Q_air_4": rng.uniform(600, 2500, N_ROWS),
        "Q_air_5": rng.uniform(600, 2500, N_ROWS),
        "Temp": rng.uniform(10, 20, N_ROWS),
        "NH4": rng.uniform(0.5, 8.0, N_ROWS),
    }
    return pd.DataFrame(data)


@pytest.fixture(scope="session")
def tmp_csv(tmp_path_factory: pytest.TempPathFactory, raw_df: pd.DataFrame) -> Path:
    """Write *raw_df* as a tab-separated CSV and return the path."""
    p = tmp_path_factory.mktemp("data") / "sensor.csv"
    raw_df.to_csv(p, sep="\t", index=False)
    return p


@pytest.fixture(scope="session")
def prepared_data(tmp_csv: Path):
    """Result of ``prepare()`` on the synthetic CSV — computed once per session."""
    from wwtp.data_prep import prepare

    return prepare(
        csv_path=tmp_csv,
        input_cols=list(DEFAULT_INPUT_COLS),
        target_col="NH4",
        lookback=LOOKBACK,
        test_size=0.2,
        val_size=0.1,
    )


# ─── Model fixtures ───────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def mock_model():
    """Minimal Keras model (lookback, n_features) → Dense(1).

    Weights are random but deterministic (seed 0).  Not trained — used only
    to check output shapes and that pipeline plumbing works end-to-end.
    """
    tf.random.set_seed(0)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(LOOKBACK, N_FEATURES)),
            tf.keras.layers.SimpleRNN(4),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mae")
    return model


@pytest.fixture(scope="session")
def mock_scalers(rng: np.random.Generator):
    """Two MinMaxScalers fitted on random data — shape-compatible with training.

    Returns:
        Dict with keys ``"feature_scaler"`` and ``"target_scaler"``.
    """
    X_dummy = rng.uniform(0, 1, (N_ROWS, N_FEATURES))
    y_dummy = rng.uniform(0, 10, (N_ROWS, 1))

    fs = MinMaxScaler()
    ts = MinMaxScaler()
    fs.fit(X_dummy)
    ts.fit(y_dummy)

    return {"feature_scaler": fs, "target_scaler": ts}
