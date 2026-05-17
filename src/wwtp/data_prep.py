"""
Data preparation for WWTP effluent prediction.

Mirrors Section 2.2-2.3 of Wongburi & Park (2023):
  - Select influent/operating parameters as inputs, effluent param as target
  - MinMax scale to [0, 1] (scikit-learn MinMaxScaler)
  - Build supervised sliding-window sequences for RNN/LSTM input
  - Chronological 70/10/20 train/val/test split — never shuffle a time series

Adaptation: the paper used influent BOD5/TSS/TP/TKN/NH3-N + flow + SVI to
predict matching effluent parameters.  The Tilburg WWTP online dataset exposes
a different but analogous structure (per its data dictionary):

    INPUTS  (controllable / influent):  Q_inf, Q_air_1..5, Temp  [7 features]
    OUTPUTS (downstream responses):     DO_1, DO_2, DO_3, NO3, NH4

We follow this stated INPUT/OUTPUT contract exactly.  Default target is NH4
(closest analog to the paper's "effluent NH3-N").  Any of the five outputs can
be swapped in via settings; nothing else needs to change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from wwtp.config import DEFAULT_INPUT_COLS, VALID_TARGETS
from wwtp.logging_cfg import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclass
class PreparedData:
    """Holds all splits and fitted scalers produced by :func:`prepare`."""

    X_train: np.ndarray  # (n_train, lookback, n_features)
    y_train: np.ndarray  # (n_train,)
    X_val: np.ndarray  # (n_val,   lookback, n_features)
    y_val: np.ndarray  # (n_val,)
    X_test: np.ndarray  # (n_test,  lookback, n_features)
    y_test: np.ndarray  # (n_test,)
    feature_scaler: MinMaxScaler
    target_scaler: MinMaxScaler
    feature_names: list[str] = field(default_factory=list)
    target_name: str = ""
    lookback: int = 1
    val_index: pd.DatetimeIndex | None = None
    test_index: pd.DatetimeIndex | None = None

    @property
    def n_features(self) -> int:
        """Number of input features."""
        return int(self.X_train.shape[2])


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_online_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Read the tab-separated online sensor CSV and parse the timestamp.

    Args:
        csv_path: Path to the ``*_online_dataset.csv`` file.

    Returns:
        DataFrame indexed by ``DATETIME``, all columns cast to float.

    Raises:
        FileNotFoundError: If *csv_path* does not exist.
        KeyError: If the ``DATETIME`` column is absent.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    logger.info("Loading dataset from %s", path)
    df = pd.read_csv(path, sep="\t")
    df["DATETIME"] = pd.to_datetime(df["DATETIME"])
    df = df.sort_values("DATETIME").set_index("DATETIME")
    df = df.apply(pd.to_numeric, errors="coerce")
    logger.info("Loaded %d rows, %d columns", len(df), df.shape[1])
    return df


def build_windows(
    features: np.ndarray,
    target: np.ndarray,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``(n_samples, lookback, n_features)`` input tensors.

    Window *t* uses ``features[t-lookback : t]`` to predict ``target[t]``.

    Args:
        features: Scaled feature array of shape ``(n_rows, n_features)``.
        target: Scaled target array of shape ``(n_rows,)``.
        lookback: Number of past timesteps per window.

    Returns:
        Tuple ``(X, y)`` where X has shape ``(n_rows-lookback, lookback,
        n_features)`` and y has shape ``(n_rows-lookback,)``.

    Raises:
        ValueError: If ``n_rows <= lookback``.
    """
    n = len(target)
    if n <= lookback:
        raise ValueError(f"Need more than {lookback} rows to build windows, got {n}.")
    X = np.stack([features[i - lookback : i] for i in range(lookback, n)])
    y = target[lookback:]
    return X.astype("float32"), y.astype("float32")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def prepare(
    csv_path: str | Path,
    input_cols: list[str] | None = None,
    target_col: str = "NH4",
    lookback: int = 8,
    test_size: float = 0.2,
    val_size: float = 0.1,
) -> PreparedData:
    """Full data-preparation pipeline.

    Performs a **chronological** train / val / test split before fitting any
    scaler so that no future information leaks into the training fold.

    Args:
        csv_path: Path to the tab-separated online sensor CSV.
        input_cols: Feature columns.  Defaults to the seven standard inputs.
        target_col: Output column to predict.  Must be one of
            ``("DO_1", "DO_2", "DO_3", "NO3", "NH4")``.
        lookback: Number of past timesteps used as the RNN input window.
        test_size: Fraction of data reserved for the final test set.
        val_size: Fraction of data reserved for validation during training.

    Returns:
        A :class:`PreparedData` instance containing all splits and scalers.

    Raises:
        KeyError: If any required column is absent from the dataset.
        ValueError: If *target_col* is not a valid output column, or if the
            split fractions leave fewer than ``lookback + 1`` rows in any fold.
    """
    if target_col not in VALID_TARGETS:
        raise ValueError(f"target_col must be one of {VALID_TARGETS}, got {target_col!r}")

    input_cols = input_cols or list(DEFAULT_INPUT_COLS)
    df = load_online_dataset(csv_path)

    all_cols = input_cols + [target_col]
    missing = [c for c in all_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Columns missing from dataset: {missing}")

    df = df[all_cols].dropna()
    n_total = len(df)

    # Chronological split — compute boundaries once from the *total* length.
    n_test = int(n_total * test_size)
    n_val = int(n_total * val_size)
    n_train = n_total - n_val - n_test

    if n_train <= lookback or n_val <= lookback or n_test <= lookback:
        raise ValueError(
            f"One of the folds is too small for lookback={lookback}. "
            f"Sizes: train={n_train}, val={n_val}, test={n_test}."
        )

    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train : n_train + n_val]
    test_df = df.iloc[n_train + n_val :]

    logger.info(
        "Split sizes — train: %d  val: %d  test: %d",
        len(train_df),
        len(val_df),
        len(test_df),
    )

    # Fit scalers on the training fold ONLY — no data leakage.
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))

    X_train_raw = feature_scaler.fit_transform(train_df[input_cols].values)
    X_val_raw = feature_scaler.transform(val_df[input_cols].values)
    X_test_raw = feature_scaler.transform(test_df[input_cols].values)

    y_train_raw = target_scaler.fit_transform(train_df[[target_col]].values).ravel()
    y_val_raw = target_scaler.transform(val_df[[target_col]].values).ravel()
    y_test_raw = target_scaler.transform(test_df[[target_col]].values).ravel()

    X_train, y_train = build_windows(X_train_raw, y_train_raw, lookback)
    X_val, y_val = build_windows(X_val_raw, y_val_raw, lookback)
    X_test, y_test = build_windows(X_test_raw, y_test_raw, lookback)

    logger.info(
        "Window shapes — X_train: %s  X_val: %s  X_test: %s",
        X_train.shape,
        X_val.shape,
        X_test.shape,
    )

    return PreparedData(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        feature_names=input_cols,
        target_name=target_col,
        lookback=lookback,
        val_index=val_df.index[lookback:],
        test_index=test_df.index[lookback:],
    )
