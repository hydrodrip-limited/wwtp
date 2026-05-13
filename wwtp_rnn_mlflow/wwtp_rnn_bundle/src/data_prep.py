"""
Data preparation for WWTP effluent prediction.

Mirrors Section 2.2-2.3 of Wongburi & Park (2023):
  - Select influent/operating parameters as inputs, effluent param as target
  - MinMax scale to [0, 1] (scikit-learn MinMaxScaler)
  - Build supervised (t-1 -> t) sliding-window sequences for RNN/LSTM
  - 80% train / 20% test split, no shuffle (time series)

Adaptation: the paper used influent BOD5/TSS/TP/TKN/NH3-N + flow + SVI to
predict the matching effluent parameters. The Tilburg WWTP online dataset
exposes a different but analogous structure (per its data dictionary):

    INPUTS  (controllable / influent):  Q_inf, Q_air_1..5, Temp
    OUTPUTS (downstream responses):     DO_1, DO_2, DO_3, NO3, NH4

We follow this stated INPUT/OUTPUT contract exactly. By default the target
is the end-of-aerobic-zone NH4 concentration — the closest analog to the
paper's "effluent NH3-N". The other outputs (DO_1..3, NO3) can be swapped in
via the --target flag; nothing else needs to change.

The framing (sequence-to-one supervised regression, MinMax scaling on the
training fold only, 80/20 chronological split, no shuffle) is identical to
the paper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# INPUTS per the dataset's data dictionary — controllable / influent variables only.
DEFAULT_INPUT_COLS = [
    "Q_inf",                                            # total influent flow [m3/h]
    "Q_air_1", "Q_air_2", "Q_air_3", "Q_air_4", "Q_air_5",  # air flow rates [m3/h]
    "Temp",                                             # end-of-aerobic-zone temperature [degC]
]
# OUTPUTS available: DO_1, DO_2, DO_3, NO3, NH4. Default to NH4 (paper analog).
DEFAULT_TARGET_COL = "NH4"
VALID_TARGETS = ("DO_1", "DO_2", "DO_3", "NO3", "NH4")


@dataclass
class PreparedData:
    X_train: np.ndarray              # shape (n_train, lookback, n_features)
    y_train: np.ndarray              # shape (n_train,)
    X_test: np.ndarray
    y_test: np.ndarray
    feature_scaler: MinMaxScaler     # fit on training inputs
    target_scaler: MinMaxScaler      # fit on training target (for inverse transform)
    feature_names: list[str] = field(default_factory=list)
    target_name: str = ""
    lookback: int = 1
    test_index: pd.DatetimeIndex | None = None   # timestamps for plotting test predictions

    @property
    def n_features(self) -> int:
        return self.X_train.shape[2]


def load_online_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Read the tab-separated online sensor CSV and parse the timestamp."""
    df = pd.read_csv(csv_path, sep="\t")
    df["DATETIME"] = pd.to_datetime(df["DATETIME"])
    df = df.sort_values("DATETIME").set_index("DATETIME")
    # Coerce numerics; the file is already numeric but be defensive
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def build_windows(
    features: np.ndarray,
    target: np.ndarray,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (n_samples, lookback, n_features) input tensors and (n_samples,) targets.
    Window t uses features[t-lookback : t] to predict target[t].
    """
    n = len(target)
    if n <= lookback:
        raise ValueError(f"Need more than {lookback} rows, got {n}")
    X = np.stack([features[i - lookback : i] for i in range(lookback, n)])
    y = target[lookback:]
    return X.astype("float32"), y.astype("float32")


def prepare(
    csv_path: str | Path,
    input_cols: list[str] = None,
    target_col: str = DEFAULT_TARGET_COL,
    lookback: int = 1,
    test_size: float = 0.2,
) -> PreparedData:
    """Full data prep pipeline. Returns a PreparedData container."""
    input_cols = input_cols or DEFAULT_INPUT_COLS
    df = load_online_dataset(csv_path)

    cols = input_cols + [target_col]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Columns missing from dataset: {missing}")

    df = df[cols].dropna()

    # Chronological 80/20 split BEFORE scaling — never let test data leak into the scaler
    split = int(len(df) * (1 - test_size))
    train_df, test_df = df.iloc[:split], df.iloc[split:]

    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))

    X_train_raw = feature_scaler.fit_transform(train_df[input_cols].values)
    X_test_raw = feature_scaler.transform(test_df[input_cols].values)

    y_train_raw = target_scaler.fit_transform(train_df[[target_col]].values).ravel()
    y_test_raw = target_scaler.transform(test_df[[target_col]].values).ravel()

    X_train, y_train = build_windows(X_train_raw, y_train_raw, lookback)
    X_test, y_test = build_windows(X_test_raw, y_test_raw, lookback)

    return PreparedData(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        feature_names=input_cols,
        target_name=target_col,
        lookback=lookback,
        test_index=test_df.index[lookback:],
    )
