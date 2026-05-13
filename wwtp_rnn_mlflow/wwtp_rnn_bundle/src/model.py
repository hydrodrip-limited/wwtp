"""
Effluent prediction models — Simple RNN and LSTM.

Mirrors Section 2.4 of Wongburi & Park (2023), Figures 4 and 5:
    Step 1: Define Network
    Step 2: Compile Network
    Step 3: Fit Network
    Step 4: Evaluate Network
    Step 5: Make Predictions

Default hyperparameters follow the paper's "optimal" setting:
    epochs = 50, batch_size = 100, optimizer = Adam, loss = MAE.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.layers import LSTM, Dense, Input, SimpleRNN
from tensorflow.keras.models import Sequential


@dataclass
class TrainConfig:
    architecture: str = "lstm"      # "rnn" or "lstm"
    units: int = 50                  # paper uses 50 for SimpleRNN and 100 for LSTM in code snippets
    epochs: int = 50                 # paper's optimal
    batch_size: int = 100            # paper's optimal
    optimizer: str = "adam"
    loss: str = "mae"
    shuffle: bool = False            # time series — never shuffle
    seed: int = 42


@dataclass
class EvalMetrics:
    rmse: float
    mae: float
    r2: float

    def as_dict(self) -> dict[str, float]:
        return {"rmse": self.rmse, "mae": self.mae, "r2": self.r2}


# -------------------- Step 1: Define Network --------------------
def define_network(cfg: TrainConfig, lookback: int, n_features: int) -> tf.keras.Model:
    """Build a sequential RNN or LSTM with a single Dense(1) head."""
    tf.keras.utils.set_random_seed(cfg.seed)

    model = Sequential(name=f"{cfg.architecture}_effluent_predictor")
    model.add(Input(shape=(lookback, n_features)))
    if cfg.architecture.lower() == "rnn":
        model.add(SimpleRNN(cfg.units))
    elif cfg.architecture.lower() == "lstm":
        model.add(LSTM(cfg.units))
    else:
        raise ValueError(f"Unknown architecture: {cfg.architecture}")
    model.add(Dense(1))
    return model


# -------------------- Step 2: Compile Network --------------------
def compile_network(model: tf.keras.Model, cfg: TrainConfig) -> tf.keras.Model:
    """Compile with Adam + MAE loss, exactly as in the paper's Figure 4/5 code blocks."""
    model.compile(optimizer=cfg.optimizer, loss=cfg.loss)
    return model


# -------------------- Step 3: Fit Network --------------------
def fit_network(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: TrainConfig,
    verbose: int = 2,
) -> tf.keras.callbacks.History:
    """Fit on training data, validate on the held-out test set (sequence — no shuffle)."""
    return model.fit(
        X_train, y_train,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        validation_data=(X_test, y_test),
        verbose=verbose,
        shuffle=cfg.shuffle,
    )


# -------------------- Step 4: Evaluate Network --------------------
def evaluate_network(
    model: tf.keras.Model,
    X: np.ndarray,
    y_scaled: np.ndarray,
    target_scaler,
) -> tuple[EvalMetrics, np.ndarray, np.ndarray]:
    """
    Score the network on (X, y_scaled). Inverse-transforms back to original units
    before computing RMSE/MAE/R^2 so metrics are interpretable in mg/L.
    """
    y_pred_scaled = model.predict(X, verbose=0).ravel()
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_true = target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).ravel()

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return EvalMetrics(rmse=rmse, mae=mae, r2=r2), y_true, y_pred


# -------------------- Step 5: Make Predictions --------------------
def make_predictions(
    model: tf.keras.Model,
    X: np.ndarray,
    target_scaler,
) -> np.ndarray:
    """Predict and inverse-scale back to the target's original units."""
    y_pred_scaled = model.predict(X, verbose=0).ravel()
    return target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
