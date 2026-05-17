"""
RNN / LSTM model definition, compilation, training and evaluation.

Mirrors Section 2.4 of Wongburi & Park (2023), Figures 4 and 5:
    Step 1: Define Network
    Step 2: Compile Network
    Step 3: Fit Network
    Step 4: Evaluate Network
    Step 5: Make Predictions

Default hyper-parameters follow the paper's "optimal" setting:
    epochs=50, batch_size=100, optimizer=Adam, loss=MAE.

Key fixes vs the original code
-------------------------------
* ``fit_network`` now uses a **separate validation set** (not the test set)
  so that test-set metrics are true held-out generalization scores.
* All inverse-scaled predictions are clipped to ``[0, ∞)`` because NH4 and
  dissolved-oxygen concentrations are physically non-negative.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Input, SimpleRNN
from tensorflow.keras.models import Sequential

from wwtp.logging_cfg import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration and result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """Hyper-parameter bundle for :func:`define_network` and :func:`fit_network`."""

    architecture: str = "rnn"  # "rnn" or "lstm"
    units: int = 50
    epochs: int = 50
    batch_size: int = 100
    optimizer: str = "adam"
    loss: str = "mae"
    shuffle: bool = False  # time series — never shuffle
    seed: int = 42


@dataclass
class EvalMetrics:
    """Evaluation metrics in the target's original physical units (mg/L)."""

    rmse: float
    mae: float
    r2: float

    def as_dict(self) -> dict[str, float]:
        """Return metrics as a plain dict suitable for MLflow logging."""
        return {"rmse": self.rmse, "mae": self.mae, "r2": self.r2}


# ---------------------------------------------------------------------------
# Step 1: Define Network
# ---------------------------------------------------------------------------


def define_network(cfg: TrainConfig, lookback: int, n_features: int) -> tf.keras.Model:
    """Build a sequential SimpleRNN or LSTM with a single ``Dense(1)`` head.

    Args:
        cfg: Hyper-parameter configuration.
        lookback: Number of past timesteps per input window.
        n_features: Number of input features per timestep.

    Returns:
        An uncompiled :class:`tf.keras.Model`.

    Raises:
        ValueError: If ``cfg.architecture`` is not ``"rnn"`` or ``"lstm"``.
    """
    tf.keras.utils.set_random_seed(cfg.seed)

    model = Sequential(name=f"{cfg.architecture}_effluent_predictor")
    model.add(Input(shape=(lookback, n_features)))

    if cfg.architecture.lower() == "rnn":
        model.add(SimpleRNN(cfg.units))
    elif cfg.architecture.lower() == "lstm":
        model.add(LSTM(cfg.units))
    else:
        raise ValueError(f"Unknown architecture {cfg.architecture!r}. Expected 'rnn' or 'lstm'.")

    model.add(Dense(1))
    logger.info(
        "Defined %s model — lookback=%d  n_features=%d  units=%d",
        cfg.architecture.upper(),
        lookback,
        n_features,
        cfg.units,
    )
    return model


# ---------------------------------------------------------------------------
# Step 2: Compile Network
# ---------------------------------------------------------------------------


def compile_network(model: tf.keras.Model, cfg: TrainConfig) -> tf.keras.Model:
    """Compile with Adam + MAE loss, exactly as in the paper's Figure 4/5 code blocks.

    Args:
        model: Uncompiled Keras model.
        cfg: Hyper-parameter configuration.

    Returns:
        The compiled model (same object, mutated in-place).
    """
    model.compile(optimizer=cfg.optimizer, loss=cfg.loss)
    return model


# ---------------------------------------------------------------------------
# Step 3: Fit Network
# ---------------------------------------------------------------------------


def fit_network(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: TrainConfig,
    verbose: int = 0,
) -> tf.keras.callbacks.History:
    """Fit on the training set, validate on the **validation** set.

    The validation set must be a distinct chronological slice that was held out
    *before* fitting any scaler.  Passing the test set here would contaminate
    the final evaluation metrics.

    Args:
        model: Compiled Keras model.
        X_train: Training inputs ``(n_train, lookback, n_features)``.
        y_train: Training targets ``(n_train,)``.
        X_val: Validation inputs — **not** the test set.
        y_val: Validation targets — **not** the test set.
        cfg: Hyper-parameter configuration.
        verbose: Keras verbosity level (0 = silent).

    Returns:
        Keras :class:`History` object.
    """
    return model.fit(
        X_train,
        y_train,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        validation_data=(X_val, y_val),
        verbose=verbose,
        shuffle=cfg.shuffle,
    )


# ---------------------------------------------------------------------------
# Step 4: Evaluate Network
# ---------------------------------------------------------------------------


def evaluate_network(
    model: tf.keras.Model,
    X: np.ndarray,
    y_scaled: np.ndarray,
    target_scaler: MinMaxScaler,
) -> tuple[EvalMetrics, np.ndarray, np.ndarray]:
    """Score the network and return metrics in the target's original units.

    Predictions are clipped to ``[0, ∞)`` after inverse-scaling because water
    quality concentrations are physically non-negative.

    Args:
        model: Trained Keras model.
        X: Input windows ``(n, lookback, n_features)``.
        y_scaled: Scaled ground-truth targets ``(n,)``.
        target_scaler: Fitted :class:`MinMaxScaler` for the target column.

    Returns:
        Tuple ``(metrics, y_true_unscaled, y_pred_unscaled)``.
    """
    y_pred_scaled = model.predict(X, verbose=0).ravel()
    y_pred = np.clip(
        target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel(),
        0.0,
        None,
    )
    y_true = target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).ravel()

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    logger.info("Evaluation — RMSE=%.4f  MAE=%.4f  R²=%.4f", rmse, mae, r2)
    return EvalMetrics(rmse=rmse, mae=mae, r2=r2), y_true, y_pred


# ---------------------------------------------------------------------------
# Step 5: Make Predictions
# ---------------------------------------------------------------------------


def make_predictions(
    model: tf.keras.Model,
    X: np.ndarray,
    target_scaler: MinMaxScaler,
) -> np.ndarray:
    """Predict and inverse-scale back to the target's original physical units.

    Args:
        model: Trained Keras model.
        X: Input windows ``(n, lookback, n_features)``.
        target_scaler: Fitted :class:`MinMaxScaler` for the target column.

    Returns:
        Array of predictions in original units (mg/L), clipped to ``[0, ∞)``.
    """
    y_pred_scaled = model.predict(X, verbose=0).ravel()
    return np.clip(
        target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel(),
        0.0,
        None,
    )
