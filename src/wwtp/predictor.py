"""Custom MLflow pyfunc model that bundles the Keras model with its scalers.

Bundling scalers into the same MLflow artifact eliminates the fragile
pattern of searching for a ``*_scalers.joblib`` file by run_id at inference
time.  A single ``mlflow.pyfunc.load_model()`` call is all that is needed.

Architecture
------------
::

    artifacts/
        model/          ← TensorFlow SavedModel
        scalers.joblib  ← {"feature_scaler": ..., "target_scaler": ...}

The :class:`WwtpPredictor` ``predict`` method handles the full pipeline:
  1. Receive a float32 array of shape ``(n, lookback, n_features)``
     (already scaled OR raw, depending on ``scaled`` flag).
  2. If raw: MinMax-scale the features.
  3. Run inference through the Keras model.
  4. Inverse-scale the output.
  5. Clip to ``[0, ∞)`` (concentrations are physically non-negative).

Usage — saving
--------------
    from wwtp.predictor import log_wwtp_model

    with mlflow.start_run():
        log_wwtp_model(
            keras_model=model,
            scalers={"feature_scaler": fs, "target_scaler": ts},
            artifact_path="model",
            signature=signature,
            input_example=X_test[:1],
        )

Usage — loading
---------------
    import mlflow

    predictor = mlflow.pyfunc.load_model("models:/wwtp_effluent_predictor/latest")
    nh4_preds = predictor.predict(X_windows)   # raw sensor windows → mg/L
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.pyfunc
import mlflow.tensorflow
import numpy as np
import pandas as pd

from wwtp.logging_cfg import get_logger

logger = get_logger(__name__)

_SCALERS_ARTIFACT = "scalers.joblib"
_TF_MODEL_SUBDIR = "tf_model.keras"


class WwtpPredictor(mlflow.pyfunc.PythonModel):
    """MLflow PythonModel that wraps a Keras model + fitted scalers.

    The model is loaded lazily in :meth:`load_context` so that it can be
    used with ``mlflow models serve`` without any code changes.
    """

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Load the TensorFlow model and scalers from the artifact store.

        Args:
            context: MLflow context providing paths to logged artifacts.
        """
        import tensorflow as tf

        tf_path = context.artifacts[_TF_MODEL_SUBDIR]
        scalers_path = context.artifacts[_SCALERS_ARTIFACT]

        logger.info("Loading TF model from %s", tf_path)
        self._model = tf.keras.models.load_model(tf_path)

        logger.info("Loading scalers from %s", scalers_path)
        self._scalers: dict = joblib.load(scalers_path)

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,  # noqa: ARG002
        model_input: np.ndarray | pd.DataFrame,
        params: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> np.ndarray:
        """Run the full inference pipeline.

        Args:
            context: MLflow context (unused at predict time).
            model_input: Array of shape ``(n, lookback, n_features)``
                containing **raw** sensor readings in physical units, or a
                pre-scaled array if *scaled=True* was passed as a parameter.
            params: Optional dict.  Set ``{"scaled": True}`` to skip the
                feature-scaling step (e.g. when calling from the HTTP client
                that pre-scales its inputs).

        Returns:
            Float32 array of shape ``(n,)`` with NH₄ predictions in mg/L,
            clipped to ``[0, ∞)``.
        """
        scaled_input = (params or {}).get("scaled", False)
        arr = np.asarray(model_input, dtype="float32")
        n, lb, nf = arr.shape

        if not scaled_input:
            arr = (
                self._scalers["feature_scaler"]
                .transform(arr.reshape(-1, nf))
                .reshape(n, lb, nf)
                .astype("float32")
            )

        raw_preds = self._model.predict(arr, verbose=0).ravel()
        return np.clip(
            self._scalers["target_scaler"].inverse_transform(raw_preds.reshape(-1, 1)).ravel(),
            0.0,
            None,
        ).astype("float32")


# ---------------------------------------------------------------------------
# Convenience logging helper
# ---------------------------------------------------------------------------


def log_wwtp_model(
    keras_model: Any,
    scalers: dict,
    artifact_path: str = "model",
    signature: Any = None,
    input_example: np.ndarray | None = None,
) -> mlflow.models.model.ModelInfo:
    """Log a :class:`WwtpPredictor` to the active MLflow run.

    Bundles the TensorFlow SavedModel and the fitted scalers into a single
    MLflow artifact so that loading is a one-liner at inference time.

    Args:
        keras_model: Trained ``tf.keras.Model`` instance.
        scalers: Dict with keys ``"feature_scaler"`` and ``"target_scaler"``.
        artifact_path: Sub-path within the MLflow run artifact store.
        signature: Optional :class:`mlflow.models.ModelSignature`.
        input_example: Optional numpy array for the input example.

    Returns:
        :class:`mlflow.models.model.ModelInfo` from the log call.

    Raises:
        KeyError: If *scalers* is missing ``"feature_scaler"`` or
            ``"target_scaler"``.
    """
    for key in ("feature_scaler", "target_scaler"):
        if key not in scalers:
            raise KeyError(f"scalers dict must contain {key!r}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # ── Save TF model ────────────────────────────────────────────────────
        tf_dir = tmp_path / _TF_MODEL_SUBDIR
        keras_model.save(str(tf_dir))

        # ── Save scalers ─────────────────────────────────────────────────────
        scalers_path = tmp_path / _SCALERS_ARTIFACT
        joblib.dump(scalers, scalers_path)

        artifacts = {
            _TF_MODEL_SUBDIR: str(tf_dir),
            _SCALERS_ARTIFACT: str(scalers_path),
        }

        info = mlflow.pyfunc.log_model(
            artifact_path=artifact_path,
            python_model=WwtpPredictor(),
            artifacts=artifacts,
            signature=signature,
            input_example=input_example,
            pip_requirements=[
                f"tensorflow=={_tf_version()}",
                "scikit-learn",
                "joblib",
                "numpy",
            ],
        )
        logger.info("Logged WwtpPredictor to artifact path %r", artifact_path)
        return info


def _tf_version() -> str:  # pragma: no cover
    try:
        import tensorflow as tf

        return tf.__version__
    except ImportError:
        return "2.16.0"
