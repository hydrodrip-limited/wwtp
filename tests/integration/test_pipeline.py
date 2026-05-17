"""Integration test — full mini-train pipeline end-to-end.

This test runs the entire 5-step pipeline (define → compile → fit → evaluate
→ predict) with a **tiny hyper-parameter budget** (2 epochs) on the synthetic
200-row fixture dataset and confirms:

1. MLflow logs the run without errors.
2. ``rmse`` and ``mae`` are finite non-negative numbers.
3. The registered WwtpPredictor pyfunc loads and can produce predictions.

Marked ``@pytest.mark.integration`` so it can be skipped in the fast unit-test
CI stage and only runs in the dedicated integration job.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pytest

from wwtp.model import (
    EvalMetrics,
    TrainConfig,
    compile_network,
    define_network,
    evaluate_network,
    fit_network,
    make_predictions,
)
from wwtp.predictor import log_wwtp_model

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def tiny_run(prepared_data, mock_scalers):
    """Run 2 epochs of RNN training and log to a temporary MLflow store.

    Returns a dict with ``run_id``, ``metrics`` (EvalMetrics), and the
    artifact URI of the logged pyfunc model.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        mlflow.set_tracking_uri(f"file://{tmpdir}/mlruns")
        mlflow.set_experiment("integration_test")

        cfg = TrainConfig(architecture="rnn", epochs=2, batch_size=32)

        with mlflow.start_run() as run:
            model = define_network(cfg, prepared_data.lookback, prepared_data.n_features)
            compile_network(model, cfg)
            fit_network(
                model,
                prepared_data.X_train,
                prepared_data.y_train,
                prepared_data.X_val,
                prepared_data.y_val,
                cfg,
                verbose=0,
            )
            metrics, y_true, y_pred = evaluate_network(
                model,
                prepared_data.X_test,
                prepared_data.y_test,
                prepared_data.target_scaler,
            )
            mlflow.log_metrics(metrics.as_dict())

            from mlflow.models import infer_signature

            sig = infer_signature(
                prepared_data.X_test[:2],
                model.predict(prepared_data.X_test[:2], verbose=0),
            )
            log_wwtp_model(
                keras_model=model,
                scalers=mock_scalers,
                artifact_path="model",
                signature=sig,
                input_example=prepared_data.X_test[:1],
            )
            run_id = run.info.run_id

        yield {
            "run_id": run_id,
            "metrics": metrics,
            "model_uri": f"runs:/{run_id}/model",
            "tracking_uri": f"file://{tmpdir}/mlruns",
            "X_test": prepared_data.X_test,
        }


class TestMiniTrainPipeline:
    def test_rmse_is_finite(self, tiny_run: dict) -> None:
        assert np.isfinite(tiny_run["metrics"].rmse)

    def test_mae_is_non_negative(self, tiny_run: dict) -> None:
        assert tiny_run["metrics"].mae >= 0.0

    def test_mlflow_run_exists(self, tiny_run: dict) -> None:
        mlflow.set_tracking_uri(tiny_run["tracking_uri"])
        run = mlflow.get_run(tiny_run["run_id"])
        assert run.info.run_id == tiny_run["run_id"]

    def test_logged_metrics_present(self, tiny_run: dict) -> None:
        mlflow.set_tracking_uri(tiny_run["tracking_uri"])
        run = mlflow.get_run(tiny_run["run_id"])
        assert "rmse" in run.data.metrics
        assert "mae" in run.data.metrics
        assert "r2" in run.data.metrics

    def test_pyfunc_model_loads_and_predicts(self, tiny_run: dict) -> None:
        mlflow.set_tracking_uri(tiny_run["tracking_uri"])
        predictor = mlflow.pyfunc.load_model(tiny_run["model_uri"])
        X = tiny_run["X_test"][:3]
        preds = np.asarray(predictor.predict(X))
        assert preds.shape == (3,) or preds.shape == (3, 1)
        assert np.all(preds >= 0.0), "WwtpPredictor must clip to [0, ∞)"
