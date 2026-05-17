"""Unit tests for ``wwtp.model``."""

from __future__ import annotations

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

LOOKBACK = 8
N_FEATURES = 7


class TestDefineNetwork:
    def test_rnn_output_shape(self) -> None:
        cfg = TrainConfig(architecture="rnn")
        model = define_network(cfg, LOOKBACK, N_FEATURES)
        assert model.output_shape == (None, 1)

    def test_lstm_output_shape(self) -> None:
        cfg = TrainConfig(architecture="lstm")
        model = define_network(cfg, LOOKBACK, N_FEATURES)
        assert model.output_shape == (None, 1)

    def test_invalid_architecture_raises(self) -> None:
        with pytest.raises(ValueError):
            cfg = TrainConfig(architecture="transformer")  # type: ignore[arg-type]
            define_network(cfg, LOOKBACK, N_FEATURES)


class TestFitNetwork:
    def test_returns_history_with_val_loss(self, mock_model, prepared_data) -> None:
        """fit_network must train against the validation set, not test set."""
        cfg = TrainConfig(epochs=1, batch_size=32)
        # Recompile the fixture model to avoid polluted state
        compile_network(mock_model, cfg)
        history = fit_network(
            mock_model,
            prepared_data.X_train[:40],
            prepared_data.y_train[:40],
            prepared_data.X_val[:10],
            prepared_data.y_val[:10],
            cfg,
            verbose=0,
        )
        assert "val_loss" in history.history
        assert "loss" in history.history


class TestEvaluateNetwork:
    def test_output_contains_expected_keys(self, mock_model, prepared_data) -> None:
        metrics, y_true, y_pred = evaluate_network(
            mock_model,
            prepared_data.X_test,
            prepared_data.y_test,
            prepared_data.target_scaler,
        )
        keys = set(metrics.as_dict())
        assert {"rmse", "mae", "r2"}.issubset(keys)

    def test_predictions_are_non_negative(self, mock_model, prepared_data) -> None:
        """Clipping must ensure all predicted concentrations ≥ 0."""
        _, _, y_pred = evaluate_network(
            mock_model,
            prepared_data.X_test,
            prepared_data.y_test,
            prepared_data.target_scaler,
        )
        assert np.all(y_pred >= 0.0), "Negative predictions found — clipping failed"

    def test_y_true_and_y_pred_same_length(self, mock_model, prepared_data) -> None:
        _, y_true, y_pred = evaluate_network(
            mock_model,
            prepared_data.X_test,
            prepared_data.y_test,
            prepared_data.target_scaler,
        )
        assert len(y_true) == len(y_pred)


class TestMakePredictions:
    def test_shape_is_1d(self, mock_model, prepared_data) -> None:
        preds = make_predictions(mock_model, prepared_data.X_test, prepared_data.target_scaler)
        assert preds.ndim == 1
        assert len(preds) == len(prepared_data.X_test)

    def test_clipped_to_zero(self, mock_model, prepared_data) -> None:
        preds = make_predictions(mock_model, prepared_data.X_test, prepared_data.target_scaler)
        assert np.all(preds >= 0.0)


class TestEvalMetrics:
    def test_as_dict_keys(self) -> None:
        m = EvalMetrics(rmse=1.0, mae=0.5, r2=0.9)
        d = m.as_dict()
        assert set(d) == {"rmse", "mae", "r2"}

    def test_values_are_float(self) -> None:
        m = EvalMetrics(rmse=1.0, mae=0.5, r2=0.9)
        for v in m.as_dict().values():
            assert isinstance(v, float)
