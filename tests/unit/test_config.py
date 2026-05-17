"""Unit tests for ``wwtp.config``."""

from __future__ import annotations

import os

import pytest
from pydantic import ValidationError

from wwtp.config import ModelConfig, Settings


class TestModelConfigDefaults:
    def test_default_lookback_is_8(self) -> None:
        cfg = ModelConfig()
        assert cfg.lookback == 8, "Default lookback must be 8 (paper's optimal value)"

    def test_default_architecture_is_rnn(self) -> None:
        cfg = ModelConfig()
        assert cfg.architecture == "rnn"

    def test_default_epochs(self) -> None:
        cfg = ModelConfig()
        assert cfg.epochs == 50

    def test_default_batch_size(self) -> None:
        cfg = ModelConfig()
        assert cfg.batch_size == 100


class TestModelConfigValidation:
    def test_rejects_lookback_zero(self) -> None:
        with pytest.raises(ValidationError):
            ModelConfig(lookback=0)

    def test_rejects_lookback_negative(self) -> None:
        with pytest.raises(ValidationError):
            ModelConfig(lookback=-5)

    def test_rejects_epochs_zero(self) -> None:
        with pytest.raises(ValidationError):
            ModelConfig(epochs=0)

    def test_rejects_batch_size_zero(self) -> None:
        with pytest.raises(ValidationError):
            ModelConfig(batch_size=0)

    def test_valid_architecture_lstm(self) -> None:
        cfg = ModelConfig(architecture="lstm")
        assert cfg.architecture == "lstm"

    def test_invalid_architecture_raises(self) -> None:
        with pytest.raises(ValidationError):
            ModelConfig(architecture="transformer")


class TestSettings:
    def test_default_settings_create_without_error(self) -> None:
        s = Settings()
        assert s is not None

    def test_env_var_overrides_lookback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MODEL__LOOKBACK", "16")
        s = Settings()
        assert s.model.lookback == 16

    def test_env_var_overrides_target_col(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DATA__TARGET_COL", "DO_1")
        s = Settings()
        assert s.data.target_col == "DO_1"

    def test_invalid_target_col_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DATA__TARGET_COL", "PHOSPHORUS")
        with pytest.raises(ValidationError):
            Settings()

    def test_shortcut_properties_return_strings(self) -> None:
        s = Settings()
        assert isinstance(s.mlflow_tracking_uri, str)
        assert isinstance(s.experiment_name, str)
        assert isinstance(s.registered_model, str)
