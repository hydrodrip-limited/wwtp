"""
Centralized configuration for the WWTP RNN project.

All runtime parameters are defined here as Pydantic models.
Values can be overridden via environment variables or a .env file,
following the twelve-factor app methodology.

Usage
-----
    from wwtp.config import Settings

    settings = Settings()
    print(settings.model.lookback)   # 8
    print(settings.inference.api_url)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Sub-configs (pure Pydantic BaseModel — no env binding)
# ---------------------------------------------------------------------------

from pydantic import BaseModel


VALID_TARGETS: tuple[str, ...] = ("DO_1", "DO_2", "DO_3", "NO3", "NH4")

DEFAULT_INPUT_COLS: list[str] = [
    "Q_inf",
    "Q_air_1",
    "Q_air_2",
    "Q_air_3",
    "Q_air_4",
    "Q_air_5",
    "Temp",
]


class DataConfig(BaseModel):
    """Parameters that govern data loading and pre-processing."""

    input_cols: list[str] = Field(default_factory=lambda: list(DEFAULT_INPUT_COLS))
    target_col: str = "NH4"
    test_size: float = Field(default=0.2, ge=0.05, le=0.4)
    val_size: float = Field(default=0.1, ge=0.05, le=0.3)

    @field_validator("target_col")
    @classmethod
    def target_must_be_valid(cls, v: str) -> str:
        if v not in VALID_TARGETS:
            raise ValueError(f"target_col must be one of {VALID_TARGETS}, got {v!r}")
        return v

    @model_validator(mode="after")
    def sizes_must_not_overlap(self) -> "DataConfig":
        if self.test_size + self.val_size >= 1.0:
            raise ValueError("test_size + val_size must be < 1.0")
        return self


class ModelConfig(BaseModel):
    """Hyper-parameters for the RNN / LSTM model."""

    architecture: Literal["rnn", "lstm"] = "rnn"
    units: int = Field(default=50, ge=1, le=2048)
    lookback: int = Field(default=8, ge=1, le=512)
    epochs: int = Field(default=50, ge=1, le=10_000)
    batch_size: int = Field(default=100, ge=1, le=65_536)
    optimizer: str = "adam"
    loss: str = "mae"
    seed: int = Field(default=42, ge=0)


class InferenceConfig(BaseModel):
    """Settings for the prediction HTTP client and Model Registry."""

    model_name: str = "wwtp_effluent_predictor"
    model_version: str = "latest"
    api_url: str = "http://localhost:5000/invocations"
    api_timeout: int = Field(default=30, ge=1, le=300)
    chunk_size: int = Field(default=512, ge=1)


class LoggingConfig(BaseModel):
    """Logging behaviour."""

    format: Literal["human", "json"] = "human"
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"


# ---------------------------------------------------------------------------
# Root Settings — reads from env vars / .env file
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """
    Root settings object.  Reads from environment variables (case-insensitive)
    and an optional .env file.  Sub-configs are populated from prefixed vars:

        DATA__TARGET_COL=NH4
        MODEL__LOOKBACK=8
        INFERENCE__API_URL=http://mlflow:5000/invocations
        LOGGING__FORMAT=json
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # Top-level shortcuts for convenience (also readable from flat env vars)
    mlflow_tracking_uri: str = "file:./mlruns"
    experiment_name: str = "wwtp_effluent_prediction"
    registered_model: str = "wwtp_effluent_predictor"
    artifact_dir: Path = Path("artifacts")

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Settings(target={self.data.target_col!r}, "
            f"lookback={self.model.lookback}, "
            f"epochs={self.model.epochs}, "
            f"tracking_uri={self.mlflow_tracking_uri!r})"
        )
