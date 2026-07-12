"""Configuration schema and loader for the TNN STOXX600 forecasting project."""

from pathlib import Path

import yaml
from pydantic import BaseModel


class DataConfig(BaseModel):
    """Configuration for data downloading and preprocessing."""

    start_date: str
    end_date: str
    train_test_split: float
    sequence_length: int


class LSTMConfig(BaseModel):
    """Configuration for the LSTM model."""

    hidden_size: int
    num_layers: int
    dropout: float


class TNNConfig(BaseModel):
    """Configuration for the TNN model."""

    kernel_output_size: int
    kernel_size: int
    hidden_size: int
    dropout: float


class ARIMAConfig(BaseModel):
    """Configuration for the ARIMA model."""

    order: tuple[int, int, int]


class ModelsConfig(BaseModel):
    """Configuration for all models."""

    lstm: LSTMConfig
    tnn: TNNConfig
    arima: ARIMAConfig


class TrainingConfig(BaseModel):
    """Configuration for the training loop."""

    horizons: list[int]
    epochs: int
    learning_rate: float
    batch_size: int


class MLflowConfig(BaseModel):
    """Configuration for MLflow experiment tracking."""

    experiment_name: str
    tracking_uri: str


class Config(BaseModel):
    """Root configuration schema for the project."""

    data: DataConfig
    models: ModelsConfig
    training: TrainingConfig
    mlflow: MLflowConfig


def load_config(path: Path = Path("configs/config.yaml")) -> Config:
    """Load and validate configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Validated Config instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    return Config.model_validate(raw)
