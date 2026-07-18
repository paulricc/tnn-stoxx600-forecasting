"""Loading trained models from disk and running inference."""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.config import LSTMConfig, TNNConfig
from src.models.lstm import LSTMModel
from src.models.tnn import TNN

logger = logging.getLogger(__name__)


def load_lstm(
    path: Path,
    input_size: int,
    model_config: LSTMConfig,
) -> LSTMModel:
    """Load a trained LSTM from a saved state dict.

    The architecture is rebuilt from the configuration before the weights are
    applied, since only parameters are persisted, not the model structure.

    Args:
        path: Path to the saved state dict.
        input_size: Number of input features per time step.
        model_config: LSTM section of the project configuration.

    Returns:
        Loaded model in eval mode.
    """
    model = LSTMModel(
        input_size=input_size,
        hidden_size=model_config.hidden_size,
        num_layers=model_config.num_layers,
        dropout=model_config.dropout,
    )
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    logger.info("Loaded LSTM from %s", path)
    return model


def load_tnn(
    path: Path,
    input_size: int,
    model_config: TNNConfig,
) -> TNN:
    """Load a trained TNN from a saved state dict.

    Args:
        path: Path to the saved state dict.
        input_size: Number of input features per time step.
        model_config: TNN section of the project configuration.

    Returns:
        Loaded model in eval mode.
    """
    model = TNN(
        input_size=input_size,
        kernel_output_size=model_config.kernel_output_size,
        kernel_size=model_config.kernel_size,
        hidden_size=model_config.hidden_size,
        dropout=model_config.dropout,
    )
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    logger.info("Loaded TNN from %s", path)
    return model


def predict_pytorch(model: nn.Module, X: np.ndarray) -> np.ndarray:
    """Run inference with a PyTorch model.

    Args:
        model: Trained model, expected to be in eval mode.
        X: Input array of shape (n_samples, sequence_length, n_features).

    Returns:
        Predictions of shape (n_samples,).
    """
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        return model(X_tensor).squeeze(-1).numpy()
