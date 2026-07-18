"""Unit tests for model loading and inference."""

import numpy as np
import pytest
import torch

from src.config import LSTMConfig, TNNConfig
from src.models.loaders import load_lstm, load_tnn, predict_pytorch
from src.models.lstm import LSTMModel
from src.models.tnn import TNN


@pytest.fixture
def lstm_config() -> LSTMConfig:
    """Small LSTM configuration for fast tests."""
    return LSTMConfig(hidden_size=8, num_layers=1, dropout=0.0, learning_rate=0.001)


@pytest.fixture
def tnn_config() -> TNNConfig:
    """Small TNN configuration for fast tests."""
    return TNNConfig(
        kernel_output_size=8,
        kernel_size=2,
        hidden_size=8,
        dropout=0.2,
        learning_rate=0.001,
    )


def test_load_lstm_restores_weights(tmp_path, lstm_config: LSTMConfig) -> None:
    """Loaded weights should match those that were saved."""
    original = LSTMModel(input_size=4, hidden_size=8, num_layers=1, dropout=0.0)
    path = tmp_path / "lstm.pt"
    torch.save(original.state_dict(), path)

    loaded = load_lstm(path, input_size=4, model_config=lstm_config)

    for key, value in original.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key])


def test_load_lstm_returns_eval_mode(tmp_path, lstm_config: LSTMConfig) -> None:
    """Loaded models must be in eval mode so inference is deterministic."""
    path = tmp_path / "lstm.pt"
    torch.save(
        LSTMModel(input_size=4, hidden_size=8, num_layers=1, dropout=0.0).state_dict(),
        path,
    )
    assert not load_lstm(path, 4, lstm_config).training


def test_load_tnn_returns_eval_mode(tmp_path, tnn_config: TNNConfig) -> None:
    """The TNN loader should also return a model in eval mode."""
    original = TNN(
        input_size=4, kernel_output_size=8, kernel_size=2, hidden_size=8, dropout=0.2
    )
    path = tmp_path / "tnn.pt"
    torch.save(original.state_dict(), path)
    assert not load_tnn(path, 4, tnn_config).training


def test_load_raises_on_missing_file(tmp_path, lstm_config: LSTMConfig) -> None:
    """Loading a non-existent checkpoint should raise rather than fail silently."""
    with pytest.raises(FileNotFoundError):
        load_lstm(tmp_path / "nope.pt", 4, lstm_config)


def test_predict_returns_one_value_per_sample() -> None:
    """Predictions should be 1-D with one entry per input sample."""
    model = LSTMModel(input_size=4, hidden_size=8, num_layers=1, dropout=0.0)
    model.eval()
    X = np.random.default_rng(0).random((5, 3, 4)).astype(np.float32)

    preds = predict_pytorch(model, X)

    assert preds.shape == (5,)


def test_predict_handles_single_sample() -> None:
    """A single window should still yield a 1-D array, not a scalar."""
    model = LSTMModel(input_size=4, hidden_size=8, num_layers=1, dropout=0.0)
    model.eval()
    X = np.random.default_rng(0).random((1, 3, 4)).astype(np.float32)

    preds = predict_pytorch(model, X)

    assert preds.shape == (1,)
    assert float(preds[0]) == pytest.approx(float(preds[0]))


def test_predict_is_deterministic_in_eval_mode(tnn_config: TNNConfig) -> None:
    """Dropout must be inactive, so repeated calls give identical results."""
    model = TNN(
        input_size=4, kernel_output_size=8, kernel_size=2, hidden_size=8, dropout=0.5
    )
    model.eval()
    X = np.random.default_rng(0).random((4, 3, 4)).astype(np.float32)

    assert np.array_equal(predict_pytorch(model, X), predict_pytorch(model, X))
