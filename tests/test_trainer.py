"""Unit tests for the training utilities."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.training.trainer import (
    evaluate_pytorch_model,
    make_dataloader,
    train_pytorch_model,
)


class TinyModel(nn.Module):
    """Minimal model mapping a sequence to a single value, for fast tests."""

    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x[:, -1, :])


@pytest.fixture
def sample_data() -> tuple[np.ndarray, np.ndarray]:
    """Create small synthetic sequences for testing."""
    rng = np.random.default_rng(0)
    X = rng.random((40, 3, 4)).astype(np.float32)
    y = rng.random(40).astype(np.float32)
    return X, y


def test_dataloader_batch_shapes(sample_data) -> None:
    """Batches should preserve sequence and feature dimensions."""
    X, y = sample_data
    loader = make_dataloader(X, y, batch_size=8)
    X_batch, y_batch = next(iter(loader))

    assert X_batch.shape[1:] == (3, 4)
    assert X_batch.shape[0] == y_batch.shape[0] == 8


def test_dataloader_covers_all_samples(sample_data) -> None:
    """The loader should yield every sample exactly once per epoch."""
    X, y = sample_data
    loader = make_dataloader(X, y, batch_size=8)
    total = sum(len(y_batch) for _, y_batch in loader)
    assert total == len(y)


def test_dataloader_no_shuffle_preserves_order(sample_data) -> None:
    """With shuffle=False the first batch should match the first samples."""
    X, y = sample_data
    loader = make_dataloader(X, y, batch_size=8, shuffle=False)
    _, y_batch = next(iter(loader))
    assert np.allclose(y_batch.numpy(), y[:8])


def test_training_changes_weights(sample_data) -> None:
    """Training should update the model parameters."""
    X, y = sample_data
    model = TinyModel(input_size=4)
    before = model.fc.weight.detach().clone()

    train_pytorch_model(
        model,
        make_dataloader(X, y, batch_size=8),
        epochs=2,
        learning_rate=0.01,
        log_every=0,
    )

    assert not torch.equal(before, model.fc.weight.detach())


def test_training_returns_model(sample_data) -> None:
    """train_pytorch_model should return the same model instance."""
    X, y = sample_data
    model = TinyModel(input_size=4)
    returned = train_pytorch_model(
        model,
        make_dataloader(X, y, batch_size=8),
        epochs=1,
        learning_rate=0.01,
        log_every=0,
    )
    assert returned is model


def test_evaluate_returns_all_metrics(sample_data) -> None:
    """Evaluation should return the four expected metric keys."""
    X, y = sample_data
    model = TinyModel(input_size=4)
    metrics = evaluate_pytorch_model(model, X, y)
    assert set(metrics.keys()) == {"rmse", "mae", "mape", "r2"}


def test_evaluate_leaves_model_in_eval_mode(sample_data) -> None:
    """Evaluation should switch the model to eval mode."""
    X, y = sample_data
    model = TinyModel(input_size=4)
    model.train()
    evaluate_pytorch_model(model, X, y)
    assert not model.training
