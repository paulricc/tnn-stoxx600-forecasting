"""Reusable training and evaluation utilities for PyTorch models."""

import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.metrics import compute_metrics

logger = logging.getLogger(__name__)


def make_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """Create a PyTorch DataLoader from numpy arrays.

    Args:
        X: Input array of shape (n_samples, sequence_length, n_features).
        y: Target array of shape (n_samples,).
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the data at each epoch.

    Returns:
        DataLoader ready for training or evaluation.
    """
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_pytorch_model(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    log_every: int = 10,
) -> nn.Module:
    """Train a PyTorch model with Adam and MSE loss.

    Args:
        model: The PyTorch model to train.
        train_loader: DataLoader for training data.
        epochs: Number of training epochs.
        learning_rate: Learning rate for the Adam optimizer.
        log_every: Log the average loss every N epochs. Set to 0 to disable.

    Returns:
        Trained model.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if log_every and (epoch + 1) % log_every == 0:
            avg_loss = epoch_loss / len(train_loader)
            logger.info("Epoch %d/%d — loss: %.4f", epoch + 1, epochs, avg_loss)

    return model


def evaluate_pytorch_model(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
) -> dict[str, float]:
    """Evaluate a trained PyTorch model.

    Args:
        model: Trained PyTorch model.
        X: Input array.
        y: Ground truth target array.

    Returns:
        Dictionary of evaluation metrics.
    """
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_pred = model(X_tensor).squeeze().numpy()

    return compute_metrics(y, y_pred)
