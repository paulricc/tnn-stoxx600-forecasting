"""Training entry point for TNN STOXX600 forecasting project."""

import logging
from datetime import date
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import load_config
from src.data.downloader import download_stoxx600
from src.data.preprocessor import Preprocessor
from src.evaluation.metrics import compute_metrics
from src.features.sequences import make_sequences
from src.models.arima import ARIMAModel
from src.models.lstm import LSTMModel
from src.models.tnn import TNN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
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
) -> nn.Module:
    """Train a PyTorch model.

    Args:
        model: The PyTorch model to train.
        train_loader: DataLoader for training data.
        epochs: Number of training epochs.
        learning_rate: Learning rate for the Adam optimizer.

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

        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / len(train_loader)
            logger.info("Epoch %d/%d — loss: %.4f", epoch + 1, epochs, avg_loss)

    return model


def evaluate_pytorch_model(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Evaluate a trained PyTorch model on test data.

    Args:
        model: Trained PyTorch model.
        X_test: Test input array.
        y_test: Test target array.

    Returns:
        Dictionary of evaluation metrics.
    """
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_pred = model(X_tensor).squeeze().numpy()

    return compute_metrics(y_test, y_pred)


def main() -> None:
    """Run the full training pipeline."""
    config = load_config()
    logger.info("Configuration loaded")

    df = download_stoxx600(
        start_date=date.fromisoformat(config.data.start_date),
        end_date=date.fromisoformat(config.data.end_date),
    )

    train_size = int(len(df) * config.data.train_test_split)
    df_train = df[:train_size]
    df_test = df[train_size:]

    preprocessor = Preprocessor()
    df_train_processed = preprocessor.fit_transform(df_train)
    df_test_processed = preprocessor.transform(df_test)

    preprocessor_path = Path("data/processed/preprocessor.joblib")
    preprocessor.save(preprocessor_path)

    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)

    for horizon in config.training.horizons:
        logger.info("Training models for horizon=%d", horizon)

        X_train, y_train = make_sequences(
            df_train_processed,
            sequence_length=config.data.sequence_length,
            horizon=horizon,
        )
        X_test, y_test = make_sequences(
            df_test_processed,
            sequence_length=config.data.sequence_length,
            horizon=horizon,
        )

        train_loader = make_dataloader(
            X_train, y_train, batch_size=config.training.batch_size
        )

        # --- LSTM ---
        with mlflow.start_run(run_name=f"lstm_horizon_{horizon}"):
            mlflow.log_params(
                {
                    "model": "lstm",
                    "horizon": horizon,
                    "hidden_size": config.models.lstm.hidden_size,
                    "num_layers": config.models.lstm.num_layers,
                    "epochs": config.training.epochs,
                    "learning_rate": config.training.learning_rate,
                }
            )

            lstm = LSTMModel(
                input_size=X_train.shape[2],
                hidden_size=config.models.lstm.hidden_size,
                num_layers=config.models.lstm.num_layers,
                dropout=config.models.lstm.dropout,
            )
            lstm = train_pytorch_model(
                lstm,
                train_loader,
                epochs=config.training.epochs,
                learning_rate=config.training.learning_rate,
            )
            metrics = evaluate_pytorch_model(lstm, X_test, y_test)
            mlflow.log_metrics(metrics)
            logger.info("LSTM horizon=%d metrics: %s", horizon, metrics)

            model_path = Path(f"data/processed/lstm_horizon_{horizon}.pt")
            torch.save(lstm.state_dict(), model_path)
            mlflow.log_artifact(str(model_path))

        # --- TNN ---
        with mlflow.start_run(run_name=f"tnn_horizon_{horizon}"):
            mlflow.log_params(
                {
                    "model": "tnn",
                    "horizon": horizon,
                    "kernel_output_size": config.models.tnn.kernel_output_size,
                    "kernel_size": config.models.tnn.kernel_size,
                    "hidden_size": config.models.tnn.hidden_size,
                    "epochs": config.training.epochs,
                    "learning_rate": config.training.learning_rate,
                }
            )

            tnn = TNN(
                input_size=X_train.shape[2],
                kernel_output_size=config.models.tnn.kernel_output_size,
                kernel_size=config.models.tnn.kernel_size,
                hidden_size=config.models.tnn.hidden_size,
                dropout=config.models.tnn.dropout,
            )
            tnn = train_pytorch_model(
                tnn,
                train_loader,
                epochs=config.training.epochs,
                learning_rate=config.training.learning_rate,
            )
            metrics = evaluate_pytorch_model(tnn, X_test, y_test)
            mlflow.log_metrics(metrics)
            logger.info("TNN horizon=%d metrics: %s", horizon, metrics)

            model_path = Path(f"data/processed/tnn_horizon_{horizon}.pt")
            torch.save(tnn.state_dict(), model_path)
            mlflow.log_artifact(str(model_path))

        # --- ARIMA ---
        with mlflow.start_run(run_name=f"arima_horizon_{horizon}"):
            mlflow.log_params(
                {
                    "model": "arima",
                    "horizon": horizon,
                    "order": str(config.models.arima.order),
                }
            )

            arima = ARIMAModel(order=config.models.arima.order)
            arima.fit(df_train_processed["Close"])
            forecast = arima.predict(n_periods=horizon)

            y_test_arima = df_test_processed["Close"].values[:horizon]
            metrics = compute_metrics(y_test_arima, forecast)
            mlflow.log_metrics(metrics)
            logger.info("ARIMA horizon=%d metrics: %s", horizon, metrics)


if __name__ == "__main__":
    main()
