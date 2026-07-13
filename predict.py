"""Prediction entry point for TNN STOXX600 forecasting project."""

import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import load_config
from src.data.downloader import download_stoxx600
from src.data.preprocessor import Preprocessor
from src.features.sequences import make_sequences
from src.models.arima import ARIMAModel
from src.models.lstm import LSTMModel
from src.models.tnn import TNN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def load_lstm(path: Path, input_size: int, config) -> LSTMModel:
    """Load a trained LSTM model from disk.

    Args:
        path: Path to the saved state dict.
        input_size: Number of input features.
        config: Model configuration.

    Returns:
        Loaded LSTM model in eval mode.
    """
    model = LSTMModel(
        input_size=input_size,
        hidden_size=config.models.lstm.hidden_size,
        num_layers=config.models.lstm.num_layers,
        dropout=config.models.lstm.dropout,
    )
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model


def load_tnn(path: Path, input_size: int, config) -> TNN:
    """Load a trained TNN model from disk.

    Args:
        path: Path to the saved state dict.
        input_size: Number of input features.
        config: Model configuration.

    Returns:
        Loaded TNN model in eval mode.
    """
    model = TNN(
        input_size=input_size,
        kernel_output_size=config.models.tnn.kernel_output_size,
        kernel_size=config.models.tnn.kernel_size,
        hidden_size=config.models.tnn.hidden_size,
        dropout=config.models.tnn.dropout,
    )
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model


def predict_pytorch(model: torch.nn.Module, X: np.ndarray) -> np.ndarray:
    """Run inference with a PyTorch model.

    Args:
        model: Trained PyTorch model in eval mode.
        X: Input array of shape (n_samples, sequence_length, n_features).

    Returns:
        Predictions array of shape (n_samples,).
    """
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        predictions = model(X_tensor).squeeze().numpy()
    return predictions


def main() -> None:
    """Run predictions with all trained models for all horizons."""
    config = load_config()
    logger.info("Configuration loaded")

    end_date = date.today()
    start_date = end_date - timedelta(days=365)

    logger.info("Downloading recent data from %s to %s", start_date, end_date)
    df = download_stoxx600(start_date=start_date, end_date=end_date)

    preprocessor_path = Path("data/processed/preprocessor.joblib")
    if not preprocessor_path.exists():
        raise FileNotFoundError("Preprocessor not found. Run train.py first.")

    preprocessor = Preprocessor.load(preprocessor_path)
    df_processed = preprocessor.transform(df)

    results = []

    for horizon in config.training.horizons:
        X, y = make_sequences(
            df_processed,
            sequence_length=config.data.sequence_length,
            horizon=horizon,
        )

        input_size = X.shape[2]
        row: dict[str, object] = {"horizon": horizon}

        # LSTM
        lstm_path = Path(f"data/processed/lstm_horizon_{horizon}.pt")
        if lstm_path.exists():
            lstm = load_lstm(lstm_path, input_size, config)
            lstm_preds = predict_pytorch(lstm, X)
            row["lstm_last_pred"] = float(lstm_preds[-1])
        else:
            logger.warning("LSTM model not found for horizon=%d", horizon)
            row["lstm_last_pred"] = None

        # TNN
        tnn_path = Path(f"data/processed/tnn_horizon_{horizon}.pt")
        if tnn_path.exists():
            tnn = load_tnn(tnn_path, input_size, config)
            tnn_preds = predict_pytorch(tnn, X)
            row["tnn_last_pred"] = float(tnn_preds[-1])
        else:
            logger.warning("TNN model not found for horizon=%d", horizon)
            row["tnn_last_pred"] = None

        # ARIMA
        arima = ARIMAModel(order=config.models.arima.order)
        arima.fit(df_processed["Close"])
        arima_preds = arima.predict(n_periods=horizon)
        row["arima_last_pred"] = float(arima_preds[-1])

        results.append(row)

    results_df = pd.DataFrame(results)
    print("\n=== Predictions (normalized scale) ===")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
