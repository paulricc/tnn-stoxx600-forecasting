"""Training entry point for TNN STOXX600 forecasting project."""

import logging
from datetime import date
from pathlib import Path

import mlflow
import torch

from src.config import load_config
from src.data.downloader import download_stoxx600
from src.data.preprocessor import Preprocessor
from src.evaluation.metrics import compute_metrics
from src.features.sequences import make_sequences
from src.models.arima import ARIMAModel
from src.models.lstm import LSTMModel
from src.models.tnn import TNN
from src.training.trainer import (
    evaluate_pytorch_model,
    make_dataloader,
    train_pytorch_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


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
                    "learning_rate": config.models.lstm.learning_rate,
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
                learning_rate=config.models.lstm.learning_rate,
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
                    "learning_rate": config.models.tnn.learning_rate,
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
                learning_rate=config.models.tnn.learning_rate,
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
            y_true_arima, y_pred_arima = arima.rolling_forecast(
                train_series=df_train_processed["Close"],
                test_series=df_test_processed["Close"],
                horizon=horizon,
            )
            metrics = compute_metrics(y_true_arima, y_pred_arima)
            mlflow.log_metrics(metrics)
            logger.info("ARIMA horizon=%d metrics: %s", horizon, metrics)


if __name__ == "__main__":
    main()
