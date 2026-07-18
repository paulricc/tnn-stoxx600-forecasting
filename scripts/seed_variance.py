"""Measure run-to-run variance of LSTM and TNN across random seeds.

Trains each model with its tuned hyperparameters at horizon=1, repeated
over several seeds, and reports mean and standard deviation of each metric.
This tells us whether the observed gap between models is larger than the
noise from random initialization.
"""

import logging
import statistics
from datetime import date

import pandas as pd

from src.config import load_config
from src.data.downloader import download_stoxx600
from src.data.preprocessor import Preprocessor
from src.features.sequences import make_sequences
from src.models.lstm import LSTMModel
from src.models.tnn import TNN
from src.training.trainer import (
    evaluate_pytorch_model,
    make_dataloader,
    train_pytorch_model,
)
from src.utils import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

SEEDS = [0, 1, 2, 3, 4]
HORIZONS = [1, 7, 30]


def main() -> None:
    """Run the seed variance experiment for LSTM and TNN across horizons."""
    config = load_config()

    df = download_stoxx600(
        start_date=date.fromisoformat(config.data.start_date),
        end_date=date.fromisoformat(config.data.end_date),
    )

    train_size = int(len(df) * config.data.train_test_split)
    df_train, df_test = df[:train_size], df[train_size:]

    preprocessor = Preprocessor()
    df_train_processed = preprocessor.fit_transform(df_train)
    df_test_processed = preprocessor.transform(df_test)

    all_results: list[dict[str, float | str | int]] = []

    for horizon in HORIZONS:
        logger.info("########## horizon = %d ##########", horizon)

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

        input_size = X_train.shape[2]
        results: dict[str, list[float]] = {"lstm": [], "tnn": []}

        for seed in SEEDS:
            logger.info("=== horizon %d, seed %d ===", horizon, seed)

            set_seed(seed)
            train_loader = make_dataloader(
                X_train, y_train, batch_size=config.training.batch_size
            )
            lstm = LSTMModel(
                input_size=input_size,
                hidden_size=config.models.lstm.hidden_size,
                num_layers=config.models.lstm.num_layers,
                dropout=config.models.lstm.dropout,
            )
            lstm = train_pytorch_model(
                lstm,
                train_loader,
                epochs=config.training.epochs,
                learning_rate=config.models.lstm.learning_rate,
                log_every=0,
            )
            lstm_rmse = evaluate_pytorch_model(lstm, X_test, y_test)["rmse"]
            results["lstm"].append(lstm_rmse)
            logger.info(
                "horizon %d, seed %d — LSTM RMSE: %.5f", horizon, seed, lstm_rmse
            )

            set_seed(seed)
            train_loader = make_dataloader(
                X_train, y_train, batch_size=config.training.batch_size
            )
            tnn = TNN(
                input_size=input_size,
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
                log_every=0,
            )
            tnn_rmse = evaluate_pytorch_model(tnn, X_test, y_test)["rmse"]
            results["tnn"].append(tnn_rmse)
            logger.info("horizon %d, seed %d — TNN RMSE: %.5f", horizon, seed, tnn_rmse)

        for name, values in results.items():
            all_results.append(
                {
                    "horizon": horizon,
                    "model": name,
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values),
                    "min": min(values),
                    "max": max(values),
                }
            )

    print(f"\n=== Test RMSE over {len(SEEDS)} seeds ===")
    print(pd.DataFrame(all_results).to_string(index=False))


if __name__ == "__main__":
    main()
