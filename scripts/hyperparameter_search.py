"""Grid search over hyperparameters, evaluated on a held-out validation set.

The test set is never touched here: it is reserved for the final evaluation
in train.py. Results are logged to MLflow as nested runs and the best
configuration is written to configs/best_params.yaml.
"""

import itertools
import logging
from datetime import date
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import yaml

from src.config import Config, load_config
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# Fewer epochs than the final training: the search only needs to rank
# configurations relative to each other, not to squeeze out the last decimal.
SEARCH_EPOCHS = 30

# Horizon used to select hyperparameters. Tuning on every horizon would
# multiply the cost; horizon=1 is the primary task in the paper.
TUNING_HORIZON = 1

KERNEL_SIZE = 2

LSTM_GRID: dict[str, list[Any]] = {
    "sequence_length": [3, 8, 14, 30],
    "hidden_size": [32, 64, 128],
    "learning_rate": [0.01, 0.001, 0.0001],
}

TNN_GRID: dict[str, list[Any]] = {
    "sequence_length": [3],
    "kernel_output_size": [32, 64, 128],
    "hidden_size": [32, 64, 128],
    "learning_rate": [0.01, 0.001, 0.0001],
}

BEST_PARAMS_PATH = Path("configs/best_params.yaml")


def expand_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Expand a parameter grid into a list of parameter combinations.

    Args:
        grid: Mapping from parameter name to list of candidate values.

    Returns:
        List of dictionaries, one per combination.
    """
    keys = list(grid.keys())
    return [dict(zip(keys, values)) for values in itertools.product(*grid.values())]


def split_data(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame chronologically into train, validation and test.

    Args:
        df: Full DataFrame, ordered by date.
        train_ratio: Fraction of rows used for training.
        val_ratio: Fraction of rows used for validation.

    Returns:
        Tuple of (train, validation, test) DataFrames.
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return df[:train_end], df[train_end:val_end], df[val_end:]


def build_model(model_name: str, params: dict[str, Any], input_size: int):
    """Instantiate a model from a parameter combination.

    Args:
        model_name: Either "lstm" or "tnn".
        params: Parameter combination from the grid.
        input_size: Number of input features per time step.

    Returns:
        An untrained PyTorch model.

    Raises:
        ValueError: If the model name is not recognised.
    """
    if model_name == "lstm":
        return LSTMModel(
            input_size=input_size,
            hidden_size=params["hidden_size"],
            num_layers=1,
            dropout=0.0,
        )
    if model_name == "tnn":
        return TNN(
            input_size=input_size,
            kernel_output_size=params["kernel_output_size"],
            kernel_size=KERNEL_SIZE,
            hidden_size=params["hidden_size"],
            dropout=0.2,
        )
    raise ValueError(f"Unknown model: {model_name}")


def search_model(
    model_name: str,
    grid: dict[str, list[Any]],
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    config: Config,
) -> dict[str, Any]:
    """Run a grid search for one model and return the best configuration.

    Args:
        model_name: Either "lstm" or "tnn".
        grid: Parameter grid to search.
        df_train: Preprocessed training DataFrame.
        df_val: Preprocessed validation DataFrame.
        config: Project configuration.

    Returns:
        Dictionary with the best parameters and their validation RMSE.
    """
    combinations = expand_grid(grid)
    logger.info("%s: evaluating %d combinations", model_name, len(combinations))

    best_rmse = float("inf")
    best_params: dict[str, Any] = {}

    with mlflow.start_run(run_name=f"{model_name}_grid_search"):
        mlflow.log_params(
            {
                "model": model_name,
                "search_epochs": SEARCH_EPOCHS,
                "tuning_horizon": TUNING_HORIZON,
                "n_combinations": len(combinations),
            }
        )

        for i, params in enumerate(combinations, start=1):
            sequence_length = params["sequence_length"]

            X_train, y_train = make_sequences(
                df_train,
                sequence_length=sequence_length,
                horizon=TUNING_HORIZON,
            )
            X_val, y_val = make_sequences(
                df_val,
                sequence_length=sequence_length,
                horizon=TUNING_HORIZON,
            )

            if len(X_train) == 0 or len(X_val) == 0:
                logger.warning("Skipping %s: not enough rows for sequences", params)
                continue

            with mlflow.start_run(run_name=f"{model_name}_trial_{i}", nested=True):
                mlflow.log_params(params)

                model = build_model(model_name, params, input_size=X_train.shape[2])
                model = train_pytorch_model(
                    model,
                    make_dataloader(
                        X_train, y_train, batch_size=config.training.batch_size
                    ),
                    epochs=SEARCH_EPOCHS,
                    learning_rate=params["learning_rate"],
                    log_every=0,
                )

                metrics = evaluate_pytorch_model(model, X_val, y_val)
                mlflow.log_metrics(metrics)

            logger.info(
                "%s trial %d/%d %s — val RMSE: %.5f",
                model_name,
                i,
                len(combinations),
                params,
                metrics["rmse"],
            )

            if metrics["rmse"] < best_rmse:
                best_rmse = metrics["rmse"]
                best_params = params

        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_val_rmse", best_rmse)

    logger.info("%s best: %s (val RMSE %.5f)", model_name, best_params, best_rmse)
    return {"params": best_params, "val_rmse": best_rmse}


def main() -> None:
    """Run the hyperparameter search for LSTM and TNN."""
    config = load_config()

    df = download_stoxx600(
        start_date=date.fromisoformat(config.data.start_date),
        end_date=date.fromisoformat(config.data.end_date),
    )

    train_ratio = config.data.train_test_split - config.data.val_split
    df_train, df_val, _ = split_data(df, train_ratio, config.data.val_split)
    logger.info(
        "Split: %d train / %d validation rows (test set untouched)",
        len(df_train),
        len(df_val),
    )

    preprocessor = Preprocessor()
    df_train_processed = preprocessor.fit_transform(df_train)
    df_val_processed = preprocessor.transform(df_val)

    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(f"{config.mlflow.experiment_name}-hpo")

    results = {
        # "lstm": search_model(
        #    "lstm", LSTM_GRID, df_train_processed, df_val_processed, config
        # ),
        "tnn": search_model(
            "tnn", TNN_GRID, df_train_processed, df_val_processed, config
        ),
    }

    BEST_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BEST_PARAMS_PATH, "w") as f:
        yaml.safe_dump(results, f, sort_keys=False)

    logger.info("Best parameters written to %s", BEST_PARAMS_PATH)


if __name__ == "__main__":
    main()
