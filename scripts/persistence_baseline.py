"""Score a naive persistence baseline on the test set.

Predicts each point as the last observed value. Any model that fails to beat
this is not extracting signal beyond a random walk.
"""

import logging
from datetime import date

import pandas as pd

from src.config import load_config
from src.data.downloader import download_stoxx600
from src.data.preprocessor import Preprocessor
from src.evaluation.metrics import compute_metrics
from src.models.baseline import persistence_forecast

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Score the persistence baseline at each horizon."""
    config = load_config()

    df = download_stoxx600(
        start_date=date.fromisoformat(config.data.start_date),
        end_date=date.fromisoformat(config.data.end_date),
    )

    train_size = int(len(df) * config.data.train_test_split)
    df_train, df_test = df[:train_size], df[train_size:]

    preprocessor = Preprocessor()
    preprocessor.fit(df_train)
    df_test_processed = preprocessor.transform(df_test)

    rows = []
    for horizon in config.training.horizons:
        y_true, y_pred = persistence_forecast(
            df_test_processed["Close"], horizon=horizon
        )
        metrics = compute_metrics(y_true, y_pred)
        rows.append({"horizon": horizon, **metrics})
        logger.info("Persistence horizon=%d: %s", horizon, metrics)

    print("\n=== Persistence baseline (test set) ===")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
