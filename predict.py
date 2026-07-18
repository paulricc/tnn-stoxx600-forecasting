"""Inference entry point: forecast future values with all trained models."""

import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.data.downloader import download_stoxx600
from src.data.preprocessor import TARGET, Preprocessor
from src.features.sequences import make_inference_window
from src.models.arima import ARIMAModel
from src.models.loaders import load_lstm, load_tnn, predict_pytorch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

LOOKBACK_DAYS = 365


def main() -> None:
    """Forecast beyond the most recent observation with every model."""
    config = load_config()

    end_date = date.today()
    start_date = end_date - timedelta(days=LOOKBACK_DAYS)
    df = download_stoxx600(start_date=start_date, end_date=end_date)

    preprocessor_path = Path("data/processed/preprocessor.joblib")
    if not preprocessor_path.exists():
        raise FileNotFoundError("Preprocessor not found. Run train.py first.")

    preprocessor = Preprocessor.load(preprocessor_path)
    df_processed = preprocessor.transform(df)

    window = make_inference_window(
        df_processed, sequence_length=config.data.sequence_length
    )
    input_size = window.shape[2]

    # Persistence: the naive baseline that outperforms every model on the
    # test set. Included so the comparison here matches the reported results.
    last_observed = float(df_processed[TARGET].iloc[-1])

    rows: list[dict[str, object]] = []

    for horizon in config.training.horizons:
        row: dict[str, object] = {
            "horizon": horizon,
            "persistence": last_observed,
        }

        lstm_path = Path(f"data/processed/lstm_horizon_{horizon}.pt")
        if lstm_path.exists():
            lstm = load_lstm(lstm_path, input_size, config.models.lstm)
            row["lstm"] = float(predict_pytorch(lstm, window)[0])
        else:
            logger.warning("No LSTM found for horizon=%d", horizon)
            row["lstm"] = None

        tnn_path = Path(f"data/processed/tnn_horizon_{horizon}.pt")
        if tnn_path.exists():
            tnn = load_tnn(tnn_path, input_size, config.models.tnn)
            row["tnn"] = float(predict_pytorch(tnn, window)[0])
        else:
            logger.warning("No TNN found for horizon=%d", horizon)
            row["tnn"] = None

        # Fitting on all available data and forecasting ahead is the correct
        # inference path for ARIMA. rolling_forecast is for evaluation only,
        # since it consumes known future values.
        arima = ARIMAModel(order=config.models.arima.order).fit(df_processed[TARGET])
        row["arima"] = float(arima.predict(n_periods=horizon)[-1])

        rows.append(row)

    print(f"\n=== Forecasts from {df.index[-1].date()}, normalized scale ===")
    print(pd.DataFrame(rows).to_string(index=False))
    print(
        "\nValues are on the training set's min-max scale and are not "
        "inverse-transformed. Values above 1.0 indicate prices exceeding the "
        "training maximum. See README limitations."
    )


if __name__ == "__main__":
    main()
