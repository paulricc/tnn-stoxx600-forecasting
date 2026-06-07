import datetime as dt
import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path("data/raw")
STOXX_TICKER = "^STOXX"
COLUMNS_TO_KEEP = ["Open", "High", "Low", "Close", "Volume"]


def download_stoxx600(
    start_date: dt.date,
    end_date: dt.date,
    output_dir: Path = RAW_DATA_DIR,
) -> pd.DataFrame:
    """
    Download STOXX Europe 600 historical data from Yahoo Finance.

    Args:
        start_date: Start date in dt.date format.
        end_date: End date in dt.date format.
        output_dir: Directory where the raw CSV will be saved.

    Returns:
        DataFrame with raw OHLCV data.
    """
    logger.info(f"downloading STOXX 600 data from {start_date} to {end_date}")
    ticker = yf.Ticker(STOXX_TICKER)
    df = ticker.history(start=start_date.isoformat(), end=end_date.isoformat())
    df = df[COLUMNS_TO_KEEP]

    if df.empty:
        raise ValueError(
            f"No data found for ticker {ticker} from {start_date} to {end_date}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "stoxx600_raw.csv"
    df.to_csv(output_path)

    logger.info(f"Saved {len(df)} to {output_path}")
    return df
