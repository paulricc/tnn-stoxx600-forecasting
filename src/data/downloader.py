import datetime as dt
import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path("data/raw")
STOXX_TICKER = "^STOXX"


def download_stoxx600(
    start_date: dt.date,
    end_date: dt.date,
    output_dir: Path = RAW_DATA_DIR,
) -> pd.DataFrame:
    logger.info(f"downloading STOXX 600 data from {start_date} to {end_date}")
    ticker = yf.Ticker(STOXX_TICKER)
    df = ticker.history(start=start_date.isoformat(), end=end_date.isoformat())

    if df.empty:
        raise ValueError(
            f"No data found for ticker {ticker} from {start_date} to {end_date}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "stoxx600_raw.csv"
    df.to_csv(output_path)

    logger.info(f"Saved {len(df)} to {output_path}")
    return df
