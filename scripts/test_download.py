import datetime as dt
import logging

from src.data.downloader import download_stoxx600

logging.basicConfig(level=logging.INFO)

df = download_stoxx600(
    start_date=dt.date(2014, 1, 1),
    end_date=dt.date(2024, 1, 1),
)

print(df.head())
print(df.shape)
print(df.columns.tolist())
