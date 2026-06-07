import datetime as dt
import logging

from src.data.downloader import download_stoxx600
from src.data.preprocessor import Preprocessor

logging.basicConfig(level=logging.INFO)

df = download_stoxx600(
    start_date=dt.date(2014, 1, 1),
    end_date=dt.date(2024, 1, 1),
)

print(df.columns.tolist())
print(df.head())

train_size = int(len(df) * 0.8)
df_train = df[:train_size]
df_test = df[train_size:]

preprocessor = Preprocessor()
df_train_preprocessed = preprocessor.fit_transform(df_train)
df_test_preprocessed = preprocessor.transform(df_test)

print(df_train_preprocessed.head())
print(df_train_preprocessed.describe())
