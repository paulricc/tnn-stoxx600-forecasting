import datetime as dt
import logging

from src.data.downloader import download_stoxx600
from src.data.preprocessor import Preprocessor
from src.features.sequences import make_sequences

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

X_train, y_train = make_sequences(df_train_preprocessed, sequence_length=3, horizon=1)
X_test, y_test = make_sequences(df_test_preprocessed, sequence_length=3, horizon=1)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")
