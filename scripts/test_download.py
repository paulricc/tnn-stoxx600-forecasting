import datetime as dt
import logging

import torch

from src.data.downloader import download_stoxx600
from src.data.preprocessor import Preprocessor
from src.features.sequences import make_sequences
from src.models.lstm import LSTMModel

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

model = LSTMModel(input_size=4, hidden_size=32)
X_sample = torch.tensor(X_train[:5])
output = model(X_sample)

print(f"Input shape: {X_sample.shape}")
print(f"Output shape: {output.shape}")
print(output)
