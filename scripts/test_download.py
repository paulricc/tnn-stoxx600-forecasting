import datetime as dt
import logging

import numpy as np
import torch

from src.data.downloader import download_stoxx600
from src.data.preprocessor import Preprocessor
from src.evaluation.metrics import compute_metrics
from src.features.sequences import make_sequences
from src.models.arima import ARIMAModel
from src.models.lstm import LSTMModel
from src.models.tnn import TNN

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

model_tnn = TNN(
    input_size=4,
    kernel_output_size=64,
    kernel_size=2,
    hidden_size=32,
)
X_sample = torch.tensor(X_train[:5])
output_tnn = model_tnn(X_sample)

close_series = df_train_preprocessed["Close"]

arima_model = ARIMAModel(order=(1, 1, 1))
arima_model.fit(close_series)
forecast = arima_model.predict(n_periods=5)


y_true_sample = y_test[:10]
y_pred_sample = y_test[:10] + np.random.normal(0, 0.01, size=10)  # finta predizione

metrics = compute_metrics(y_true_sample, y_pred_sample)
print(metrics)
