# TNN STOXX Europe 600 Forecasting

A production-grade implementation of the Time-series Neural Network (TNN) for stock price forecasting, benchmarked against LSTM and ARIMA models on the STOXX Europe 600 index.

Based on: [Zhang et al., "Time-Series Neural Network: A High-Accuracy Time-Series Forecasting Method Based on Kernel Filter and Time Attention", Information 2023, 14, 500](https://doi.org/10.3390/info14090500)

---

## Models

- **TNN** — custom PyTorch implementation of the Time-series Neural Network from the paper, combining a 1D Kernel Filter and a Time Attention mechanism
- **LSTM** — PyTorch baseline
- **ARIMA** — classical statistical baseline via statsmodels

---

## Results (horizon = 1 day)

| Model | RMSE | MAE | MAPE | R² |
|-------|------|-----|------|-----|
| TNN   | 0.025 | 0.019 | 2.587 | 0.937 |
| LSTM  | 0.029 | 0.022 | 2.933 | 0.918 |
| ARIMA | 0.554 | 0.554 | 5.28  | —    |

> Metrics computed on normalized data. TNN outperforms LSTM across all horizons (1, 7, 30 days), consistent with the original paper.

---

## Project Structure

tnn-stoxx600-forecasting/
├── src/
│   ├── data/           # Downloader and preprocessor
│   ├── features/       # Time series sequence generation
│   ├── models/         # TNN, LSTM, ARIMA
│   └── evaluation/     # Metrics (RMSE, MAE, MAPE, R²)
├── configs/            # YAML configuration
├── tests/              # Unit tests
├── data/
│   ├── raw/            # Downloaded CSV data
│   └── processed/      # Saved models and preprocessor
├── train.py            # Training entry point
└── predict.py          # Inference entry point

---

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
git clone https://github.com/YOUR_USERNAME/tnn-stoxx600-forecasting.git
cd tnn-stoxx600-forecasting

uv sync
uv pip install -e .
```

---

## Usage

### Train all models

```bash
uv run python train.py
```

This will:
- Download STOXX Europe 600 data from Yahoo Finance (2014–2024)
- Preprocess and normalize the data
- Train TNN, LSTM, and ARIMA for horizons of 1, 7, and 30 days
- Log all parameters, metrics, and model artifacts to MLflow

### View results in MLflow

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open `http://localhost:5000` to compare models interactively.

### Run inference

```bash
uv run python predict.py
```

Downloads the most recent year of data and produces predictions from all trained models across all horizons.

---

## Configuration

All hyperparameters and settings live in `configs/config.yaml`:

```yaml
data:
  start_date: "2014-01-01"
  end_date: "2024-01-01"
  sequence_length: 3

models:
  lstm:
    hidden_size: 32
  tnn:
    kernel_output_size: 64
    kernel_size: 2
    hidden_size: 32
  arima:
    order: [1, 1, 1]

training:
  horizons: [1, 7, 30]
  epochs: 100
  learning_rate: 0.001
```

---

## Stack

- **Python 3.13** — uv, Ruff, mypy, pytest, pre-commit
- **ML** — PyTorch, statsmodels, scikit-learn, pandas, yfinance
- **MLOps** — MLflow (SQLite backend)
- **Infrastructure** — Docker, GitHub Actions
