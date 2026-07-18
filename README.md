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


## Results

### Model comparison across horizons

Test-set RMSE over 5 random seeds, using hyperparameters selected on a held-out
validation set at horizon=1:

| Horizon | LSTM (mean ± std) | TNN (mean ± std) |
|---------|-------------------|------------------|
| 1 day   | 0.0242 ± 0.0017   | 0.0249 ± 0.0007  |
| 7 days  | 0.0584 ± 0.0029   | 0.0571 ± 0.0022  |
| 30 days | 0.1193 ± 0.0119   | 0.1181 ± 0.0066  |

**The two architectures are statistically indistinguishable at every horizon.**
At each one, the difference between the means is smaller than the standard
deviation across seeds, and the observed ranges overlap heavily. LSTM is
marginally ahead at 1 day, TNN at 7 and 30, but none of these differences
exceed the noise from random initialization.

TNN is consistently more stable: its standard deviation is 39%, 75% and 56% of
LSTM's at horizons 1, 7 and 30 respectively. The margin varies, but the
direction holds at every horizon, suggesting the kernel filter and attention
mechanism genuinely reduce sensitivity to initialization even though they do
not improve accuracy.

Both models degrade steeply as the horizon grows — RMSE increases roughly
fivefold from 1 to 30 days for each.

### Non-replication of the paper's central claim

Zhang et al. report that TNN substantially outperforms LSTM, with the largest
margin at long horizons: RMSE 0.14 vs 0.28 at 30 days on the S&P 500, a
twofold gap. That advantage did not reproduce here at any horizon, including
the one where the paper claims it is greatest — the two models finish within
1% of each other at 30 days.

This is a non-replication under different conditions rather than a refutation.
The setups differ in index (STOXX Europe 600 vs S&P 500), feature set,
preprocessing, and hyperparameter selection. Notably, the paper reports
single-run figures with no measure of variance, and this project's own results
show why that matters: an early single-run comparison here appeared to show a
clear TNN advantage, which disappeared entirely once run-to-run variance was
measured across seeds.

### Methodological notes and limitations

**Hyperparameters were tuned at horizon=1 only** (see
`scripts/hyperparameter_search.py`), then reused at 7 and 30 days to keep the
search tractable. Both models receive identical treatment, so the comparison
between them remains fair, but the absolute figures at longer horizons are
likely not optimal — a longer `sequence_length`, for instance, may suit a
30-day horizon better.

**The hyperparameter search itself ranked configurations on single runs**, and
is therefore subject to the same variance problem it later helped uncover. The
selected values should be read as reasonable defaults rather than optima.

**ARIMA is not evaluated like-for-like.** The deep learning models are scored
on ~500 rolling windows; ARIMA is fitted once at the train/test boundary and
scored on `horizon` points forecast from that single origin. At horizon=1 this
means a single prediction, which is why R² is undefined. A rolling-origin
evaluation is needed before the ARIMA figures can be compared directly to the
others.
