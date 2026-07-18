# TNN STOXX Europe 600 Forecasting

A from-scratch PyTorch implementation of the Time-series Neural Network (TNN), benchmarked against LSTM, ARIMA, and a naive persistence baseline on the STOXX Europe 600 index.

Based on: [Zhang et al., "Time-Series Neural Network: A High-Accuracy Time-Series Forecasting Method Based on Kernel Filter and Time Attention", *Information* 2023, 14, 500](https://doi.org/10.3390/info14090500)

**Summary of findings:** the paper's central claim — that TNN substantially outperforms LSTM — did not reproduce on this dataset. More importantly, neither neural model beats a one-line persistence baseline at any forecast horizon. Details and interpretation below.

---

## Models

- **TNN** — custom implementation of the architecture described in the paper, combining a 1D kernel filter and a time attention mechanism with a learned weight generator
- **LSTM** — recurrent baseline, reimplemented in PyTorch (the original thesis used Keras)
- **ARIMA** — classical statistical baseline via statsmodels, evaluated with a fixed-parameter rolling-origin forecast
- **Persistence** — naive baseline predicting each value as the last observed value

---

## Results

All figures are test-set RMSE on **min-max normalized** data, so they are not directly comparable to results computed on other scales or datasets. Neural models are reported as mean ± standard deviation over 5 random seeds; ARIMA and persistence are deterministic and have no seed variance.

| Horizon | Persistence | ARIMA  | LSTM            | TNN             |
|---------|-------------|--------|-----------------|-----------------|
| 1 day   | **0.0205**  | 0.0321 | 0.0242 ± 0.0017 | 0.0249 ± 0.0007 |
| 7 days  | **0.0529**  | 0.0576 | 0.0584 ± 0.0029 | 0.0571 ± 0.0022 |
| 30 days | **0.1032**  | 0.1049 | 0.1181 ± 0.0119 | 0.1193 ± 0.0066 |

### No model beats the naive baseline

The persistence baseline outperforms every model at every horizon. At 1 day the margin over the best neural model is roughly 15%, far outside the variation observed across random seeds.

This is the central result, and it holds for a reason that is well established in the financial forecasting literature but easy to overlook: **daily index price levels behave close to a random walk.** Under a random walk, the optimal forecast of tomorrow's level is today's level. No architecture recovers signal that is not present in the target.

### Why high R² is misleading on this task

TNN reaches R² ≈ 0.92 and LSTM ≈ 0.95 at a 1-day horizon — figures comparable to the 0.95 that Zhang et al. report as evidence their architecture works. On the same data, the persistence baseline reaches **R² = 0.958** while fitting nothing.

On price *levels*, a high R² largely reflects the autocorrelation of the series rather than predictive skill. Reported without a naive baseline, it makes almost any model appear successful. The root cause is the choice of target: forecasting levels is a task where doing nothing is close to optimal. Forecasting **returns** instead — where persistence predicts zero by construction and any signal must be earned — is the appropriate reframing and the natural next step for this work.

### Model comparison

Setting the baseline aside, the models compare as follows.

**TNN and LSTM are statistically indistinguishable at every horizon.** At each one, the difference between the means is smaller than the standard deviation across seeds, and the observed ranges overlap substantially. LSTM is marginally ahead at 1 day, TNN at 7 and 30 days, but none of these differences exceed initialization noise. Note that this is an informal comparison of overlapping intervals, not a formal significance test (see limitations).

**TNN is consistently more stable.** Its standard deviation is 39%, 75%, and 56% of LSTM's at horizons 1, 7, and 30 respectively. The margin varies, but the direction holds at every horizon, suggesting the kernel filter and attention mechanism reduce sensitivity to initialization even though they do not improve accuracy. With only 5 seeds, this observation is suggestive rather than established.

**ARIMA is worst at 1 day and best of the three models at 30 days.** The neural advantage shrinks as the horizon grows and reverses by 30 days.

**All models degrade steeply with horizon**, with RMSE increasing roughly fivefold from 1 to 30 days. At 30 days, R² is negative for both ARIMA (−0.17) and persistence (−0.13), meaning neither outperforms simply predicting the test-set mean.

### Non-replication of the paper's central claim

Zhang et al. report that TNN substantially outperforms LSTM, with the largest margin at long horizons: RMSE 0.14 vs 0.28 at 30 days on the S&P 500, a twofold gap. That advantage did not reproduce here at any horizon, including the one where the paper claims it is greatest — the two models finish within 1% of each other at 30 days.

This is a non-replication under different conditions, not a refutation. The setups differ in index (STOXX Europe 600 vs S&P 500), time period, feature set, preprocessing, and hyperparameter selection, and this project has not attempted to replicate the paper's exact experimental conditions. Two methodological differences are worth noting, since this project's own results show that both matter:

- **The paper reports single-run figures with no measure of variance.** An early single-run comparison in this project appeared to show a clear TNN advantage, which disappeared entirely once results were averaged across seeds.
- **The paper reports no naive baseline.** Its headline R² of 0.95 is below what persistence achieves on this dataset.

---

## Limitations

The following are known weaknesses of this study. Several would need addressing before the numbers above could support stronger claims.

**Statistical testing.** Model comparisons rest on visual inspection of mean ± standard deviation across 5 seeds. No formal significance test was performed, and 5 seeds is a small sample for estimating variance. The conclusion that TNN and LSTM are indistinguishable is robust to this — their means differ by less than one standard deviation — but the claim that TNN is *more stable* rests on a variance estimate from very few samples.

**Single train/test split.** The data is split chronologically once (70% train, 10% validation, 20% test). Results depend on that specific split, and no walk-forward or blocked cross-validation was performed. A different split boundary could produce materially different figures.

**ARIMA order is not selected empirically.** The order `(1, 1, 1)` is hardcoded in the configuration rather than chosen by the AIC/BIC grid search used in the original thesis. A properly selected order would likely improve ARIMA's results, and its current placement in the ranking should be read with that in mind.

**ARIMA uses fixed coefficients.** The model is estimated once on the training set; its state is then updated with each observed test value without refitting. Refitting at every origin would be more faithful to a true production setting but roughly 500× the computational cost.

**Hyperparameters were tuned at horizon=1 only.** They are reused at 7 and 30 days to keep the search tractable. Both neural models receive identical treatment, so the comparison between them remains fair, but the absolute figures at longer horizons are likely not optimal — a longer `sequence_length`, for instance, may suit a 30-day horizon better.

**The hyperparameter search ranked configurations on single runs,** and is therefore subject to the same variance problem it later helped uncover. Rerunning the TNN search with a wider learning-rate grid produced a *worse* best-of-grid score than a narrower earlier run, which is only possible if seed noise dominates the differences between configurations. The selected values should be read as reasonable defaults, not optima.

**Evaluation windows differ marginally between model families.** The sequence models lose the first `sequence_length` rows of the test set to window construction, so they are scored on 3 fewer points out of roughly 500 than ARIMA and persistence. The effect on the reported figures is negligible but the comparison is not exactly like-for-like.

**Min-max normalization does not extrapolate.** Scaling parameters are fitted on the training set, as required to avoid leakage. When `predict.py` is run on recent data, prices above the 2014–2024 training maximum normalize to values greater than 1.0. This is expected behaviour rather than a bug, but it makes the current inference path unsuitable for production use without a different scaling strategy.

**Predictions are not inverse-transformed.** All outputs remain on the normalized scale. Recovering interpretable price predictions would require inverting the normalization, which is straightforward but not yet implemented.

---

## Project structure

```
tnn-stoxx600-forecasting/
├── src/
│   ├── config.py             # Pydantic config schema and YAML loader
│   ├── utils.py              # Seeding utility for reproducibility
│   ├── data/
│   │   ├── downloader.py     # Yahoo Finance data download
│   │   └── preprocessor.py   # Outliers, interpolation, normalization (fit/transform)
│   ├── features/
│   │   └── sequences.py      # Sliding-window sequence construction
│   ├── models/
│   │   ├── tnn.py            # Kernel filter + time attention
│   │   ├── lstm.py           # LSTM baseline
│   │   ├── arima.py          # ARIMA wrapper with rolling-origin forecast
│   │   └── baseline.py       # Naive persistence baseline
│   ├── training/
│   │   └── trainer.py        # Shared training and evaluation loop
│   └── evaluation/
│       └── metrics.py        # RMSE, MAE, MAPE, R²
├── scripts/
│   ├── hyperparameter_search.py   # Grid search on the validation set
│   ├── seed_variance.py           # Repeated training across random seeds
│   └── persistence_baseline.py    # Naive baseline evaluation
├── configs/
│   └── config.yaml           # All hyperparameters and settings
├── tests/                    # Unit tests
├── data/
│   ├── raw/                  # Downloaded CSV data (gitignored)
│   └── processed/            # Saved models and preprocessor (gitignored)
├── train.py                  # Training entry point
├── predict.py                # Inference entry point
└── Dockerfile
```

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

### Reproduce the results

```bash
uv run python train.py                          # Train all models, log to MLflow
uv run python scripts/seed_variance.py          # Neural models across 5 seeds
uv run python scripts/persistence_baseline.py   # Naive baseline
uv run python scripts/hyperparameter_search.py  # Grid search on validation set
```

`train.py` downloads STOXX Europe 600 data from Yahoo Finance (2014–2024), preprocesses it, trains TNN, LSTM, and ARIMA at horizons of 1, 7, and 30 days, and logs all parameters, metrics, and model artifacts to MLflow.

### View results in MLflow

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open `http://localhost:5000` to compare runs interactively.

### Run inference

```bash
uv run python predict.py
```

Downloads the most recent year of data and produces predictions from all trained models across all horizons. See the limitations section regarding normalization of out-of-range inputs.

### Run the test suite

```bash
uv run pytest tests/ -v
```

---

## Configuration

All hyperparameters and settings live in `configs/config.yaml`:

```yaml
data:
  start_date: "2014-01-01"
  end_date: "2024-01-01"
  train_test_split: 0.8
  val_split: 0.1
  # Shared by LSTM and TNN. Both models independently selected 3
  # in scripts/hyperparameter_search.py.
  sequence_length: 3

models:
  # Selected via scripts/hyperparameter_search.py on a held-out validation
  # set (horizon=1, 30 epochs). See limitations: the search ranked
  # configurations on single runs and is subject to seed noise.
  lstm:
    hidden_size: 128
    num_layers: 1
    dropout: 0.0
    learning_rate: 0.01
  tnn:
    kernel_output_size: 64
    kernel_size: 2
    hidden_size: 32
    dropout: 0.2
    learning_rate: 0.001
  arima:
    order: [1, 1, 1]   # Not empirically selected; see limitations

training:
  horizons: [1, 7, 30]
  epochs: 100
  batch_size: 32

mlflow:
  experiment_name: "tnn-stoxx600-forecasting"
  tracking_uri: "sqlite:///mlflow.db"
```

---

## Stack

- **Python 3.13** — uv, Ruff, mypy, pytest, pre-commit
- **ML** — PyTorch, statsmodels, scikit-learn, pandas, yfinance
- **MLOps** — MLflow with SQLite backend
- **Infrastructure** — Docker, GitHub Actions

---

## Next steps

1. **Reframe the target as returns rather than levels.** This is the single change most likely to make the comparison meaningful, since it removes the trivial autocorrelation that currently dominates every metric.
2. **Replace the single split with walk-forward validation**, so results do not depend on one arbitrary boundary.
3. **Select the ARIMA order via AIC/BIC**, as in the original thesis, rather than hardcoding it.
4. **Increase seed count and apply a formal significance test** to the model comparison.
