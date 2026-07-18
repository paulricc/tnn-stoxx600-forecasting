"""Unit tests for the ARIMA model wrapper."""

import numpy as np
import pandas as pd
import pytest

from src.models.arima import ARIMAModel


@pytest.fixture
def sample_series() -> pd.Series:
    """Create a short synthetic series with trend and noise."""
    rng = np.random.default_rng(0)
    values = np.cumsum(rng.normal(0, 1, 120)) + 100
    return pd.Series(values)


def test_predict_raises_if_not_fitted(sample_series: pd.Series) -> None:
    """predict() should raise RuntimeError if called before fit()."""
    model = ARIMAModel(order=(1, 1, 1))
    with pytest.raises(RuntimeError):
        model.predict(n_periods=5)


def test_fit_returns_self(sample_series: pd.Series) -> None:
    """fit() should return self to allow method chaining."""
    model = ARIMAModel(order=(1, 1, 1))
    assert model.fit(sample_series) is model


def test_predict_length(sample_series: pd.Series) -> None:
    """predict() should return exactly n_periods values."""
    model = ARIMAModel(order=(1, 1, 1)).fit(sample_series)
    assert len(model.predict(n_periods=7)) == 7


def test_rolling_forecast_output_length(sample_series: pd.Series) -> None:
    """Rolling forecast should return one prediction per valid origin."""
    train, test = sample_series[:100], sample_series[100:]
    horizon = 3

    model = ARIMAModel(order=(1, 1, 1))
    y_true, y_pred = model.rolling_forecast(train, test, horizon=horizon)

    assert len(y_true) == len(y_pred) == len(test) - horizon + 1


def test_rolling_forecast_true_values_match_test_series(
    sample_series: pd.Series,
) -> None:
    """y_true should be the test values offset by horizon - 1."""
    train, test = sample_series[:100], sample_series[100:]
    horizon = 3

    model = ARIMAModel(order=(1, 1, 1))
    y_true, _ = model.rolling_forecast(train, test, horizon=horizon)

    expected = np.asarray(test)[horizon - 1 :]
    assert np.allclose(y_true, expected)


def test_rolling_forecast_empty_when_test_too_short(
    sample_series: pd.Series,
) -> None:
    """If horizon exceeds the test length, return empty arrays."""
    train, test = sample_series[:100], sample_series[100:105]

    model = ARIMAModel(order=(1, 1, 1))
    y_true, y_pred = model.rolling_forecast(train, test, horizon=10)

    assert len(y_true) == 0
    assert len(y_pred) == 0
