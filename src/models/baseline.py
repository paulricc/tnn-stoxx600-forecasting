"""Naive baselines for time series forecasting."""

import numpy as np
import pandas as pd


def persistence_forecast(
    test_series: pd.Series,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Forecast each point as the last observed value, `horizon` steps back.

    This is the optimal forecast under a pure random walk, and therefore the
    baseline any model must beat to demonstrate it extracts usable signal.

    Args:
        test_series: Test time series.
        horizon: Number of steps ahead being forecast.

    Returns:
        Tuple of (y_true, y_pred), each of shape (n_origins,), where
        n_origins is len(test_series) - horizon + 1.
    """
    values = np.asarray(test_series, dtype=np.float64)
    n_origins = len(values) - horizon

    if n_origins <= 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    y_pred = values[:n_origins]
    y_true = values[horizon:]
    return y_true, y_pred
