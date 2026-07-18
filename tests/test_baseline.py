"""Unit tests for the naive persistence baseline."""

import numpy as np
import pandas as pd

from src.models.baseline import persistence_forecast


def test_persistence_is_not_perfect_at_horizon_one() -> None:
    """Persistence must not predict a point using itself."""
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    y_true, y_pred = persistence_forecast(series, horizon=1)

    assert np.array_equal(y_pred, np.array([1.0, 2.0, 3.0, 4.0]))
    assert np.array_equal(y_true, np.array([2.0, 3.0, 4.0, 5.0]))


def test_persistence_alignment_at_longer_horizon() -> None:
    """At horizon=2, each prediction should be the value two steps earlier."""
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    y_true, y_pred = persistence_forecast(series, horizon=2)

    assert np.array_equal(y_pred, np.array([1.0, 2.0, 3.0]))
    assert np.array_equal(y_true, np.array([3.0, 4.0, 5.0]))


def test_persistence_empty_when_series_too_short() -> None:
    """If the horizon exceeds the series length, return empty arrays."""
    series = pd.Series([1.0, 2.0, 3.0])
    y_true, y_pred = persistence_forecast(series, horizon=5)

    assert len(y_true) == 0
    assert len(y_pred) == 0
