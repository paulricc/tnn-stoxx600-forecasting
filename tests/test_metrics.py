"""Unit tests for the evaluation metrics."""

import numpy as np
import pytest

from src.evaluation.metrics import compute_metrics


def test_perfect_predictions_give_zero_error() -> None:
    """Perfect predictions should give RMSE=0, MAE=0, MAPE=0, R2=1."""
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    metrics = compute_metrics(y, y)

    assert metrics["rmse"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["mae"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["mape"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["r2"] == pytest.approx(1.0, abs=1e-6)


def test_metrics_keys() -> None:
    """compute_metrics should return all four expected keys."""
    y = np.array([1.0, 2.0, 3.0])
    metrics = compute_metrics(y, y)
    assert set(metrics.keys()) == {"rmse", "mae", "mape", "r2"}


def test_metrics_are_floats() -> None:
    """All metric values should be Python floats."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 2.1, 3.1])
    metrics = compute_metrics(y_true, y_pred)
    for value in metrics.values():
        assert isinstance(value, float)


def test_rmse_larger_than_mae() -> None:
    """RMSE should be >= MAE since it penalizes large errors more."""
    y_true = np.array([1.0, 2.0, 10.0])
    y_pred = np.array([1.1, 2.1, 8.0])
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["rmse"] >= metrics["mae"]


def test_constant_prediction_gives_negative_r2() -> None:
    """Predicting a constant should give R2 <= 0."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.full_like(y_true, fill_value=3.0)
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["r2"] <= 0.0
