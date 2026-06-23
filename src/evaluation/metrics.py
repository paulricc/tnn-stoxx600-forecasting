import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute RMSE, MAE, MAPE, and R2 for a set of predictions.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        Dictionary with keys "rmse", "mae", "mape", "r2".
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = _mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "mape": float(mape),
        "r2": float(r2),
    }


def _mean_absolute_percentage_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1e-8,
) -> float:
    """Compute MAPE, guarding against division by zero.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        epsilon: Small constant added to the denominator to avoid division by zero.

    Returns:
        MAPE as a percentage.
    """
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100)
