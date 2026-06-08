import logging

import numpy as np
import pandas as pd

from src.data.preprocessor import FEATURES, TARGET

logger = logging.getLogger(__name__)


def make_sequences(
    df: pd.DataFrame, sequence_length: int, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """Create input/target sequences from a preprocessed DataFrame.

    For each time step t, the input is a window of `sequence_length` rows
    of features ending at t, and the target is the Close price at t + horizon.

    Args:
        df: Preprocessed DataFrame with OHLCV columns.
        sequence_length: Number of past time steps to use as input.
        horizon: Number of steps ahead to predict.

    Returns:
        Tuple of (X, y) where:
            X has shape (n_samples, sequence_length, n_features)
            y has shape (n_samples,)
    """
    n_samples = len(df) - sequence_length - horizon + 1

    X_array = np.zeros((n_samples, sequence_length, len(FEATURES)), dtype=np.float32)
    y_array = np.zeros(n_samples, dtype=np.float32)

    features = df[FEATURES].values
    target = df[TARGET].values

    for i in range(n_samples):
        X_array[i] = features[i : i + sequence_length]
        y_array[i] = target[i + sequence_length + horizon - 1]

    logger.info(f"Created {len(X_array)} sequences with shape {X_array.shape}")

    return X_array, y_array
