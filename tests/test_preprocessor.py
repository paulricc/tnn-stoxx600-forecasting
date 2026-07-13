"""Unit tests for the Preprocessor class."""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessor import COLS, Preprocessor


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame(
        {
            "Open": np.random.uniform(300, 400, n),
            "High": np.random.uniform(310, 410, n),
            "Low": np.random.uniform(290, 390, n),
            "Close": np.random.uniform(300, 400, n),
            "Volume": np.random.uniform(1e6, 1e7, n),
        }
    )


def test_fit_sets_min_max(sample_df: pd.DataFrame) -> None:
    """fit() should compute and store min and max for each column."""
    preprocessor = Preprocessor()
    preprocessor.fit(sample_df)

    assert preprocessor.min_ is not None
    assert preprocessor.max_ is not None
    assert list(preprocessor.min_.index) == COLS
    assert list(preprocessor.max_.index) == COLS


def test_fit_returns_self(sample_df: pd.DataFrame) -> None:
    """fit() should return self to allow method chaining."""
    preprocessor = Preprocessor()
    result = preprocessor.fit(sample_df)
    assert result is preprocessor


def test_transform_raises_if_not_fitted(sample_df: pd.DataFrame) -> None:
    """transform() should raise RuntimeError if called before fit()."""
    preprocessor = Preprocessor()
    with pytest.raises(RuntimeError):
        preprocessor.transform(sample_df)


def test_normalized_values_between_zero_and_one(sample_df: pd.DataFrame) -> None:
    """After fit_transform, all values should be in [0, 1]."""
    preprocessor = Preprocessor()
    df_processed = preprocessor.fit_transform(sample_df)

    assert df_processed[COLS].min().min() >= 0.0
    assert df_processed[COLS].max().max() <= 1.0


def test_transform_does_not_modify_original(sample_df: pd.DataFrame) -> None:
    """transform() should not modify the original DataFrame."""
    preprocessor = Preprocessor()
    preprocessor.fit(sample_df)
    original_values = sample_df[COLS].copy()
    preprocessor.transform(sample_df)

    pd.testing.assert_frame_equal(sample_df[COLS], original_values)


def test_test_set_uses_train_min_max(sample_df: pd.DataFrame) -> None:
    """transform() on test set should use min/max from training set."""
    train = sample_df[:80]
    test = sample_df[80:]

    preprocessor = Preprocessor()
    preprocessor.fit(train)

    assert preprocessor.min_ is not None
    train_min = preprocessor.min_.copy()
    preprocessor.transform(test)

    pd.testing.assert_series_equal(preprocessor.min_, train_min)


def test_outlier_removal_replaces_with_median(sample_df: pd.DataFrame) -> None:
    """Outliers beyond 3 std should be replaced with the column median."""
    preprocessor = Preprocessor()
    sample_df_with_outlier = sample_df.copy()
    sample_df_with_outlier.loc[0, "Close"] = 999999.0

    median_before = sample_df_with_outlier["Close"].median()
    result = preprocessor._remove_outliers(sample_df_with_outlier)

    assert result.loc[0, "Close"] == pytest.approx(median_before, rel=1e-3)
