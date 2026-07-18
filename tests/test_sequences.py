"""Unit tests for the make_sequences function."""

import numpy as np
import pandas as pd
import pytest

from src.features.sequences import make_sequences


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a sample preprocessed DataFrame for testing."""
    np.random.seed(42)
    n = 50
    return pd.DataFrame(
        {
            "Open": np.random.uniform(0, 1, n),
            "High": np.random.uniform(0, 1, n),
            "Low": np.random.uniform(0, 1, n),
            "Close": np.random.uniform(0, 1, n),
            "Volume": np.random.uniform(0, 1, n),
        }
    )


def test_output_shapes(sample_df: pd.DataFrame) -> None:
    """X and y should have correct shapes."""
    sequence_length = 3
    horizon = 1
    X, y = make_sequences(sample_df, sequence_length=sequence_length, horizon=horizon)

    n_samples = len(sample_df) - sequence_length - horizon + 1
    assert X.shape == (n_samples, sequence_length, 4)
    assert y.shape == (n_samples,)


def test_output_dtype(sample_df: pd.DataFrame) -> None:
    """X and y should be float32 arrays."""
    X, y = make_sequences(sample_df, sequence_length=3, horizon=1)
    assert X.dtype == np.float32
    assert y.dtype == np.float32


def test_correct_target_alignment(sample_df: pd.DataFrame) -> None:
    """y[i] should correspond to Close at position i + sequence_length + horizon - 1."""
    sequence_length = 3
    horizon = 1
    X, y = make_sequences(sample_df, sequence_length=sequence_length, horizon=horizon)

    close_values = sample_df["Close"].values
    for i in range(len(y)):
        expected = close_values[i + sequence_length + horizon - 1]
        assert y[i] == pytest.approx(expected, rel=1e-5)


def test_longer_horizon_reduces_samples(sample_df: pd.DataFrame) -> None:
    """A longer horizon should produce fewer samples."""
    _, y1 = make_sequences(sample_df, sequence_length=3, horizon=1)
    _, y7 = make_sequences(sample_df, sequence_length=3, horizon=7)
    assert len(y1) > len(y7)


def test_minimum_length_raises_or_returns_empty(sample_df: pd.DataFrame) -> None:
    """If sequence_length + horizon > len(df), should return empty arrays."""
    tiny_df = sample_df[:5]
    X, y = make_sequences(tiny_df, sequence_length=3, horizon=10)
    assert len(X) == 0
    assert len(y) == 0


def test_inference_window_uses_final_rows(sample_df: pd.DataFrame) -> None:
    """The inference window must come from the end of the data."""
    from src.features.sequences import make_inference_window

    window = make_inference_window(sample_df, sequence_length=3)

    assert window.shape == (1, 3, 4)
    expected = sample_df[["Open", "High", "Low", "Volume"]].values[-3:]
    assert np.allclose(window[0], expected)


def test_inference_window_differs_from_last_training_window(
    sample_df: pd.DataFrame,
) -> None:
    """The inference window should extend past the last window make_sequences
    can produce, since the latter is limited by the need for a target."""
    from src.features.sequences import make_inference_window

    X, _ = make_sequences(sample_df, sequence_length=3, horizon=7)
    window = make_inference_window(sample_df, sequence_length=3)

    assert not np.allclose(X[-1], window[0])


def test_inference_window_raises_when_too_short() -> None:
    """Too few rows should raise rather than silently return a short window."""
    from src.features.sequences import make_inference_window

    tiny = pd.DataFrame(
        {c: [1.0, 2.0] for c in ["Open", "High", "Low", "Close", "Volume"]}
    )
    with pytest.raises(ValueError):
        make_inference_window(tiny, sequence_length=5)
