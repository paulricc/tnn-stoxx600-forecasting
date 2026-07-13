from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd

logger = logging.getLogger(__name__)

FEATURES = ["Open", "High", "Low", "Volume"]
TARGET = "Close"
COLS = FEATURES + [TARGET]


class Preprocessor:
    """Handles outlier removal, missing value interpolation, and min-max normalization.

    Follows the scikit-learn fit/transform pattern to prevent data leakage.

    Attributes:
        min_: Minimum values computed on the training set.
        max_: Maximum values computed on the training set.
        is_fitted_: Whether the preprocessor has been fitted.
    """

    def __init__(self):
        self.min_: pd.Series | None = None
        self.max_: pd.Series | None = None
        self.is_fitted_: bool = False

    def fit(self, df: pd.DataFrame) -> Preprocessor:
        """Compute min and max from training data.

        Args:
            df: Training DataFrame with OHLCV columns.

        Returns:
            self
        """
        self.min_ = df[COLS].min()
        self.max_ = df[COLS].max()
        self.is_fitted_ = True
        logger.info(f"Preprocessor fitted on {len(df)} rows")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing to a DataFrame using fitted parameters.

        Args:
            df: DataFrame to preprocess.

        Returns:
            Preprocessed DataFrame.

        Raises:
            RuntimeError: If called before fit.
        """
        if not self.is_fitted_:
            raise RuntimeError("Preprocessor must be fitted before calling transform")
        df = df.copy()
        df = self._remove_outliers(df)
        df = self._interpolate_missing(df)
        df = self._normalize(df)
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step. Use only on training data.

        Args:
            df: Training DataFrame.

        Returns:
            Preprocessed training DataFrame.
        """
        return self.fit(df).transform(df)

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace outliers beyond 3 standard deviations with the column median.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with outliers replaced.
        """
        for col in COLS:
            mean = df[col].mean()
            std = df[col].std()
            median = df[col].median()
            mask = (df[col] < mean - 3 * std) | (df[col] > mean + 3 * std)
            n_outliers = mask.sum()

            if n_outliers > 0:
                logger.info(f"Column {col}: replacing {n_outliers} outliers")

            df.loc[mask, col] = median
        return df

    def _interpolate_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values using linear interpolation.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with missing values filled.
        """
        n_missing = df.isnull().sum().sum()
        if n_missing > 0:
            logger.info(f"Interpolating {n_missing} missing values")

        df = df.interpolate(method="linear")
        return df

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply min-max normalization using fitted parameters.

        Args:
            df: Input DataFrame.

        Returns:
            Normalized DataFrame.
        """
        assert self.min_ is not None and self.max_ is not None
        df[COLS] = (df[COLS] - self.min_[COLS]) / (self.max_[COLS] - self.min_[COLS])
        return df

    def save(self, path: Path) -> None:
        """Save the fitted preprocessor to disk.

        Args:
            path: Path where the preprocessor will be saved.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Preprocessor saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> Preprocessor:
        """Load a fitted preprocessor from disk.

        Args:
            path: Path to the saved preprocessor.

        Returns:
            Loaded Preprocessor instance.
        """
        prepocessor = joblib.load(path)
        logger.info(f"Preprocessor loaded from {path}")
        return prepocessor
