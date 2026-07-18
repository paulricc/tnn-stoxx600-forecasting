"""ARIMA model wrapper for time series forecasting."""

import logging

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA as StatsmodelsARIMA

logger = logging.getLogger(__name__)


class ARIMAModel:
    """Wrapper around statsmodels ARIMA with a scikit-learn-like interface.

    Attributes:
        order: The (p, d, q) order of the ARIMA model.
        fitted_model_: The fitted statsmodels results object, set after fit().
    """

    def __init__(self, order: tuple[int, int, int]) -> None:
        """Initialize the ARIMA model.

        Args:
            order: The (p, d, q) order, typically chosen via AIC/BIC grid search.
        """
        self.order = order
        self.fitted_model_ = None

    def fit(self, series: pd.Series) -> "ARIMAModel":
        """Fit the ARIMA model on a univariate time series.

        Args:
            series: Training time series (e.g. the Close price column).

        Returns:
            self
        """
        model = StatsmodelsARIMA(series, order=self.order)
        self.fitted_model_ = model.fit()
        logger.info("ARIMA%s fitted on %d observations", self.order, len(series))
        return self

    def predict(self, n_periods: int) -> np.ndarray:
        """Forecast future values.

        Args:
            n_periods: Number of steps ahead to forecast.

        Returns:
            Array of forecasted values of shape (n_periods,).

        Raises:
            RuntimeError: If called before fit.
        """
        if self.fitted_model_ is None:
            raise RuntimeError("ARIMAModel must be fitted before calling predict.")

        forecast = self.fitted_model_.forecast(steps=n_periods)
        return np.asarray(forecast)

    def rolling_forecast(
        self,
        train_series: pd.Series,
        test_series: pd.Series,
        horizon: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Produce rolling-origin forecasts over a test series.

        The model is fitted once on the training series. It then walks forward
        through the test series, forecasting `horizon` steps ahead from each
        origin and appending the observed value before moving on. Coefficients
        are estimated once and held fixed; only the state is updated.

        This gives ARIMA a number of predictions comparable to the sequence
        models, which are evaluated on rolling windows over the same test set.

        Args:
            train_series: Training time series.
            test_series: Test time series, following on from train_series.
            horizon: Number of steps ahead to forecast at each origin.

        Returns:
            Tuple of (y_true, y_pred), each of shape (n_origins,), where
            n_origins is len(test_series) - horizon + 1. Both are empty if the
            test series is shorter than the horizon.
        """
        results = StatsmodelsARIMA(train_series, order=self.order).fit()

        test_values = np.asarray(test_series, dtype=np.float64)
        n_origins = len(test_values) - horizon + 1

        if n_origins <= 0:
            logger.warning(
                "Test series of length %d is too short for horizon=%d",
                len(test_values),
                horizon,
            )
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

        y_true = np.empty(n_origins, dtype=np.float64)
        y_pred = np.empty(n_origins, dtype=np.float64)

        for i in range(n_origins):
            y_pred[i] = float(np.asarray(results.forecast(steps=horizon))[-1])
            y_true[i] = test_values[i + horizon - 1]
            results = results.append([test_values[i]], refit=False)

        self.fitted_model_ = results
        logger.info(
            "ARIMA%s rolling forecast: %d origins at horizon=%d",
            self.order,
            n_origins,
            horizon,
        )
        return y_true, y_pred
