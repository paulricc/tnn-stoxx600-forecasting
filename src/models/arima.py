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
