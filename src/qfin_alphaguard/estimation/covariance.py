"""Covariance estimator for daily return matrices."""

import numpy as np
from sklearn.covariance import LedoitWolf

TRADING_DAYS = 252


def sample_covariance(returns: np.ndarray) -> np.ndarray:
    """Plain sample covariance,annualized."""
    return np.cov(returns, rowvar=False) * TRADING_DAYS


def ledoit_wolf_covariance(returns: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit wolf shrunk covariance , annualized, and the shrinkage intesity."""
    model = LedoitWolf().fit(returns)
    return model.covariance_ * TRADING_DAYS, float(model.shrinkage_)
