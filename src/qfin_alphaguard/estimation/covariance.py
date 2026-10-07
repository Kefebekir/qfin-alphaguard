"""Covariance estimator for daily return matrices."""

import numpy as np
from sklearn.covariance import LedoitWolf

TRADING_DAYS = 252


def sample_covariance(returns: np.ndarray) -> np.ndarray:
    """Plain sample covariance, annualised."""
    return np.cov(returns, rowvar=False) * TRADING_DAYS


def ledoit_wolf_covariance(returns: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf shrunk covariance, annualised, and the shrinkage intensity."""
    model = LedoitWolf().fit(returns)
    return model.covariance_ * TRADING_DAYS, float(model.shrinkage_)


def ewma_covariance(returns: np.ndarray, halflife: float = 63) -> np.ndarray:
    """Exponentially weighted covariance, annualised. Recent days count more."""
    n_days = returns.shape[0]
    decay = 0.5 ** (1 / halflife)
    weights = decay ** np.arange(n_days - 1, -1, -1)
    weights /= weights.sum()

    mean = weights @ returns
    centred = returns - mean
    return (centred.T * weights) @ centred * TRADING_DAYS
