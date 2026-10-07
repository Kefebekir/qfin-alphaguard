"""Long-only minimum variance portfolio optimization."""

import cvxpy as cp
import numpy as np


def min_variance_weights(cov: np.ndarray, max_weight: float = 0.20) -> np.ndarray:
    """Weights that minimise portfolio variance.
    Fully invested, no short selling and no single asset can exceed max_weight."""

    n = cov.shape[0]
    if max_weight * n < 1:
        raise ValueError(f"max_weight {max_weight} is too low for {n} assets.")

    w = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov)))
    constraints = [cp.sum(w) == 1, w >= 0, w <= max_weight]
    problem = cp.Problem(objective, constraints)
    problem.solve()

    if problem.status != cp.OPTIMAL:
        raise RuntimeError(f"optimization failed with status {problem.status}")

    weights = np.clip(w.value, 0.0, None)
    return weights / weights.sum()


def portfolio_volatility(weights: np.ndarray, cov: np.ndarray) -> float:
    """Annualised volatility of a portfolio, given an annualised covariance."""
    return float(np.sqrt(weights @ cov @ weights))
