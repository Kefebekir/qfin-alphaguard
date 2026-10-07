import numpy as np
import pytest

from qfin_alphaguard.optimize.min_variance import (
    min_variance_weights,
    portfolio_volatility,
)


@pytest.fixture
def cov():
    rng = np.random.default_rng(0)
    returns = rng.normal(0, 0.01, size=(500, 8))
    returns[:, 0] *= 0.5
    return np.cov(returns, rowvar=False) * 252


def test_weights_satisfy_constraints(cov):
    w = min_variance_weights(cov, max_weight=0.3)

    assert w.sum() == pytest.approx(1.0)
    assert (w >= 0).all()
    assert (w <= 0.3 + 1e-6).all()


def test_beats_equal_weight(cov):
    w = min_variance_weights(cov, max_weight=0.3)
    equal = np.full(len(w), 1 / len(w))

    assert portfolio_volatility(w, cov) <= portfolio_volatility(equal, cov)


def test_low_volatility_asset_gets_most_weight(cov):
    w = min_variance_weights(cov, max_weight=0.3)

    assert w.argmax() == 0


def test_impossible_cap_raises(cov):
    with pytest.raises(ValueError):
        min_variance_weights(cov, max_weight=0.1)
