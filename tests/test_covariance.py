import numpy as np

from qfin_alphaguard.config import Config
from qfin_alphaguard.data.synthetic import generate_prices
from qfin_alphaguard.estimation.covariance import (
    ewma_covariance,
    ledoit_wolf_covariance,
    sample_covariance,
)
from qfin_alphaguard.features.returns import log_returns, to_matrix


def _matrix():
    matrix, _ = to_matrix(log_returns(generate_prices(Config())))
    return matrix


def test_sample_covariance_shape_and_symmetry():
    cov = sample_covariance(_matrix())
    n = len(Config().tickers)

    assert cov.shape == (n, n)
    assert np.allclose(cov, cov.T)
    assert (np.diag(cov) > 0).all()


def test_ledoit_wolf_is_no_worse_contditioned():
    matrix = _matrix()
    sample = sample_covariance(matrix)
    shrunk, delta = ledoit_wolf_covariance(matrix)

    assert 0.0 <= delta <= 1.0
    assert np.linalg.cond(shrunk) <= np.linalg.cond(sample)


def test_shrinkage_grows_when_data_is_scarce():
    matrix = _matrix()
    _, delta_full = ledoit_wolf_covariance(matrix)
    _, delta_short = ledoit_wolf_covariance(matrix[:60])

    assert delta_short > delta_full


def test_ewma_shape_and_symmetry():
    cov = ewma_covariance(_matrix())
    n = len(Config().tickers)

    assert cov.shape == (n, n)
    assert np.allclose(cov, cov.T)
    assert (np.diag(cov) > 0).all()


def test_ewma_with_huge_halflife_matches_equal_weights():
    matrix = _matrix()
    ewma = ewma_covariance(matrix, halflife=1e9)
    equal = np.cov(matrix, rowvar=False, bias=True) * 252

    assert np.allclose(ewma, equal, rtol=1e-3)


def test_ewma_reacts_to_recent_shock():
    rng = np.random.default_rng(1)
    returns = rng.normal(0, 0.01, size=(1000, 3))
    returns[-20:, 0] *= 5

    ewma = ewma_covariance(returns)
    sample = sample_covariance(returns)

    assert ewma[0, 0] > 2 * sample[0, 0]
