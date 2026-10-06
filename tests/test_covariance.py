import numpy as np

from qfin_alphaguard.config import Config
from qfin_alphaguard.data.synthetic import generate_prices
from qfin_alphaguard.estimation.covariance import (
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
