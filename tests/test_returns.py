import numpy as np
import polars as pl

from qfin_alphaguard.config import Config
from qfin_alphaguard.data.synthetic import generate_prices
from qfin_alphaguard.features.returns import log_returns, to_matrix


def test_first_day_of_each_ticker_is_dropped():
    prices = generate_prices(Config())
    returns = log_returns(prices)
    n_tickers = prices["ticker"].n_unique()
    assert len(returns) == len(prices) - n_tickers


def test_returns_do_not_leak_across_tickers():
    prices = pl.DataFrame(
        {
            "date": [1, 2, 1, 2],
            "ticker": ["A", "A", "B", "B"],
            "close": [100.0, 110.0, 50.0, 55.0],
        }
    )
    returns = log_returns(prices)

    assert len(returns) == 2
    assert np.allclose(returns["log_return"].to_numpy(), [np.log(1.1), np.log(1.1)])


def test_matrix_shape_and_ticker_order():
    config = Config()
    returns = log_returns(generate_prices(config))
    matrix, tickers = to_matrix(returns)

    assert matrix.shape == (returns["date"].n_unique(), len(config.tickers))
    assert sorted(tickers) == sorted(config.tickers)
