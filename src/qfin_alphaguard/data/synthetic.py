"""Generate synthetic price data for testing without network access."""

from datetime import date

import numpy as np
import polars as pl

from qfin_alphaguard.config import Config
from qfin_alphaguard.sessions import session_days


def generate_prices(config: Config) -> pl.DataFrame:
    """Daily bars in the same columns as real data.

    Synthetic prices need no adjustment, every ticker counts as an index
    member on every day, and the days are the exchange's real sessions.
    """
    rng = np.random.default_rng(config.seed)
    # The exchange's real trading days, so holidays have no bars here either.
    dates = pl.Series(
        "date",
        session_days(
            date.fromisoformat(config.start_date), date.fromisoformat(config.end_date)
        ),
    )

    n_days = len(dates)
    n_assets = len(config.tickers)
    shape = (n_days, n_assets)

    # One shared market factor plus asset-specific noise
    market = rng.normal(0.0003, 0.01, size=n_days)
    idiosyncratic = rng.normal(0.0, 0.015, size=shape)
    betas = rng.uniform(0.6, 1.4, size=n_assets)

    returns = market[:, None] * betas + idiosyncratic
    closes = 100 * np.exp(np.cumsum(returns, axis=0))

    # Drawn after the closes, so a seed gives the same closes as before the
    # open, high, low and volume columns existed.
    previous_closes = np.vstack([np.full((1, n_assets), 100.0), closes[:-1]])
    opens = previous_closes * np.exp(rng.normal(0.0, 0.003, size=shape))
    highs = np.maximum(opens, closes) * np.exp(np.abs(rng.normal(0.0, 0.005, shape)))
    lows = np.minimum(opens, closes) * np.exp(-np.abs(rng.normal(0.0, 0.005, shape)))
    volumes = rng.integers(1_000_000, 10_000_000, size=shape)

    frames = []
    for i, ticker in enumerate(config.tickers):
        frames.append(
            pl.DataFrame(
                {
                    "date": dates,
                    "ticker": [ticker] * n_days,
                    "open": opens[:, i],
                    "high": highs[:, i],
                    "low": lows[:, i],
                    "close": closes[:, i],
                    "volume": volumes[:, i],
                    "adjustment": np.ones(n_days),
                    "split_factor": np.ones(n_days),
                    "sp500": [True] * n_days,
                }
            )
        )
    return pl.concat(frames).sort(["ticker", "date"])


if __name__ == "__main__":
    df = generate_prices(Config())
    print(df.shape)
    print(df.head())
