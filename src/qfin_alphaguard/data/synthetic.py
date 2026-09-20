"""Generate synthetic price data for testing without network access."""

from datetime import date
import numpy as np
import polars as pl

from qfin_alphaguard.config import Config


def generate_prices(config: Config) -> pl.DataFrame:
    """Return daily close prices in long format:date,ticker,close."""
    rng = np.random.default_rng(config.seed)
    dates = pl.date_range(
        start=date.fromisoformat(config.start_date),
        end=date.fromisoformat(config.end_date),
        interval="1d",
        eager=True,
    )
    dates = dates.filter(dates.dt.weekday() <= 5)

    n_days = len(dates)
    n_assets = len(config.tickers)

    # One shared market factor and plus asset specific noise
    market = rng.normal(0.0003, 0.01, size=n_days)
    idiosyncratic = rng.normal(0.0, 0.015, size=(n_days, n_assets))
    betas = rng.uniform(0.6, 1.4, size=n_assets)

    returns = market[:, None] * betas + idiosyncratic
    prices = 100 * np.exp(np.cumsum(returns, axis=0))

    frames = []
    for i, ticker in enumerate(config.tickers):
        frames.append(
            pl.DataFrame(
                {"date": dates, "ticker": [ticker] * n_days, "close": prices[:, i]}
            )
        )
    return pl.concat(frames)


if __name__ == "__main__":
    df = generate_prices(Config())
    print(df.shape)
    print(df.head())
