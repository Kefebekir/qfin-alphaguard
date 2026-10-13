"""Shared test data: a small store of daily and minute bars, as qfin writes it."""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from qfin_alphaguard.data.intraday import STORED_SCHEMA, intraday_path
from qfin_alphaguard.sessions import session_days, sessions

DAILY_SCHEMA = {
    "date": pl.Date,
    "ticker": pl.String,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Int64,
    "adjustment": pl.Float64,
    "split_factor": pl.Float64,
    "sp500": pl.Boolean,
}
FIRST_DAILY = date(2015, 1, 2)
LAST_DAILY = date(2016, 1, 8)
MINUTE_DAYS = (date(2016, 1, 5), date(2016, 1, 6), date(2016, 1, 7))


def daily_bars(ticker, days, closes, member=True):
    """Daily bars closing at `closes`, without splits or dividends."""
    closes = np.asarray(closes, dtype=float)
    return pl.DataFrame(
        {
            "date": days,
            "ticker": [ticker] * len(days),
            "open": closes,
            "high": closes * 1.005,
            "low": closes * 0.995,
            "close": closes,
            "volume": [1_000_000] * len(days),
            "adjustment": [1.0] * len(days),
            "split_factor": [1.0] * len(days),
            "sp500": [member] * len(days),
        },
        schema=DAILY_SCHEMA,
    )


def minute_bars(ticker, day, price):
    """Every minute of the session on `day`, flat at `price`."""
    (session,) = sessions(day, day)
    starts = [
        session.open + timedelta(minutes=minute)
        for minute in range((session.close - session.open) // timedelta(minutes=1))
    ]
    n = len(starts)
    return pl.DataFrame(
        {
            "start": starts,
            "ticker": [ticker] * n,
            "open": [price] * n,
            "high": [price] * n,
            "low": [price] * n,
            "close": [price] * n,
            "volume": [1_000] * n,
            "source": ["EODHD"] * n,
        },
        schema=STORED_SCHEMA,
    )


def build_store(raw: Path) -> pl.DataFrame:
    """Daily bars for A, B and SPY from 2 January 2015 to 8 January 2016, and
    A's and B's minutes on 5, 6 and 7 January 2016, under `raw`."""
    days = session_days(FIRST_DAILY, LAST_DAILY)
    rng = np.random.default_rng(7)
    frames = []
    for ticker, start, member in (
        ("A", 100.0, True),
        ("B", 50.0, True),
        ("SPY", 200.0, False),
    ):
        steps = rng.normal(0.0005, 0.012, len(days))
        frames.append(
            daily_bars(ticker, days, start * np.exp(np.cumsum(steps)), member)
        )
    daily = pl.concat(frames).sort("ticker", "date")
    raw.mkdir(parents=True, exist_ok=True)
    daily.write_parquet(raw / "prices.parquet")
    for ticker in ("A", "B"):
        closes = dict(
            daily.filter(pl.col("ticker") == ticker).select("date", "close").rows()
        )
        minutes = pl.concat(
            [minute_bars(ticker, day, closes[day]) for day in MINUTE_DAYS]
        )
        path = intraday_path(raw, ticker, 2016)
        path.parent.mkdir(parents=True, exist_ok=True)
        minutes.write_parquet(path)
    return daily


@pytest.fixture
def store(tmp_path):
    """The small store under tmp_path/data/raw: (its root, the daily bars)."""
    raw = tmp_path / "data" / "raw"
    return raw, build_store(raw)


@pytest.fixture
def make_daily():
    """daily_bars, for tests that need bars of their own."""
    return daily_bars
