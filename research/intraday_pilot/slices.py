"""Intraday pilot: each stock-day's returns over the parts of the session.

For every stock of each year's trading universe and every session, from the
stored 1-minute bars (prices as traded):

- on: the previous close to the open; fh: the first 30 minutes;
- mid: 30 minutes after the open to 60 minutes before the close;
- slh: the second-to-last half hour; lh: the last half hour.

The price at a moment is the close of the last bar that ended by then. Half
days close at 13:00, and every part is measured from the session's own
close. Also kept: the volume up to 30 minutes before the close, and the
realised variance of 5-minute returns up to then.

What spans days comes from the daily bars, whose prices are adjusted for
splits and dividends, so a split or an ex-dividend day makes no false move,
and a stock's previous session is its previous daily bar even when it was
not in the trading universe that day:

- on: the overnight return, yesterday's close to today's open;
- prev_day, prev_5d: the returns from close to close of the last 1 and 5
  sessions before today; vol21: the standard deviation of the last 21;
- split_factor: multiplying a day's volume by it gives today's share count,
  so volumes can be compared across a split.

Written as Parquet.
"""

import json
import sys
from datetime import date
from pathlib import Path

import polars as pl

from qfin_alphaguard.data.intraday import intraday_path
from qfin_alphaguard.sessions import sessions

ROOT = Path(sys.argv[1])  # data/raw
DAILY = Path(sys.argv[2])  # data/raw/prices.parquet
UNIVERSES = {int(k): v for k, v in json.loads(Path(sys.argv[3]).read_text()).items()}
OUT = Path(sys.argv[4])


def schedule(year: int) -> pl.DataFrame:
    found = sessions(date(year, 1, 1), date(year, 12, 31))
    return pl.DataFrame(
        {
            "day": [s.day for s in found],
            "open_at": [s.open for s in found],
            "length": [int((s.close - s.open).total_seconds() // 60) for s in found],
        },
        schema={
            "day": pl.Date,
            "open_at": pl.Datetime("us", "UTC"),
            "length": pl.Int64,
        },
    )


frames = []
for year, tickers in sorted(UNIVERSES.items()):
    files = [intraday_path(ROOT, t, year) for t in tickers]
    files = [f for f in files if f.exists()]
    if not files:
        continue
    k, length = pl.col("k"), pl.col("length")
    minutes = (
        pl.concat(
            [
                pl.scan_parquet(f).select(
                    "start", "ticker", "open", "high", "low", "close", "volume"
                )
                for f in files
            ]
        )
        .with_columns(pl.col("start").dt.date().alias("day"))
        .collect()
        .join(schedule(year), on="day")
        .with_columns(
            (pl.col("start") - pl.col("open_at")).dt.total_minutes().alias("k")
        )
        .filter((k >= 0) & (k < length))
        .sort("ticker", "start")
    )
    five = (k <= length - 31) & ((k + 1) % 5 == 0)
    frames.append(
        minutes.group_by("ticker", "day").agg(
            pl.col("open").first().alias("p_open"),
            pl.col("close").filter(k <= 29).last().alias("p_30"),
            pl.col("close").filter(k <= length - 61).last().alias("p_m60"),
            pl.col("close").filter(k <= length - 31).last().alias("p_m30"),
            pl.col("close").last().alias("p_close"),
            pl.col("volume").filter(k <= 29).sum().alias("vol_30"),
            pl.col("high").filter(k <= 29).max().alias("high_30"),
            pl.col("low").filter(k <= 29).min().alias("low_30"),
            pl.col("volume").filter(k <= length - 31).sum().alias("vol_to_m30"),
            pl.col("volume").sum().alias("vol_day"),
            (pl.col("close").filter(five).log().diff() ** 2).sum().alias("rv_to_m30"),
            pl.len().alias("minutes"),
            length.first().alias("length"),
        )
    )

c = pl.col("close")
daily = (
    pl.read_parquet(DAILY)
    .sort("ticker", "date")
    .with_columns((c / c.shift(1)).log().over("ticker").alias("cc"))
    .with_columns(
        (pl.col("open") / c.shift(1)).log().over("ticker").alias("on"),
        pl.col("cc").shift(1).over("ticker").alias("prev_day"),
        pl.col("cc").shift(1).rolling_sum(5).over("ticker").alias("prev_5d"),
        pl.col("cc").shift(1).rolling_std(21).over("ticker").alias("vol21"),
    )
    .select(
        "ticker",
        pl.col("date").alias("day"),
        "on",
        "prev_day",
        "prev_5d",
        "vol21",
        "split_factor",
    )
)


def ret(a: str, b: str) -> pl.Expr:
    return (pl.col(b) / pl.col(a)).log()


days = (
    pl.concat(frames)
    .join(daily, on=["ticker", "day"], how="left")
    .sort("ticker", "day")
    .with_columns(
        ret("p_open", "p_30").alias("fh"),
        ret("p_30", "p_m60").alias("mid"),
        ret("p_m60", "p_m30").alias("slh"),
        ret("p_m30", "p_close").alias("lh"),
    )
)
days.write_parquet(OUT)
print(days.height, "stock-days", days["day"].min(), "..", days["day"].max())
