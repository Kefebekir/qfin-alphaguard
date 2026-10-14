"""Intraday pilot: is the opening-gap reversal in a price one can trade at?

Each morning, the S&P 500 members that fell most from yesterday's close to
today's open tend to rise from the open to the close, and those that rose
most tend to fall (open_to_close.py). A single wrong opening print would
show the same pattern without any profit to trade: it moves the gap one way
and the open-to-close return the other. So here the same stocks are bought at
later prices from the consolidated tape: the open of the second minute
(09:31), of the sixth (09:35) and of the sixteenth (09:45), each held to
the official close. If the profit lives in prices one
can trade at, it stays; if it is a bad print, it is gone by 09:31.

The signal is the official open against yesterday's official close (both
from the daily bars), as known when the opening auction ends. Long the k
largest falls; long and short adds the k largest rises.

Usage: gap_check.py PRICES MINUTES_DIR (opening_minutes.py's output)
"""

import math
import sys
from datetime import date
from pathlib import Path

import polars as pl

from qfin_alphaguard.data.universe import dollar_volume
from qfin_alphaguard.sessions import sessions

PRICES, MINUTES = Path(sys.argv[1]), Path(sys.argv[2])
K = 10
ENTRIES = {"open": None, "09:31": 1, "09:35": 5, "09:45": 15}

opens = pl.DataFrame(
    [
        {"day": s.day, "open_at": s.open}
        for s in sessions(date(2016, 1, 1), date(2026, 12, 31))
    ],
    schema={"day": pl.Date, "open_at": pl.Datetime("us", "UTC")},
)
daily = (
    pl.read_parquet(PRICES)
    .sort("ticker", "date")
    .with_columns(
        (pl.col("open") / pl.col("close").shift(1).over("ticker") - 1).alias("gap"),
        (pl.col("open") / pl.col("adjustment")).alias("open_traded"),
        (pl.col("close") / pl.col("adjustment")).alias("close_traded"),
        dollar_volume().shift(1).rolling_mean(20).over("ticker").alias("dollar_volume"),
    )
    .filter(pl.col("sp500") & (pl.col("volume") > 0) & pl.col("gap").is_not_null())
    .select(
        pl.col("date").alias("day"),
        "ticker",
        "gap",
        "open_traded",
        "close_traded",
        "dollar_volume",
    )
)
minutes = (
    pl.scan_parquet(MINUTES / "*.parquet")
    .join(opens.lazy(), on="day")
    .with_columns(((pl.col("start") - pl.col("open_at")).dt.total_minutes()).alias("k"))
    .sort("start")
    .collect()
)
prices = minutes.group_by("day", "ticker").agg(
    pl.col("open").filter(pl.col("k") == 0).first().alias("tape_open"),
    *[
        # the first trade at or after the entry minute: what a market order
        # sent then would meet first
        pl.col("open").filter(pl.col("k") >= k).first().alias(name)
        for name, k in ENTRIES.items()
        if k is not None
    ],
)
d = (
    daily.join(prices, on=["day", "ticker"], how="inner")
    # The tape must agree with the daily bars about the open, within 1%:
    # otherwise the minutes are another company's or the day's prices differ.
    .filter((pl.col("tape_open") / pl.col("open_traded") - 1).abs() < 0.01)
    .rename({"open_traded": "open"})
    .drop_nulls()
)
print(
    f"{d.height} member-days with the first 16 minutes, {d['day'].min()} .. {d['day'].max()}"
)
first_minute = (d["tape_open"] / d["open"] - 1).abs()
print(
    f"official open against the tape's first trade: median {first_minute.median() * 1e4:.1f} bps, "
    f"90th percentile {first_minute.quantile(0.9) * 1e4:.1f} bps"
)

for universe, frame in (
    ("all members", d),
    (
        "100 most traded",
        d.filter(pl.col("dollar_volume").rank(descending=True).over("day") <= 100),
    ),
):
    ranked = frame.with_columns(
        pl.col("gap").rank("ordinal").over("day").alias("fell"),
        pl.col("gap").rank("ordinal", descending=True).over("day").alias("rose"),
    )
    print(f"-- {universe}: top {K}, bps a day held to the official close")
    for name in ENTRIES:
        ret = pl.col("close_traded") / pl.col(name) - 1
        legs = (
            ranked.group_by("day")
            .agg(
                ret.filter(pl.col("fell") <= K).mean().alias("long"),
                ret.filter(pl.col("rose") <= K).mean().alias("short"),
                ret.mean().alias("all"),
            )
            .sort("day")
        )
        long, short, every = (legs[c].to_numpy() for c in ("long", "short", "all"))
        excess = long - every
        spread = long - short
        print(
            f"   from {name:6}: long {long.mean() * 1e4:+6.1f}, over all {excess.mean() * 1e4:+6.1f} "
            f"(t {excess.mean() / excess.std(ddof=1) * math.sqrt(len(excess)):+.1f}), "
            f"long minus short {spread.mean() * 1e4:+6.1f} "
            f"(t {spread.mean() / spread.std(ddof=1) * math.sqrt(len(spread)):+.1f}); "
            f"Sharpe of long minus short {spread.mean() / spread.std(ddof=1) * math.sqrt(252):+.2f}"
        )
    years = ranked.with_columns(pl.col("day").dt.year().alias("year"))
    by_year = (
        years.group_by("day", "year")
        .agg(
            (pl.col("close_traded") / pl.col("09:35") - 1)
            .filter(pl.col("fell") <= K)
            .mean()
            .alias("long"),
            (pl.col("close_traded") / pl.col("09:35") - 1)
            .filter(pl.col("rose") <= K)
            .mean()
            .alias("short"),
        )
        .group_by("year")
        .agg(
            ((pl.col("long") - pl.col("short")).mean() * 1e4)
            .round(1)
            .alias("long_minus_short_from_0935")
        )
        .sort("year")
    )
    print("   by year, from 09:35:", dict(by_year.iter_rows()))
