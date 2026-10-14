"""Intraday pilot: size QQQ momentum with a HAR forecast of tomorrow's variance.

Decision 0011 sizes each day at min(cap, 2% / the standard deviation of the
last 14 daily returns). The HAR model (Corsi, 2009) forecasts the next day's
realised variance from the daily, weekly and monthly averages of past
realised variance, here measured from 5-minute returns plus the return from
the previous close to the first 5-minute mark. It is refitted each year on
the years before it only. Both sizings use the same strategy days and costs.
"""

import math
import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from qfin_alphaguard.data.intraday import intraday_path
from qfin_alphaguard.sessions import sessions

ROOT = Path(sys.argv[1])
DAYS = Path(sys.argv[2])
CAP = 2.0
PER_SHARE, PER_VALUE = 0.001, 0.15e-4

opens = {s.day: s.open for s in sessions(date(2016, 1, 1), date(2026, 12, 31))}
bars = pl.concat(
    [
        pl.read_parquet(intraday_path(ROOT, "QQQ", y)).select("start", "close")
        for y in range(2016, 2027)
        if intraday_path(ROOT, "QQQ", y).exists()
    ]
).with_columns(pl.col("start").dt.date().alias("day"))
five = (
    bars.filter(pl.col("start").dt.minute() % 5 == 4)  # bars ending on a 5-minute mark
    .sort("start")
    .with_columns(pl.col("close").log().diff().over("day").alias("r5"))
)
rv = (
    five.group_by("day")
    .agg(
        (pl.col("r5") ** 2).sum().alias("rv_intraday"),
        pl.col("close").last().alias("last"),
    )
    .sort("day")
)
# The overnight term runs from the previous close to the day's first
# 5-minute mark, so with the 5-minute returns after it the day's realised
# variance covers the whole close-to-close move.
first = five.group_by("day").agg(pl.col("close").first().alias("first")).sort("day")
rv = (
    rv.join(first, on="day")
    .sort("day")
    .with_columns((pl.col("first") / pl.col("last").shift(1)).log().alias("overnight"))
    .with_columns((pl.col("rv_intraday") + pl.col("overnight") ** 2).alias("rv"))
)
rv = rv.with_columns(
    pl.col("rv").shift(1).alias("rv_d"),
    pl.col("rv").shift(1).rolling_mean(5).alias("rv_w"),
    pl.col("rv").shift(1).rolling_mean(22).alias("rv_m"),
).drop_nulls()

d = (
    pl.read_parquet(DAYS)
    .join(rv.select("day", "rv", "rv_d", "rv_w", "rv_m"), on="day")
    .sort("day")
)
years = d["day"].dt.year().to_numpy()
X = np.column_stack([np.ones(len(d)), d["rv_d"], d["rv_w"], d["rv_m"]])
y = d["rv"].to_numpy()
forecast = np.full(len(d), np.nan)
for year in range(2018, 2027):
    train, test = years < year, years == year
    if test.any():
        beta, *_ = np.linalg.lstsq(X[train], y[train], rcond=None)
        forecast[test] = np.maximum(X[test] @ beta, 1e-8)
tested = ~np.isnan(forecast)
gross = d["gross_1x"].to_numpy()
unit_cost = d["traded_1x"].to_numpy() * (PER_VALUE + PER_SHARE / d["price"].to_numpy())
sizes = {
    "14-day volatility": np.minimum(CAP, 0.02 / d["vol14"].to_numpy()),
    "HAR forecast": np.minimum(CAP, 0.02 / np.sqrt(forecast)),
}
for name, s in sizes.items():
    r = (gross * s - unit_cost * s)[tested]
    growth = np.prod(1 + r) ** (252 / len(r)) - 1
    sharpe = r.mean() / r.std(ddof=1) * math.sqrt(252)
    worst = np.min(np.cumprod(1 + r) / np.maximum.accumulate(np.cumprod(1 + r)) - 1)
    print(
        f"{name:18} {growth:+.1%} a year, Sharpe {sharpe:.2f}, max drawdown {worst:.1%}, "
        f"average size {s[tested].mean():.2f}x"
    )
# How well each predicts the next day's realised volatility.
real = np.sqrt(y[tested])
for name, pred in (
    ("14-day volatility", d["vol14"].to_numpy()[tested]),
    ("HAR forecast", np.sqrt(forecast[tested])),
):
    print(
        f"{name:18} correlation with the day's realised volatility {np.corrcoef(pred, real)[0, 1]:.3f}"
    )
