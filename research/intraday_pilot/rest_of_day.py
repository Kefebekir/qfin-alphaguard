"""Intraday pilot: at 10:00, which of our 50 stocks will do best by the close?

Features known at 10:00: the opening gap, the first 30 minutes' return and
range, their volume against the same 30 minutes of the last 14 days (how
much a stock is "in play"), the previous day's and last 5 days' returns, the
last 21 days' volatility, and the same stock's moves against the universe's
average. Target: the return from 10:00 to the close. Walk-forward by year,
on earlier years only. Long only: buy the top k at 10:00, sell at the close,
against the universe over the same hours. Costs: a round trip of c bps.

The previous days' returns and volatility come from the adjusted daily bars
(slices.py). Volumes are put in today's share count first, so a split does
not look like a burst of trading; and a stock back in the universe after a
gap gets no "in play" value until it has 14 recent sessions.
"""

import math
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

SLICES = Path(sys.argv[1])
COSTS = (0.0, 2.0, 3.0, 5.0)

d = (
    pl.read_parquet(SLICES)
    .filter(pl.col("minutes") >= pl.col("length") - 5)
    .sort("ticker", "day")
    .with_columns(
        (pl.col("p_close") / pl.col("p_30")).log().alias("rest"),
        ((pl.col("high_30") - pl.col("low_30")) / pl.col("p_open")).alias("range30"),
        (pl.col("p_close") / pl.col("p_30") - 1).alias("rest_simple"),
    )
    .with_columns(
        pl.when(
            (pl.col("day") - pl.col("day").shift(14).over("ticker")).dt.total_days()
            <= 30
        )
        .then(
            (
                pl.col("vol_30")
                * pl.col("split_factor")
                / (pl.col("vol_30") * pl.col("split_factor"))
                .shift(1)
                .rolling_mean(14)
                .over("ticker")
            ).log()
        )
        .alias("in_play"),
    )
    .with_columns(
        (pl.col("fh") - pl.col("fh").mean().over("day")).alias("fh_rel"),
        (pl.col("on") - pl.col("on").mean().over("day")).alias("gap_rel"),
        pl.col("fh").mean().over("day").alias("market_fh"),
        pl.col("on").mean().over("day").alias("market_gap"),
        (pl.col("fh") / pl.col("vol21")).alias("fh_in_vol"),
        (pl.col("on") / pl.col("vol21")).alias("gap_in_vol"),
    )
    .drop_nulls()
)
FEATURES = [
    "on",
    "fh",
    "range30",
    "in_play",
    "prev_day",
    "prev_5d",
    "vol21",
    "fh_rel",
    "gap_rel",
    "market_fh",
    "market_gap",
    "fh_in_vol",
    "gap_in_vol",
]
d = d.with_columns(
    ((pl.col("rest").rank().over("day") - 1) / (pl.len().over("day") - 1) - 0.5).alias(
        "y"
    )
)
years = d["day"].dt.year().to_numpy()
X, y = d.select(FEATURES).to_numpy(), d["y"].to_numpy()
pred = {"ridge": np.full(len(d), np.nan), "gbt": np.full(len(d), np.nan)}
for year in range(2018, 2027):
    train, test = years < year, years == year
    if not test.any():
        continue
    mean, sd = X[train].mean(0), X[train].std(0) + 1e-12
    for name, model in (
        ("ridge", Ridge(alpha=10.0)),
        (
            "gbt",
            HistGradientBoostingRegressor(
                max_iter=300,
                learning_rate=0.05,
                max_leaf_nodes=31,
                min_samples_leaf=500,
                l2_regularization=1.0,
                early_stopping=False,
                random_state=0,
            ),
        ),
    ):
        model.fit((X[train] - mean) / sd, y[train])
        pred[name][test] = model.predict((X[test] - mean) / sd)
d = d.with_columns(
    pl.Series("ridge", pred["ridge"]),
    pl.Series("gbt", pred["gbt"]),
    pl.col("fh").alias("morning_momentum"),
    (-pl.col("fh")).alias("morning_reversal"),
    (pl.col("in_play") * pl.col("fh").sign()).alias("in_play_momentum"),
).filter(pl.col("day").dt.year() >= 2018)

universe = d.group_by("day").agg(pl.col("rest_simple").mean().alias("all"))
print("10:00 to the close, 2018-2026, our universe (about 50 stocks a day)")
for signal in (
    "ridge",
    "gbt",
    "morning_momentum",
    "morning_reversal",
    "in_play_momentum",
):
    ic = d.group_by("day").agg(
        pl.corr(pl.col(signal).rank(), pl.col("rest").rank()).alias("ic")
    )["ic"]
    ic = ic.drop_nulls().to_numpy()
    print(
        f"{signal:17} rank IC {ic.mean():+.4f} (t {ic.mean() / ic.std(ddof=1) * math.sqrt(len(ic)):+.1f})"
    )
    for k in (5,):
        picks = (
            d.sort(signal, descending=True)
            .group_by("day", maintain_order=True)
            .head(k)
            .group_by("day")
            .agg(pl.col("rest_simple").mean().alias("pick"))
            .join(universe, on="day")
            .sort("day")
        )
        excess = (picks["pick"] - picks["all"]).to_numpy()
        raw = picks["pick"].to_numpy()
        line = f"   top {k}: excess {excess.mean() * 1e4:+.2f} bps a day (t {excess.mean() / excess.std(ddof=1) * math.sqrt(len(excess)):+.1f})"
        for c in COSTS:
            net = raw - c / 1e4
            line += f" | {c:g} bps: {net.mean() * 252:+.1%} a year, Sharpe {net.mean() / net.std(ddof=1) * math.sqrt(252):+.2f}"
        print(line)
