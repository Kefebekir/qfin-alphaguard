"""Intraday pilot: at 09:35, which S&P 500 members will rise most by the close?

The opening-gap reversal (gap_check.py) lives mostly in the first minutes.
Here a model looks at every member at 09:35, when a small account can trade,
with what is known then:

- the opening gap, and the move and range from the open to 09:35;
- the first 5 minutes' volume against the same 5 minutes of the last 14
  sessions (how much the stock is "in play");
- yesterday's and the last 5 days' returns, the last 20 days' volatility,
  and the gap and the move in units of it;
- the same against all members' average, and that average itself;
- the stock's dollar volume and price.

Target: the return from the first trade at or after 09:35 to the official
close, ranked across members. Ridge and gradient-boosted trees, walk-forward
by year (trained on the years before), against the gap alone. Long only, the
k highest each day, among all members and among the 100 most traded.
Volumes are put in today's share count before they are compared across
days, so a split does not look like a stock in play; a stock back in the
index after a gap gets no "in play" value until it has 14 recent sessions.
The scores are written to OUT, for gap_quotes.py and gap_costs.py.

Usage: morning_ml.py PRICES MINUTES_DIR OUT
"""

import math
import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from qfin_alphaguard.data.universe import dollar_volume
from qfin_alphaguard.sessions import sessions

PRICES, MINUTES, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
K = 10
ENTRY = 5  # minutes after the open

opens = pl.DataFrame(
    [
        {"day": s.day, "open_at": s.open}
        for s in sessions(date(2016, 1, 1), date(2026, 12, 31))
    ],
    schema={"day": pl.Date, "open_at": pl.Datetime("us", "UTC")},
)
c = pl.col("close")
daily = (
    pl.read_parquet(PRICES)
    .sort("ticker", "date")
    .with_columns(
        (pl.col("open") / c.shift(1).over("ticker") - 1).alias("gap"),
        (c / c.shift(1) - 1).shift(1).over("ticker").alias("yesterday"),
        (c.shift(1) / c.shift(6) - 1).over("ticker").alias("last_5d"),
        (c / c.shift(1)).log().shift(1).rolling_std(20).over("ticker").alias("vol20"),
        (pl.col("open") / pl.col("adjustment")).alias("open_traded"),
        (c / pl.col("adjustment")).alias("close_traded"),
        dollar_volume().shift(1).rolling_mean(20).over("ticker").alias("dollar_volume"),
    )
    .filter(pl.col("sp500") & (pl.col("volume") > 0))
    .select(
        pl.col("date").alias("day"),
        "ticker",
        "gap",
        "yesterday",
        "last_5d",
        "vol20",
        "open_traded",
        "close_traded",
        "dollar_volume",
        "split_factor",
    )
)
k = pl.col("k")
first = (
    pl.scan_parquet(MINUTES / "*.parquet")
    .join(opens.lazy(), on="day")
    .with_columns(((pl.col("start") - pl.col("open_at")).dt.total_minutes()).alias("k"))
    .sort("start")
    .group_by("day", "ticker")
    .agg(
        pl.col("open").filter(k == 0).first().alias("tape_open"),
        pl.col("close").filter(k < ENTRY).last().alias("p_0935"),
        pl.col("high").filter(k < ENTRY).max().alias("high5"),
        pl.col("low").filter(k < ENTRY).min().alias("low5"),
        pl.col("volume").filter(k < ENTRY).sum().alias("volume5"),
        pl.col("open").filter(k >= ENTRY).first().alias("entry"),
    )
    .collect()
    .join(daily.select("day", "ticker", "split_factor"), on=["day", "ticker"])
    .sort("ticker", "day")
    .with_columns((pl.col("volume5") * pl.col("split_factor")).alias("shares5_now"))
    .with_columns(
        pl.when(
            (pl.col("day") - pl.col("day").shift(14).over("ticker")).dt.total_days()
            <= 30
        )
        .then(
            (
                pl.col("shares5_now")
                / pl.col("shares5_now").shift(1).rolling_mean(14).over("ticker")
            ).log()
        )
        .alias("in_play")
    )
    .drop("split_factor")
)
d = (
    daily.join(first, on=["day", "ticker"], how="inner")
    .filter((pl.col("tape_open") / pl.col("open_traded") - 1).abs() < 0.01)
    .with_columns(
        (pl.col("p_0935") / pl.col("open_traded") - 1).alias("move5"),
        ((pl.col("high5") - pl.col("low5")) / pl.col("open_traded")).alias("range5"),
        (pl.col("close_traded") / pl.col("entry") - 1).alias("target"),
        pl.col("dollar_volume").log().alias("size"),
        pl.col("open_traded").log().alias("price"),
    )
    .with_columns(
        (pl.col("gap") / pl.col("vol20")).alias("gap_z"),
        (pl.col("move5") / pl.col("vol20")).alias("move5_z"),
        (pl.col("gap") - pl.col("gap").mean().over("day")).alias("gap_rel"),
        (pl.col("move5") - pl.col("move5").mean().over("day")).alias("move5_rel"),
        pl.col("gap").mean().over("day").alias("market_gap"),
        pl.col("move5").mean().over("day").alias("market_move5"),
        (pl.col("dollar_volume").rank(descending=True).over("day") <= 100).alias(
            "top100"
        ),
    )
    .fill_nan(None)
    .drop_nulls()
    .filter(pl.col("in_play").is_finite())
)
FEATURES = [
    "gap",
    "move5",
    "range5",
    "in_play",
    "yesterday",
    "last_5d",
    "vol20",
    "gap_z",
    "move5_z",
    "gap_rel",
    "move5_rel",
    "market_gap",
    "market_move5",
    "size",
    "price",
]
d = d.with_columns(
    (
        (pl.col("target").rank().over("day") - 1) / (pl.len().over("day") - 1) - 0.5
    ).alias("y")
).sort("day", "ticker")
print(f"{d.height} member-days, {d['day'].min()} .. {d['day'].max()}")
X, y = d.select(FEATURES).to_numpy(), d["y"].to_numpy()
years = d["day"].dt.year().to_numpy()
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
                min_samples_leaf=1000,
                l2_regularization=1.0,
                early_stopping=False,
                random_state=0,
            ),
        ),
    ):
        model.fit((X[train] - mean) / sd, y[train])
        pred[name][test] = model.predict((X[test] - mean) / sd)
    print(f"  {year} done", flush=True)

d = d.with_columns(
    pl.Series("ridge", pred["ridge"]),
    pl.Series("gbt", pred["gbt"]),
    (-pl.col("gap")).alias("gap_reversal"),
    (-pl.col("move5")).alias("move5_reversal"),
).filter(pl.col("day").dt.year() >= 2018)
d.select(
    "day", "ticker", "target", "ridge", "gbt", "gap_reversal", "top100"
).write_parquet(OUT)
print(
    f"From 09:35 to the official close, 2018-2026 out of sample, top {K}, before costs"
)
for universe, frame in (("all members", d), ("100 most traded", d.filter("top100"))):
    print(f"-- {universe}")
    for signal in ("ridge", "gbt", "gap_reversal", "move5_reversal"):
        ic = (
            frame.group_by("day")
            .agg(pl.corr(pl.col(signal).rank(), pl.col("target").rank()).alias("ic"))[
                "ic"
            ]
            .drop_nulls()
            .to_numpy()
        )
        days = (
            frame.with_columns(
                pl.col(signal).rank("ordinal", descending=True).over("day").alias("r")
            )
            .group_by("day")
            .agg(
                pl.col("target").filter(pl.col("r") <= K).mean().alias("long"),
                pl.col("target").mean().alias("all"),
            )
            .sort("day")
        )
        long, every = days["long"].to_numpy(), days["all"].to_numpy()
        excess = long - every
        print(
            f"   {signal:15} rank IC {ic.mean():+.4f} (t {ic.mean() / ic.std(ddof=1) * math.sqrt(len(ic)):+.1f}); "
            f"long {long.mean() * 1e4:+.1f} bps a day, over all {excess.mean() * 1e4:+.1f} "
            f"(t {excess.mean() / excess.std(ddof=1) * math.sqrt(len(excess)):+.1f}), "
            f"Sharpe of long {long.mean() / long.std(ddof=1) * math.sqrt(252):+.2f}"
        )
