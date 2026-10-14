"""Morning picks after the spread: buy at the ask, sell at the bid.

Reads gap_quotes.py's quotes (its folder of yearly files). Each day, among
all S&P 500 members and among the 100 most traded, the k largest falls from
yesterday's official close to today's official open are bought, and (long
and short) the k largest rises are sold short. With --scores, the k stocks
with the highest score are bought instead (long only), such as
morning_ml.py's model, ranked among the stocks it scored that day:

- entry at 09:31 or 09:35: a market order, at the ask to buy, the bid to sell;
- exit at 15:59 by a market order (bid to sell, ask to buy back), or at the
  official close by a closing-auction order, which Alpaca takes for whole
  shares only;
- Alpaca's regulatory fees: about 0.3 bps on every sale.

Mid to mid shows the same trades without the spread.

Usage: gap_costs.py PRICES QUOTES_DIR [--scores FILE --signal NAME]
"""

import argparse
import math
from pathlib import Path

import numpy as np
import polars as pl

from qfin_alphaguard.data.universe import dollar_volume

parser = argparse.ArgumentParser()
parser.add_argument("prices", type=Path)
parser.add_argument("quotes", type=Path)
parser.add_argument("--scores", type=Path)
parser.add_argument("--signal", default="gbt")
args = parser.parse_args()
K = 10
FEE = 0.3e-4

daily = (
    pl.read_parquet(args.prices)
    .sort("ticker", "date")
    .with_columns(
        (pl.col("open") / pl.col("close").shift(1).over("ticker") - 1).alias("gap"),
        (pl.col("close") / pl.col("adjustment")).alias("close_traded"),
        dollar_volume().shift(1).rolling_mean(20).over("ticker").alias("dollar_volume"),
    )
    .filter(pl.col("sp500") & (pl.col("volume") > 0) & pl.col("gap").is_not_null())
    .with_columns(
        (pl.col("dollar_volume").rank(descending=True).over("date") <= 100).alias(
            "top100"
        )
    )
    .select(pl.col("date").alias("day"), "ticker", "gap", "close_traded", "top100")
)
quotes = pl.read_parquet(args.quotes / "20*.parquet")
if args.scores:
    # The model's own universe: the stocks it scored, and its 100 most traded.
    scored = pl.read_parquet(args.scores).select("day", "ticker", args.signal, "top100")
    daily = daily.drop("top100").join(scored, on=["day", "ticker"])


def mid(q: str) -> pl.Expr:
    return (pl.col(f"{q}_bid") + pl.col(f"{q}_ask")) / 2


def summary(r: np.ndarray) -> str:
    sharpe = r.mean() / r.std(ddof=1) * math.sqrt(252)
    t = r.mean() / r.std(ddof=1) * math.sqrt(len(r))
    return f"{r.mean() * 1e4:+6.2f} bps a day (t {t:+.1f}), {r.mean() * 252:+6.1%} a year, Sharpe {sharpe:+.2f}"


for universe, frame in (
    ("all members", daily),
    ("100 most traded", daily.filter("top100")),
):
    if args.scores:  # "fell" marks the stocks bought, here the model's top k
        ranked = frame.with_columns(
            (
                pl.col(args.signal).rank("ordinal", descending=True).over("day") <= K
            ).alias("fell"),
            pl.lit(False).alias("rose"),
        ).filter("fell")
    else:
        ranked = frame.with_columns(
            (pl.col("gap").rank("ordinal").over("day") <= K).alias("fell"),
            (pl.col("gap").rank("ordinal", descending=True).over("day") <= K).alias(
                "rose"
            ),
        ).filter(pl.col("fell") | pl.col("rose"))
    t = (
        ranked.join(quotes, on=["day", "ticker"], how="inner")
        .drop_nulls()
        # The daily bars' adjustment rebuilds a price as traded, but not for
        # a few odd corporate actions (AIV's 2020 spin-off is counted as both
        # a split and a dividend). Those official closes are dropped.
        .with_columns(
            pl.when((pl.col("close_traded") / mid("q1559") - 1).abs() < 0.02)
            .then("close_traded")
            .alias("close_traded")
        )
    )
    picked = (
        f"the {K} highest {args.signal} scores (long)"
        if args.scores
        else f"the {K} largest falls (long) and rises (short)"
    )
    print(f"-- {universe}: {picked}, {t['day'].n_unique()} days")
    half = ((pl.col("q0935_ask") - pl.col("q0935_bid")) / (2 * mid("q0935"))).filter(
        "fell"
    )
    print(
        f"   half spread of the stocks bought: 09:31 "
        f"{t.filter('fell').select(((pl.col('q0931_ask') - pl.col('q0931_bid')) / (2 * mid('q0931'))).mean())[0, 0] * 1e4:.1f} bps, "
        f"09:35 {t.select(half.mean())[0, 0] * 1e4:.1f} bps, 15:59 "
        f"{t.filter('fell').select(((pl.col('q1559_ask') - pl.col('q1559_bid')) / (2 * mid('q1559'))).mean())[0, 0] * 1e4:.1f} bps"
    )
    # A score made at 09:35 cannot trade at 09:31; the gap is known at the open.
    for entry in ("q0935",) if args.scores else ("q0931", "q0935"):
        variants = {
            "mid to mid": (
                mid("q1559") / mid(entry) - 1,
                1 - mid("q1559") / mid(entry),
            ),
            "market in and out": (
                pl.col("q1559_bid") / pl.col(f"{entry}_ask") - 1 - FEE,
                1 - pl.col("q1559_ask") / pl.col(f"{entry}_bid") - FEE,
            ),
            "market in, closing auction out": (
                pl.col("close_traded") / pl.col(f"{entry}_ask") - 1 - FEE,
                1 - pl.col("close_traded") / pl.col(f"{entry}_bid") - FEE,
            ),
        }
        for label, (long_ret, short_ret) in variants.items():
            days = (
                t.group_by("day")
                .agg(
                    long_ret.filter("fell").mean().alias("long"),
                    short_ret.filter("rose").mean().alias("short"),
                )
                .drop_nulls("long" if args.scores else None)
                .sort("day")
            )
            print(
                f"   {entry[1:3]}:{entry[3:]} {label:31} long only "
                f"{summary(days['long'].to_numpy())}"
            )
            if not args.scores:
                both = (days["long"] + days["short"]).to_numpy() / 2
                print(f"   {'':5} {'':31} long and short, half each {summary(both)}")
    by_year = (
        t.group_by("day")
        .agg(
            (pl.col("q1559_bid") / pl.col("q0935_ask") - 1 - FEE)
            .filter("fell")
            .mean()
            .alias("long")
        )
        .with_columns(pl.col("day").dt.year().alias("year"))
        .group_by("year")
        .agg((pl.col("long").mean() * 1e4).round(1).alias("bps"))
        .sort("year")
    )
    print(
        "   long only, market in at 09:35 and out at 15:59, bps a day by year:",
        dict(by_year.iter_rows()),
    )
