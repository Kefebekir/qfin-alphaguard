"""Intraday pilot: the 5-minute opening range breakout on stocks in play
(Zarattini, Barbon and Aziz, SSRN 4729284), on our 50 stocks.

The paper, 2016-2023 on all US stocks above 5 USD: each day, the 20 stocks
whose first 5 minutes trade the most against the same 5 minutes of the last
14 days (at least 100%), in the direction of the first 5-minute bar; a stop
order at the range's high (long) or low (short); a stop loss 10% of the
14-day ATR away; out at the close; 1% of the account at risk a trade, 4x at
most. It charged 0.0035 USD a share and no spread or slippage.

Here, on each year's trading universe, with the paper's filters (an
opening price above 5 USD, at least 1 million shares a day over the last 14
days, and a 14-day ATR above 0.50 USD): the top k by that relative volume
(at least 100%), with s bps of slippage on every fill, the entry, the stop
or the exit at the close (a small account cannot use the closing auction for
fractional shares). Long and short, and long only (a cash account).

The ATR is measured as a share of the price on the adjusted daily bars, and
volumes are put in today's share count, so a split makes no false move.
"""

import json
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from qfin_alphaguard.data.intraday import intraday_path
from qfin_alphaguard.sessions import sessions

ROOT = Path(sys.argv[1])
DAILY = Path(sys.argv[2])
UNIVERSES = {int(k): v for k, v in json.loads(Path(sys.argv[3]).read_text()).items()}
K = 5
SLIPS = (0.0, 2.0, 5.0, 10.0)

opens = {s.day: s.open for s in sessions(date(2016, 1, 1), date(2026, 12, 31))}
c, prev_close = pl.col("close"), pl.col("close").shift(1)
daily = (
    pl.read_parquet(DAILY)
    .sort("ticker", "date")
    .with_columns(
        (
            pl.max_horizontal(
                pl.col("high") - pl.col("low"),
                (pl.col("high") - prev_close).abs(),
                (pl.col("low") - prev_close).abs(),
            )
            / prev_close
        )
        .over("ticker")
        .alias("tr_pct"),
        (pl.col("volume") / pl.col("split_factor")).alias("shares"),
    )
    # What is known before the day: the last 14 days up to yesterday.
    .with_columns(
        pl.col("tr_pct").rolling_mean(14).shift(1).over("ticker").alias("atr_pct"),
        pl.col("shares").rolling_mean(14).shift(1).over("ticker").alias("shares14"),
    )
    .select(
        "ticker", pl.col("date").alias("day"), "atr_pct", "shares14", "split_factor"
    )
)


def minutes(year: int) -> pl.DataFrame:
    """The year's 1-minute bars of its trading universe, with k, the minutes
    since the open."""
    files = [intraday_path(ROOT, t, year) for t in UNIVERSES[year]]
    return (
        pl.concat(
            [
                pl.read_parquet(f).select(
                    "start", "ticker", "open", "high", "low", "close", "volume"
                )
                for f in files
                if f.exists()
            ]
        )
        .with_columns(pl.col("start").dt.date().alias("day"))
        .sort("ticker", "start")
        .with_columns(
            (
                pl.col("start")
                - pl.col("day").replace_strict(
                    opens, return_dtype=pl.Datetime("us", "UTC")
                )
            )
            .dt.total_minutes()
            .alias("k")
        )
        .filter(pl.col("k") >= 0)
    )


# The opening ranges of every year first, so that a stock's relative volume
# can look back across the new year; a stock back in the universe after a
# gap has none until it has 14 recent sessions.
ranges = pl.concat(
    [
        minutes(year)
        .filter(pl.col("k") < 5)
        .group_by("ticker", "day")
        .agg(
            pl.col("open").first().alias("or_open"),
            pl.col("high").max().alias("or_high"),
            pl.col("low").min().alias("or_low"),
            pl.col("close").last().alias("or_close"),
            pl.col("volume").sum().alias("or_volume"),
        )
        .with_columns(pl.lit(year).alias("year"))
        for year in sorted(UNIVERSES)
    ]
)
candidates = (
    ranges.join(daily, on=["ticker", "day"])
    .sort("ticker", "day")
    .with_columns((pl.col("or_volume") * pl.col("split_factor")).alias("or_shares_now"))
    .with_columns(
        pl.when(
            (pl.col("day") - pl.col("day").shift(14).over("ticker")).dt.total_days()
            <= 30
        )
        .then(
            pl.col("or_shares_now")
            / pl.col("or_shares_now").shift(1).rolling_mean(14).over("ticker")
        )
        .alias("rel_volume")
    )
    .drop_nulls()
    .filter(
        (pl.col("rel_volume") >= 1.0)
        & (pl.col("or_close") != pl.col("or_open"))
        & (pl.col("or_open") > 5)
        & (pl.col("shares14") >= 1e6)
        & (pl.col("atr_pct") * pl.col("or_open") > 0.5)
    )
)
trades = []
for year in sorted(UNIVERSES):
    m = minutes(year)
    chosen = (
        candidates.filter(pl.col("year") == year)
        .sort("rel_volume", descending=True)
        .group_by("day", maintain_order=True)
        .head(K)
    )
    after = m.filter(pl.col("k") >= 5).join(
        chosen.select("ticker", "day"), on=["ticker", "day"]
    )
    groups = {key: g for key, g in after.group_by("ticker", "day")}
    for row in chosen.iter_rows(named=True):
        g = groups.get((row["ticker"], row["day"]))
        if g is None:
            continue
        side = 1 if row["or_close"] > row["or_open"] else -1
        trigger = row["or_high"] if side == 1 else row["or_low"]
        o, h, lo, c = (g[x].to_numpy() for x in ("open", "high", "low", "close"))
        hit = np.nonzero(h >= trigger)[0] if side == 1 else np.nonzero(lo <= trigger)[0]
        if len(hit) == 0:
            continue
        i = hit[0]
        entry = max(trigger, o[i]) if side == 1 else min(trigger, o[i])
        risk = 0.1 * row["atr_pct"]  # the loss at the stop, as a share of the entry
        stop = entry * (1 - side * risk)
        # A 1-minute bar does not say whether its low came before or after the
        # entry. Two readings bracket the truth: "pessimistic" counts the entry
        # bar's whole range against the stop, "optimistic" only its close.
        for reading in ("pessimistic", "optimistic"):
            exit_price, stopped = c[-1], False
            for j in range(i, len(c)):
                if j == i:
                    worst = lo[j] if side == 1 else h[j]
                    if reading == "optimistic":
                        worst = c[j]
                    if side * (worst - stop) <= 0:
                        exit_price, stopped = stop, True
                        break
                elif side * (o[j] - stop) <= 0:  # opened through the stop
                    exit_price, stopped = o[j], True
                    break
                elif (side == 1 and lo[j] <= stop) or (side == -1 and h[j] >= stop):
                    exit_price, stopped = stop, True
                    break
            trades.append(
                {
                    "reading": reading,
                    "day": row["day"],
                    "ticker": row["ticker"],
                    "side": side,
                    "gross": side * (exit_price / entry - 1),
                    "risk": risk,
                    "stopped": stopped,
                }
            )

t = pl.DataFrame(trades).with_columns(
    pl.min_horizontal(pl.lit(0.01) / pl.col("risk"), pl.lit(4.0 / K)).alias("weight")
)
last_day = max(t["day"])
all_days = pl.DataFrame(
    {"day": sorted(d for d in opens if date(2016, 2, 1) <= d <= last_day)}
)
years = np.array([d.year for d in all_days["day"]])
for reading in ("pessimistic", "optimistic"):
    tr = t.filter(pl.col("reading") == reading)
    print(
        f"{reading}: trades {tr.height}, stopped {tr['stopped'].mean():.0%}, "
        f"long {(tr['side'] == 1).mean():.0%}, average risk {tr['risk'].mean():.2%}, "
        f"average weight {tr['weight'].mean():.2f}"
    )
    for label, subset in (
        ("long and short", tr),
        ("long only", tr.filter(pl.col("side") == 1)),
    ):
        for slip in SLIPS:
            day = (
                subset.with_columns(
                    # two fills: the entry, and the stop or the exit at the close
                    (pl.col("weight") * (pl.col("gross") - 2 * slip / 1e4)).alias("pnl")
                )
                .group_by("day")
                .agg(pl.col("pnl").sum())
            )
            r = (
                all_days.join(day, on="day", how="left")
                .fill_null(0.0)["pnl"]
                .to_numpy()
            )
            growth = np.prod(1 + r) ** (252 / len(r)) - 1
            sharpe = r.mean() / r.std(ddof=1) * math.sqrt(252)
            late = r[years >= 2024]
            late_sharpe = late.mean() / late.std(ddof=1) * math.sqrt(252)
            print(
                f"  {label:15} slippage {slip:4.1f} bps: {growth:+.1%} a year, "
                f"Sharpe {sharpe:+.2f}; 2024-2026 Sharpe {late_sharpe:+.2f}"
            )
