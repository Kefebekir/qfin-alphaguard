"""Dealer gamma exposure (GEX) of QQQ's and SPY's options, day by day, 2016-2022, and the
pre-registered test G1/G2 (preregistration.txt). PHASE=build writes gex.parquet and a
sanity summary; PHASE=test runs the test once.

GEX(day) = sum over calls of gamma x OI x 100 x S^2 x 0.01 - the same over puts
(dealers assumed long the calls and short the puts customers hold), Black-Scholes gamma
with one volatility for every contract (VXN for QQQ, VIX for SPY), the T-bill rate, no
dividend; divided by the underlying's 20-session average dollar volume; ranked against
its own previous 250 sessions.
"""

import math
import os
from datetime import date
from itertools import pairwise
from pathlib import Path

import numpy as np
import polars as pl

from qfin_alphaguard.data.intraday import intraday_path
from qfin_alphaguard.sessions import sessions

PHASE = os.environ.get("PHASE", "none")
SESS = sorted(s.day for s in sessions(date(2016, 1, 1), date(2022, 12, 31)))
PREV = {b: a for a, b in pairwise(SESS)}


def cboe(name):
    return pl.read_csv(f"data/raw/cboe/{name}_History.csv").select(
        pl.col("DATE").str.to_date("%m/%d/%Y").alias("day"),
        (pl.col("CLOSE") / 100).alias("sigma"),
    )


rate = pl.read_csv("data/raw/fred/DTB3.csv", null_values=["."]).select(
    pl.col("observation_date").str.to_date().alias("day"),
    (pl.col("DTB3") / 100).alias("r"),
)


def underlying(ticker, years=range(2016, 2023)):
    """Each session's close and dollar volume from the stored minutes."""
    m = pl.concat(
        [
            pl.read_parquet(intraday_path(Path("data/raw"), ticker, y)).select(
                "start", "close", "volume"
            )
            for y in years
        ]
    ).with_columns(pl.col("start").dt.date().alias("day"))
    return (
        m.sort("start")
        .group_by("day", maintain_order=True)
        .agg(
            pl.col("close").last().alias("S"),
            (pl.col("close") * pl.col("volume")).sum().alias("dollars"),
        )
        .sort("day")
    )


def gex_series(ticker, vol_index, years=range(2016, 2023), oi_files=None):
    if oi_files is None:
        oi_files = [f"data/raw/opra/{ticker}_oi_{y}.parquet" for y in years]
    oi = pl.concat([pl.read_parquet(f) for f in oi_files])
    oi = oi.with_columns(
        pl.col("ts_event")
        .dt.convert_time_zone("America/New_York")
        .dt.date()
        .alias("day")
    )
    sym = pl.col("symbol").str.strip_chars().str.slice(-15)  # YYMMDD C/P strike x 1000
    oi = oi.with_columns(
        sym.str.slice(0, 6).str.to_date("%y%m%d").alias("expiry"),
        sym.str.slice(6, 1).alias("cp"),
        (sym.str.slice(7, 8).cast(pl.Int64) / 1000).alias("K"),
    ).filter(pl.col("quantity") > 0)
    u = underlying(ticker, years)
    u = u.with_columns(
        pl.col("dollars").shift(1).rolling_mean(20).alias("adv")
    )  # 20 sessions before
    prev = {b: a for a, b in pairwise(u["day"].to_list())}
    s_prev = dict(zip(u["day"], u["S"], strict=True))
    adv = dict(zip(u["day"], u["adv"], strict=True))
    sig = dict(zip(*cboe(vol_index).to_dict(as_series=False).values(), strict=True))
    rr = dict(zip(*rate.drop_nulls().to_dict(as_series=False).values(), strict=True))
    rows = []
    for (day,), g in oi.group_by("day"):
        before = prev.get(day)
        if before is None or before not in s_prev or adv.get(day) is None:
            continue
        S, sigma = s_prev[before], sig.get(before)
        r = rr.get(before)
        if sigma is None or r is None:
            continue
        T = (g["expiry"] - day).dt.total_days().to_numpy().astype(float)
        keep = (T >= 0) & (T <= 365)
        T = np.maximum(T[keep], 0.5) / 365
        K = g["K"].to_numpy()[keep]
        n = g["quantity"].to_numpy()[keep].astype(float)
        call = g["cp"].to_numpy()[keep] == "C"
        d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        gamma = np.exp(-(d1**2) / 2) / math.sqrt(2 * math.pi) / (S * sigma * np.sqrt(T))
        dollar_gamma = gamma * n * 100 * S**2 * 0.01
        gex = dollar_gamma[call].sum() - dollar_gamma[~call].sum()
        rows.append(
            {
                "day": day,
                "gex": gex,
                "gex_adv": gex / adv[day],
                "contracts": int(keep.sum()),
            }
        )
    out = pl.DataFrame(rows).sort("day")
    x = out["gex_adv"].to_numpy()
    rank = np.full(len(x), np.nan)
    for i in range(250, len(x)):
        rank[i] = (x[i - 250 : i] < x[i]).mean()
    return out.with_columns(pl.Series("rank", rank))


if PHASE == "build":
    wanted = os.environ.get("TICKERS", "QQQ,SPY").split(",")
    for ticker, vol in (("QQQ", "VXN"), ("SPY", "VIX")):
        if ticker not in wanted:
            continue
        g = gex_series(ticker, vol)
        g.write_parquet(f"data/step5/gamma_real/gex_{ticker}.parquet")
        q = g["gex_adv"].quantile
        print(
            f"{ticker}: {g.height} days {g['day'].min()}..{g['day'].max()}, contracts median "
            f"{g['contracts'].median():.0f}; GEX / ADV quantiles 5/50/95%: {q(0.05):+.3f} {q(0.5):+.3f} "
            f"{q(0.95):+.3f}; share positive {np.mean(g['gex_adv'].to_numpy() > 0):.0%}"
        )
elif PHASE == "test":
    COST = 0.3e-4 + 0.0000206 / 2
    d = (
        pl.read_parquet("data/intraday_pilot/policy/decisions.parquet")
        .filter(pl.col("ticker") == "QQQ")
        .sort("day", "t")
    )
    d = d.with_columns(pl.col("rule").shift(1).over("day").fill_null(0).alias("before"))
    last = pl.col("t") == pl.col("t").max().over("day")
    d = d.with_columns(
        (
            pl.col("size") * pl.col("rule") * pl.col("ret")
            - pl.col("size")
            * COST
            * (
                (pl.col("rule") - pl.col("before")).abs()
                + pl.when(last).then(pl.col("rule").abs()).otherwise(0)
            )
        ).alias("net")
    )
    rule = d.group_by("day").agg(pl.col("net").sum().alias("pnl")).sort("day")
    for name, ticker, hurdle in (("G1", "QQQ", -2.24), ("G2", "SPY", -2.24)):
        g = pl.read_parquet(f"data/step5/gamma_real/gex_{ticker}.parquet")
        j = (
            rule.join(g, on="day")
            .sort("day")
            .filter(
                pl.col("rank").is_not_nan()
                & pl.col("day").dt.year().is_between(2017, 2022)
            )
        )
        y, x = j["pnl"].to_numpy(), j["rank"].to_numpy()
        xc = x - x.mean()
        b = xc @ (y - y.mean()) / (xc @ xc)
        e = y - y.mean() - b * xc
        t = b / math.sqrt((e @ e) / (len(y) - 2) / (xc @ xc))
        print(
            f"{name} ({ticker} GEX), {len(y)} days 2017-2022: slope {b * 1e4:+.2f} bps per unit rank, t {t:+.2f} -> {'PASS' if t <= hurdle else 'fail'} (needs <= {hurdle})"
        )
        for lo, hi in ((0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.01)):
            p = y[(x >= lo) & (x < hi)]
            print(
                f"   rank {lo:.2f}-{min(hi, 1):.2f}: {len(p)} days, rule mean {p.mean() * 1e4:+.2f} bps, Sharpe {p.mean() / p.std(ddof=1) * math.sqrt(252):.2f}"
            )
        for yr in range(2017, 2023):
            on = j["day"].dt.year().to_numpy() == yr
            xx, yy = x[on], y[on]
            xc2 = xx - xx.mean()
            bb = xc2 @ (yy - yy.mean()) / (xc2 @ xc2)
            print(f"   {yr}: slope {bb * 1e4:+.2f} bps per unit rank")
