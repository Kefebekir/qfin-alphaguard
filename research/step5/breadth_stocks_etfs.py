"""Exploration: does the noise-area rule work beyond QQQ? (Breadth, in the fundamental law's sense.)

1. On every stored stock (policy_data.py with TICKERS set to the stocks, into
   data/intraday_pilot/policy/stocks), at a 1 bp half spread: each stock's Sharpe ratio, and an
   equal-weight portfolio of all of them, before and after costs, 2016-2022 and 2023-2026.
2. On the 15 other stored ETFs: Sharpe before costs, after 0.3 and 1 bp, and the correlation
   with the QQQ rule.
"""

import math

import numpy as np
import polars as pl

FEE = 0.0000206 / 2


def day_pnl(d, half_spread):
    d = d.sort("ticker", "day", "t").with_columns(
        pl.col("rule").shift(1).over("ticker", "day").fill_null(0).alias("before")
    )
    last = pl.col("t") == pl.col("t").max().over("ticker", "day")
    turn = (pl.col("rule") - pl.col("before")).abs() + pl.when(last).then(
        pl.col("rule").abs()
    ).otherwise(0)
    d = d.with_columns(
        (
            pl.col("size") * pl.col("rule") * pl.col("ret")
            - pl.col("size") * (half_spread + FEE) * turn
        ).alias("net"),
        (pl.col("size") * pl.col("rule") * pl.col("ret")).alias("gross"),
        (pl.col("size") * turn).alias("turn"),
    )
    return (
        d.group_by("ticker", "day")
        .agg(pl.col("net").sum(), pl.col("gross").sum(), pl.col("turn").sum())
        .sort("ticker", "day")
    )


def sharpe(x):
    x = np.asarray(x)
    return (
        x.mean() / x.std(ddof=1) * math.sqrt(252)
        if len(x) > 20 and x.std() > 0
        else np.nan
    )


etfs = pl.read_parquet("data/intraday_pilot/policy/decisions.parquet")
qqq = day_pnl(etfs.filter(pl.col("ticker") == "QQQ"), 0.3e-4).select(
    "day", pl.col("net").alias("qqq")
)
stocks = day_pnl(
    pl.read_parquet("data/intraday_pilot/policy/stocks/decisions.parquet"), 1.0e-4
)
for label, lo, hi in (("2016-2022", 2016, 2022), ("2023-2026", 2023, 2026)):
    part = stocks.filter(pl.col("day").dt.year().is_between(lo, hi))
    per = (
        part.group_by("ticker")
        .agg(pl.col("net"), pl.col("gross"), pl.len().alias("n"))
        .filter(pl.col("n") >= 250)
    )
    net = np.array([sharpe(x) for x in per["net"]])
    gross = np.array([sharpe(x) for x in per["gross"]])
    print(f"{label}: {per.height} stocks with a year or more")
    print(
        f"  per-stock Sharpe after 1 bp: median {np.nanmedian(net):+.2f}, above 0: {np.mean(net > 0):.0%}; before costs median {np.nanmedian(gross):+.2f}"
    )
    port = (
        part.group_by("day")
        .agg(pl.col("net").mean().alias("port"), pl.col("gross").mean().alias("gross"))
        .join(qqq, on="day")
        .sort("day")
    )
    print(
        f"  equal-weight portfolio: Sharpe {sharpe(port['port']):.2f} after costs, {sharpe(port['gross']):.2f} before; corr with the QQQ rule {np.corrcoef(port['port'], port['qqq'])[0, 1]:+.2f}"
    )
all_etfs = day_pnl(etfs, 0.0).join(qqq, on="day")
print("ETF: Sharpe before costs | after 0.3 bp | after 1 bp | corr with the QQQ rule")
for t in [
    "SPY",
    "IWM",
    "DIA",
    "TLT",
    "GLD",
    "HYG",
    "XLK",
    "XLF",
    "XLE",
    "XLV",
    "XLY",
    "XLI",
    "XLP",
    "XLU",
    "XLB",
]:
    x = all_etfs.filter(pl.col("ticker") == t).sort("day")
    g, tv = x["gross"].to_numpy(), x["turn"].to_numpy()
    n03, n1 = g - tv * (0.3e-4 + FEE), g - tv * (1e-4 + FEE)
    print(
        f"  {t:5} {sharpe(g):6.2f} | {sharpe(n03):6.2f} | {sharpe(n1):6.2f} | {np.corrcoef(n03, x['qqq'])[0, 1]:+.2f}"
    )
