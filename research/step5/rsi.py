import math

import numpy as np
import polars as pl


def rsi(close, n=5):
    d = np.diff(close, prepend=np.nan)
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    au, ad = np.full(len(close), np.nan), np.full(len(close), np.nan)
    au[n], ad[n] = up[1 : n + 1].mean(), dn[1 : n + 1].mean()
    for i in range(n + 1, len(close)):
        au[i] = (au[i - 1] * (n - 1) + up[i]) / n
        ad[i] = (ad[i - 1] * (n - 1) + dn[i]) / n
    return 100 - 100 / (1 + au / np.where(ad == 0, np.nan, ad))


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
rule = d.group_by("day").agg(pl.col("net").sum().alias("pnl"))
for name, t in (("R1 SPY RSI(5)", "SPY"), ("R2 QQQ RSI(5)", "QQQ")):
    px = pl.read_parquet(f"data/raw/etf_daily/{t}.parquet").sort("date")
    px = px.with_columns(pl.Series("rsi", rsi(px["close"].to_numpy()))).with_columns(
        pl.col("rsi").shift(1).alias("rsi_prev")
    )
    j = (
        rule.join(px.select(pl.col("date").alias("day"), "rsi_prev"), on="day")
        .drop_nulls()
        .filter(pl.col("rsi_prev").is_not_nan())
        .sort("day")
    )
    for label, part in (
        ("2016-2026", j),
        ("after the paper, 2024-06 on", j.filter(pl.col("day") >= pl.date(2024, 6, 1))),
    ):
        x, y = part["rsi_prev"].to_numpy(), part["pnl"].to_numpy() * 1e4
        xc = x - x.mean()
        b = xc @ (y - y.mean()) / (xc @ xc)
        e = y - y.mean() - b * xc
        tt = b / math.sqrt(e @ e / (len(y) - 2) / (xc @ xc))
        print(
            f"{name} {label}: slope {b:+.3f} bps per RSI point, t {tt:+.2f} (n {len(y)})"
            + (
                f" -> {'PASS' if tt <= -2.0 else 'fail'}"
                if label == "2016-2026"
                else ""
            )
        )
