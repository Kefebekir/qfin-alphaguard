"""Batch test B6 (preregistrations/07_batch.txt): Bitcoin's weekly time-series momentum."""

import math

import polars as pl

d = pl.read_parquet("data/raw/crypto/btc_usd_daily.parquet").sort("time")
w = (
    d.group_by_dynamic("time", every="1w")
    .agg(pl.col("c").last())
    .sort("time")
    .with_columns((pl.col("c") / pl.col("c").shift(1) - 1).alias("r"))
    .drop_nulls("r")
)
r = w["r"].to_numpy()
x, y = r[:-1], r[1:]
xc = x - x.mean()
b = xc @ (y - y.mean()) / (xc @ xc)
e = y - y.mean() - b * xc
t = b / math.sqrt(e @ e / (len(y) - 2) / (xc @ xc))
print(
    f"B6: BTC next week's return on last week's: slope {b:+.3f}, t {t:+.2f} ({len(y)} weeks, "
    f"{w['time'].min()} .. {w['time'].max()}) -> {'PASS' if t >= 2.64 else 'fail'}"
)
