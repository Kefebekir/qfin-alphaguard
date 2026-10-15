import math
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from qfin_alphaguard.data.intraday import intraday_path
from qfin_alphaguard.sessions import sessions

LAST = 2022
opens = {s.day: s.open for s in sessions(date(2016, 1, 1), date(LAST, 12, 31))}
bars = (
    pl.concat(
        [
            pl.read_parquet(intraday_path(Path("data/raw"), "QQQ", y)).select(
                "start", "close"
            )
            for y in range(2016, LAST + 1)
        ]
    )
    .with_columns(pl.col("start").dt.date().alias("day"))
    .sort("start")
)
rows = []
for (day,), g in bars.group_by("day", maintain_order=True):
    if day not in opens:
        continue
    k = ((g["start"] - opens[day]).dt.total_minutes()).to_numpy().astype(int)
    keep = k >= 0
    k = k[keep]
    c = g["close"].to_numpy()[keep]
    n = k.max() + 1
    p = np.full(n, np.nan)
    p[k] = c
    p = (
        pl.Series(p)
        .fill_nan(None)
        .fill_null(strategy="forward")
        .fill_null(strategy="backward")
        .to_numpy()
    )
    lp = np.log(p)
    r1 = np.diff(lp)
    full = (n // 15) * 15
    r15 = np.diff(lp[::15][: full // 15 + 1])
    vr = np.var(r15, ddof=1) / (15 * np.var(r1, ddof=1)) if len(r15) > 3 else np.nan
    m = lp[:30]
    vr_morning = np.var(np.diff(m[::5]), ddof=1) / (5 * np.var(np.diff(m), ddof=1))
    rv = math.sqrt(np.sum(r1**2) * 252)
    rows.append({"day": day, "vr": vr, "vr_morning": vr_morning, "rv": rv})
f = pl.DataFrame(rows).sort("day")
vix = pl.read_csv("data/raw/cboe/VIX_History.csv").select(
    pl.col("DATE").str.to_date("%m/%d/%Y").alias("day"),
    (pl.col("CLOSE") / 100).alias("vix"),
)
f = f.join(vix, on="day", how="left").with_columns(
    pl.col("vr").shift(1).alias("vr_yesterday"),
    pl.col("vr").shift(1).rolling_mean(5).alias("vr_5d"),
    pl.col("vr").shift(1).rolling_mean(20).alias("vr_20d"),
    (pl.col("rv") / pl.col("vix")).shift(1).alias("rv_over_iv_yesterday"),
    (pl.col("rv") / pl.col("vix")).shift(1).rolling_mean(5).alias("rv_over_iv_5d"),
)
COST = 0.3e-4 + 0.0000206 / 2
d = (
    pl.read_parquet("data/intraday_pilot/policy/decisions.parquet")
    .filter((pl.col("ticker") == "QQQ") & (pl.col("day").dt.year() <= LAST))
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
days = (
    d.group_by("day")
    .agg(pl.col("net").sum().alias("pnl"))
    .join(f, on="day")
    .sort("day")
)


def stats(v):
    v = np.asarray(v)
    return f"n {len(v):5}, mean {v.mean() * 1e4:+7.2f} bps, t {v.mean() / v.std(ddof=1) * math.sqrt(len(v)):+5.2f}, Sharpe {v.mean() / v.std(ddof=1) * math.sqrt(252):5.2f}"


print(
    "mean VR (1 = random walk):",
    round(f["vr"].mean(), 3),
    "| mean morning VR:",
    round(f["vr_morning"].mean(), 3),
)
for col in (
    "vr_yesterday",
    "vr_5d",
    "vr_20d",
    "vr_morning",
    "rv_over_iv_yesterday",
    "rv_over_iv_5d",
):
    g = days.drop_nulls(col).filter(pl.col(col).is_finite())
    e = np.quantile(g[col].to_numpy(), [1 / 3, 2 / 3])
    print(f"{col}: edges {e.round(3)}")
    for i, (lo, hi) in enumerate(((-np.inf, e[0]), (e[0], e[1]), (e[1], np.inf))):
        part = g.filter((pl.col(col) > lo) & (pl.col(col) <= hi))
        print(f"    tercile {i + 1}: {stats(part['pnl'].to_numpy())}")
days.write_parquet("data/step5/gamma_proxies_2016_2022.parquet")
