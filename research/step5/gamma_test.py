"""The gamma-state test: H1 and H2, written down before 2023-2026 is computed.

PHASE=calibrate: the scale k for each variant from 2017-2022 only.
PHASE=holdout: the test on 2023-01-01 .. 2026-10-09 (run once).
"""

import math
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from qfin_alphaguard.data.intraday import intraday_path
from qfin_alphaguard.sessions import sessions

PHASE = os.environ["PHASE"]
COST = 0.3e-4 + 0.0000206 / 2
opens = {s.day: s.open for s in sessions(date(2016, 1, 1), date(2026, 12, 31))}
bars = (
    pl.concat(
        [
            pl.read_parquet(intraday_path(Path("data/raw"), "QQQ", y)).select(
                "start", "close"
            )
            for y in range(2016, 2027)
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
    rows.append(
        {
            "day": day,
            "vr": np.var(r15, ddof=1) / (15 * np.var(r1, ddof=1)),
            "rv": math.sqrt(np.sum(r1**2) * 252),
        }
    )
f = pl.DataFrame(rows).sort("day")
vix = pl.read_csv("data/raw/cboe/VIX_History.csv").select(
    pl.col("DATE").str.to_date("%m/%d/%Y").alias("day"),
    (pl.col("CLOSE") / 100).alias("vix"),
)
f = (
    f.join(vix, on="day", how="left")
    .sort("day")
    .with_columns(
        pl.col("vr").shift(1).alias("vr_y"),
        (pl.col("rv") / pl.col("vix")).shift(1).alias("rviv_y"),
    )
)
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
days = (
    d.group_by("day")
    .agg(pl.col("net").sum().alias("pnl"), pl.col("size").first())
    .join(f, on="day")
    .sort("day")
)
assert days["day"].is_sorted()


def trailing_rank(x, n=250):
    r = np.full(len(x), np.nan)
    for i in range(n, len(x)):
        past = x[i - n : i]
        past = past[~np.isnan(past)]
        if len(past) > 50 and not np.isnan(x[i]):
            r[i] = (past < x[i]).mean()
    return r


score = (
    trailing_rank(days["vr_y"].to_numpy()) + trailing_rank(days["rviv_y"].to_numpy())
) / 2
m1 = np.minimum(
    np.where(score < 1 / 3, 0.0, np.where(score < 2 / 3, 1.0, 1.5)),
    2.0 / days["size"].to_numpy(),
)
ev = pl.read_csv("data/raw/calendar/events.csv").with_columns(
    pl.col("day").str.to_date()
)
fomc = set(ev.filter(pl.col("event") == "fomc")["day"])


def third_friday(y, mo):
    first = date(y, mo, 1)
    fr = first + timedelta(days=(4 - first.weekday()) % 7)
    return fr + timedelta(days=14)


session_days = set(days["day"])
opex = set()
for y in range(2016, 2027):
    for mo in range(1, 13):
        if mo in (3, 6, 9, 12):
            continue  # monthly expiries only
        fr = third_friday(y, mo)
        while fr not in session_days and fr.month == mo and fr > date(y, mo, 14):
            fr -= timedelta(days=1)
        opex.add(fr)
skip = np.array([dd in fomc or dd in opex for dd in days["day"]])
m2 = np.where(skip, 0.0, m1)
p = days["pnl"].to_numpy()
year = days["day"].dt.year().to_numpy()
ok = ~np.isnan(score)


def sharpe(v):
    return v.mean() / v.std(ddof=1) * math.sqrt(252)


if PHASE == "calibrate":
    sel = ok & (year >= 2017) & (year <= 2022)
    for name, m in (("H1", m1), ("H2", m2)):
        k = p[sel].std() / (p[sel] * m[sel]).std()
        print(
            f"{name}: k = {k:.4f} (rule vol / sized vol on 2017-2022, {sel.sum()} days); Sharpe rule {sharpe(p[sel]):.2f}, sized {sharpe(p[sel] * m[sel]):.2f}"
        )
    sys.exit()
K = {"H1": float(os.environ["K1"]), "H2": float(os.environ["K2"])}
sel = (
    ok
    & (days["day"].to_numpy() >= np.datetime64("2023-01-01"))
    & (days["day"].to_numpy() <= np.datetime64("2026-10-09"))
)
print(
    f"holdout: {sel.sum()} days, {days['day'].to_numpy()[sel][0]} .. {days['day'].to_numpy()[sel][-1]}"
)
print(
    f"rule: Sharpe {sharpe(p[sel]):.2f}, {np.prod(1 + p[sel]) ** (252 / sel.sum()) - 1:+.1%} a year"
)
for name, m in (("H1", m1), ("H2", m2)):
    r = p[sel] * m[sel]
    diff = K[name] * r - p[sel]
    t = diff.mean() / diff.std(ddof=1) * math.sqrt(len(diff))
    raw = r - p[sel]
    traw = raw.mean() / raw.std(ddof=1) * math.sqrt(len(raw))
    print(
        f"{name}: Sharpe {sharpe(r):.2f}, {np.prod(1 + r) ** (252 / len(r)) - 1:+.1%} a year, mean size {m[sel].mean():.2f}; "
        f"equal vol (k {K[name]:.4f}) minus rule {diff.mean() * 252:+.2%} a year, t {t:+.2f} -> {'PASS' if t >= 2.24 else 'fail'} (needs 2.24); raw {raw.mean() * 252:+.2%}, t {traw:+.2f}"
    )
    for y in range(2023, 2027):
        on = sel & (year == y)
        print(
            f"   {y}: rule Sharpe {sharpe(p[on]):5.2f}, {name} {sharpe(p[on] * m[on]):5.2f}"
        )
for lo, hi in ((0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.01)):
    on = sel & (score >= lo) & (score < hi)
    print(
        f"   score {lo:.2f}-{min(hi, 1):.2f}: {on.sum()} days, rule Sharpe {sharpe(p[on]):.2f}"
    )
