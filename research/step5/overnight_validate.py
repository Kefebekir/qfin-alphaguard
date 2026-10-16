"""The pre-registered validation (preregistration_validation.txt): V1, V2 on 2014-2015
minutes from EODHD, V3 on daily bars before 2016."""

import math
from datetime import date, timedelta
from itertools import pairwise

import polars as pl

from qfin_alphaguard.data.intraday import regular_session
from qfin_alphaguard.sessions import sessions

SESS = {s.day: s for s in sessions(date(2013, 12, 1), date(2016, 1, 31))}
PREV = {b.day: a.day for a, b in pairwise(sorted(SESS.values(), key=lambda s: s.day))}


def ols(y, x):
    x = x - x.mean()
    b = (x @ (y - y.mean())) / (x @ x)
    e = y - y.mean() - b * x
    return b, b / math.sqrt((e @ e) / (len(y) - 2) / (x @ x)), len(y)


for t in ("SPY", "QQQ"):
    m = (
        regular_session(
            pl.read_parquet(f"data/raw/minutes_pre2016/{t}_2014_2015.parquet")
        )
        .with_columns(pl.col("start").dt.date().alias("day"))
        .sort("start")
    )
    rows = []
    for (day,), g in m.group_by("day", maintain_order=True):
        s = SESS[day]

        def close_at(minute, g=g, s=s):
            # the last close at or before the bar starting `minute` after the open
            part = g.filter(pl.col("start") <= s.open + timedelta(minutes=minute))
            return part["close"][-1] if part.height else None

        n = int((s.close - s.open).total_seconds() // 60)
        rows.append(
            {
                "day": day,
                "open": g["open"][0],
                "close": g["close"][-1],
                "c1529": close_at(n - 31),
                "c1549": close_at(n - 11),
                "last_start": g["start"][-1],
                "close_time": s.close,
            }
        )
    d = pl.DataFrame(rows).sort("day")
    nxt = d.select(pl.col("day").alias("next_day"), pl.col("open").alias("next_open"))
    d = d.with_columns(
        pl.col("day").shift(-1).alias("next_day"),
        pl.col("open").shift(-1).alias("next_open"),
    )
    d = d.filter(
        pl.Series(
            [
                PREV.get(b) == a if b is not None else False
                for a, b in zip(d["day"], d["next_day"])
            ]
        )
    )
    d = d.filter(
        pl.col("last_start") >= pl.col("close_time") - timedelta(minutes=2)
    )  # the session's last minutes are there
    d = d.drop_nulls(["c1529", "c1549"]).with_columns(
        (pl.col("next_open") / pl.col("close") - 1).alias("overnight"),
        (pl.col("close") / pl.col("c1529") - 1).alias("last30"),
        (pl.col("c1549") / pl.col("c1529") - 1).alias("to1550"),
    )
    b1, t1, n1 = ols(d["overnight"].to_numpy(), d["last30"].to_numpy())
    b2, t2, n2 = ols(d["overnight"].to_numpy(), d["to1550"].to_numpy())
    print(
        f"{t} 2014-2015 minutes, {n1} nights: V1 slope {b1:+.3f}, t {t1:+.2f}; V2 (15:30-15:50 signal) slope {b2:+.3f}, t {t2:+.2f}"
    )
for t, first in (("SPY", 1993), ("QQQ", 1999)):
    x = (
        pl.read_parquet(f"data/raw/etf_daily/{t}.parquet")
        .filter(pl.col("date").dt.year() <= 2015)
        .sort("date")
    )
    x = x.with_columns(
        (pl.col("open").shift(-1) / pl.col("close") - 1).alias("overnight_next"),
        (pl.col("close") / pl.col("open") - 1).alias("intraday"),
    ).drop_nulls("overnight_next")
    b3, t3, n3 = ols(x["overnight_next"].to_numpy(), x["intraday"].to_numpy())
    print(f"{t} daily {first}-2015, {n3} nights: V3 slope {b3:+.4f}, t {t3:+.2f}")
    for lo, hi in ((first, 2003), (2004, 2009), (2010, 2015)):
        p = x.filter(pl.col("date").dt.year().is_between(lo, hi))
        bb, tt, nn = ols(p["overnight_next"].to_numpy(), p["intraday"].to_numpy())
        print(f"     {lo}-{hi}: slope {bb:+.4f}, t {tt:+.2f}")
