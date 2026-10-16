import contextlib
import io
import math
import runpy
from datetime import timedelta

import polars as pl

with contextlib.redirect_stdout(io.StringIO()):
    ns = runpy.run_path("research/step5/calendar_effects.py")
daily, SESS = ns["daily"], ns["SESS"]
from pathlib import Path

from qfin_alphaguard.data.intraday import intraday_path

ev = pl.read_csv("data/raw/calendar/events.csv").with_columns(
    pl.col("day").str.to_date()
)
macro = set(ev.filter(pl.col("event").is_in(["cpi", "nfp"]))["day"])


def last_half_hour(ticker):
    rows = []
    for y in range(2016, 2027):
        b = (
            pl.read_parquet(intraday_path(Path("data/raw"), ticker, y))
            .select("start", "close")
            .with_columns(pl.col("start").dt.date().alias("day"))
        )
        for (day,), g in b.group_by("day"):
            s = SESS.get(day)
            if s is None:
                continue
            cut = s.close - timedelta(minutes=31)
            before = g.filter(pl.col("start") <= cut)
            inside = g.filter((pl.col("start") > cut) & (pl.col("start") < s.close))
            if before.height and inside.height:
                rows.append(
                    {
                        "day": day,
                        "last30": inside["close"][-1]
                        / before.sort("start")["close"][-1]
                        - 1,
                    }
                )
    return pl.DataFrame(rows)


def ols_t(y, x):
    x = x - x.mean()
    b = (x @ (y - y.mean())) / (x @ x)
    e = y - y.mean() - b * x
    se = math.sqrt((e @ e) / (len(y) - 2) / (x @ x))
    return b, b / se


for t in ("QQQ", "SPY"):
    d = daily(t).join(last_half_hour(t), on="day", how="left").sort("day")
    assert d["day"].is_sorted()
    d = d.with_columns(
        pl.col("intraday").shift(1).alias("prev_intraday"),
        pl.col("last30").shift(1).alias("prev_last30"),
    ).drop_nulls(["prev_intraday", "overnight"])
    y = d["overnight"].to_numpy()
    x = d["prev_intraday"].to_numpy()
    b, tt = ols_t(y, x)
    print(
        f"{t} O1: overnight on the session before's open-to-close: slope {b:+.4f}, t {tt:+.2f}  (n {len(y)})"
    )
    dd = d.drop_nulls("prev_last30")
    b2, t2 = ols_t(dd["overnight"].to_numpy(), dd["prev_last30"].to_numpy())
    print(
        f"     overnight on the session before's last half hour: slope {b2:+.4f}, t {t2:+.2f}"
    )
    for lo, hi in ((2016, 2019), (2020, 2026)):
        p = d.filter(pl.col("day").dt.year().is_between(lo, hi))
        b3, t3 = ols_t(p["overnight"].to_numpy(), p["prev_intraday"].to_numpy())
        print(f"     {lo}-{hi}: slope {b3:+.4f}, t {t3:+.2f}")
    on = d.filter(pl.col("day").is_in(list(macro)))["overnight"].to_numpy()
    off = d.filter(~pl.col("day").is_in(list(macro)))["overnight"].to_numpy()
    diff = on.mean() - off.mean()
    se = math.sqrt(on.var(ddof=1) / len(on) + off.var(ddof=1) / len(off))
    print(
        f"{t} O2: overnight into CPI/payrolls mornings {on.mean() * 1e4:+.2f} bps (n {len(on)}) vs other nights {off.mean() * 1e4:+.2f} bps; difference t {diff / se:+.2f}"
    )
