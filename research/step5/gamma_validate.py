"""The pre-registered validation V-G (preregistration_validation.txt), run once.

QQQ's GEX is computed by gex.py's gex_series, unchanged, over 2016-2026: the 2016-2022 yearly
open interest files and the 2023-2026 daily ones, one record per contract and day (from the
Pillar format each exchange repeats it), so the trailing 250-session rank runs on into 2023.
Then, on 2023-01-03 to 2026-10-09: the QQQ rule's mean daily result after costs on days whose
rank is at least 2/3, against the other days (one-sided Welch t <= -2.0 passes).
"""

import math
import runpy
from pathlib import Path

import numpy as np
import polars as pl

g = runpy.run_path("research/step5/gex.py")  # PHASE unset: only its functions
gex_series = g["gex_series"]

ROOT = Path("data/step5/gamma_real/extended")
ROOT.mkdir(parents=True, exist_ok=True)
files = [f"data/raw/opra/QQQ_oi_{y}.parquet" for y in range(2016, 2023)]
daily = sorted(Path("data/raw/opra/QQQ_daily").glob("20*.parquet"))
for year in range(2023, 2027):
    part = pl.concat(
        [pl.read_parquet(f) for f in daily if f.name.startswith(str(year))]
    )
    part = (
        part.with_columns(
            pl.col("ts_event")
            .dt.convert_time_zone("America/New_York")
            .dt.date()
            .alias("_day")
        )
        .unique(["_day", "symbol"], keep="first", maintain_order=True)
        .drop("_day")
    )
    out = ROOT / f"QQQ_oi_{year}.parquet"
    part.write_parquet(out)
    files.append(str(out))
print(f"{len(daily)} sessions of 2023-2026 open interest")

gex = gex_series("QQQ", "VXN", years=range(2016, 2027), oi_files=files)
gex.write_parquet(ROOT / "gex_QQQ_2016_2026.parquet")
old = pl.read_parquet("data/step5/gamma_real/gex_QQQ.parquet")
check = gex.filter(pl.col("day") <= old["day"].max()).select("day", "gex_adv")
print("2016-2022 GEX unchanged:", check.equals(old.select("day", "gex_adv")))

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
j = (
    rule.join(gex, on="day")
    .sort("day")
    .filter(
        pl.col("rank").is_not_nan()
        & (pl.col("day") >= pl.date(2023, 1, 3))
        & (pl.col("day") <= pl.date(2026, 10, 9))
    )
)
top = j.filter(pl.col("rank") >= 2 / 3)["pnl"].to_numpy()
rest = j.filter(pl.col("rank") < 2 / 3)["pnl"].to_numpy()
diff = top.mean() - rest.mean()
t = diff / math.sqrt(top.var(ddof=1) / len(top) + rest.var(ddof=1) / len(rest))
print(
    f"V-G, {j.height} days 2023-2026: top third {top.mean() * 1e4:+.2f} bps ({len(top)} days) vs "
    f"other days {rest.mean() * 1e4:+.2f} bps ({len(rest)}); Welch t {t:+.2f} -> "
    f"{'PASS' if t <= -2.0 else 'fail'} (needs <= -2.0)"
)


def sharpe(x):
    return x.mean() / x.std(ddof=1) * math.sqrt(252)


x, y = j["rank"].to_numpy(), j["pnl"].to_numpy()
for lo, hi in ((0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.01)):
    p = y[(x >= lo) & (x < hi)]
    print(
        f"   rank {lo:.2f}-{min(hi, 1):.2f}: {len(p)} days, mean {p.mean() * 1e4:+.2f} bps, Sharpe {sharpe(p):.2f}"
    )
years = j["day"].dt.year().to_numpy()
for yr in range(2023, 2027):
    a, b = y[(years == yr) & (x >= 2 / 3)], y[(years == yr) & (x < 2 / 3)]
    print(
        f"   {yr}: top third {a.mean() * 1e4:+.2f} bps ({len(a)}), others {b.mean() * 1e4:+.2f} bps ({len(b)})"
    )
flat = np.where(x >= 2 / 3, 0.0, y)
print(
    f"   the rule {np.prod(1 + y) ** (252 / len(y)) - 1:+.1%} a year, Sharpe {sharpe(y):.2f}; flat on top-third "
    f"days {np.prod(1 + flat) ** (252 / len(flat)) - 1:+.1%}, Sharpe {sharpe(flat):.2f}"
)
