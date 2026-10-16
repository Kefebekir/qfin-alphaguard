"""Intraday pilot: where the QQQ rule earns, by time of day and by kind of day.

Exploration for Phase 2's step 5, on 2016-2022 only; 2023-2026 is kept for a
test written down after it. The literature says intraday momentum comes from
trades that must happen: options dealers hedging (momentum when their gamma
is negative, reversal when positive: Baltussen, Da, Lammers and Martens
2021; Dim, Eraker and Vilkov 2024), leveraged ETFs rebalancing in the last
half hour in the direction of the day's move (Barbon, Beckmeyer, Buraschi
and Moerke), and news (stronger on news days: Gao, Han, Li and Zhou 2018).
This splits the rule's results (policy_data.py's decisions, its own
positions, after QQQ's costs) by:

- the decision's time of day;
- the kind of day: an FOMC announcement (14:00), a CPI or payrolls release
  (08:30, before the open), monthly options expiry, quarterly expiry, the
  last session of a month;
- the volatility regime from the last close: VIX, its term structure
  (VIX9D / VIX and VIX / VIX3M; above 1 is stress) and VVIX;
- for the last two decisions (15:00, 15:30), how far the day has moved
  from the open: leveraged ETFs must buy or sell in proportion to it.

Each line: decisions or days, the mean result in bps of the account, and a
t-statistic over days.

Usage: flow_explore.py POLICY_DIR CBOE_DIR EVENTS_CSV
"""

import math
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

POLICY, CBOE, EVENTS = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
LAST_YEAR = 2022
COST = 0.3e-4 + 0.0000206 / 2


def cboe(name: str, column: str = "CLOSE") -> pl.DataFrame:
    """A Cboe index's daily close, and the close of the session before."""
    frame = pl.read_csv(CBOE / f"{name}_History.csv")
    return (
        frame.select(
            pl.col("DATE").str.to_date("%m/%d/%Y").alias("day"),
            pl.col(column).cast(pl.Float64).alias(name),
        )
        .sort("day")
        .with_columns(pl.col(name).shift(1).alias(f"{name}_prev"))
    )


def third_friday(year: int, month: int) -> date:
    first = date(year, month, 1)
    friday = first + timedelta(days=(4 - first.weekday()) % 7)
    return friday + timedelta(days=14)


d = pl.read_parquet(POLICY / "decisions.parquet").filter(
    (pl.col("ticker") == "QQQ") & (pl.col("day").dt.year() <= LAST_YEAR)
)
d = d.sort("day", "t").with_columns(
    (pl.col("size") * pl.col("rule") * pl.col("ret")).alias("gross"),
    pl.col("rule").shift(1).over("day").fill_null(0).alias("before"),
)
# Each decision's result after costs: what it changed is paid at its fill;
# the close of the day's last position is charged to the last decision.
last = pl.col("t") == pl.col("t").max().over("day")
d = d.with_columns(
    (
        pl.col("gross")
        - pl.col("size")
        * COST
        * (
            (pl.col("rule") - pl.col("before")).abs()
            + pl.when(last).then(pl.col("rule").abs()).otherwise(0)
        )
    ).alias("net")
)
days = d.group_by("day").agg(pl.col("net").sum().alias("pnl")).sort("day")

vix = cboe("VIX")
for other in ("VIX9D", "VIX3M"):
    vix = vix.join(cboe(other), on="day", how="left")
vix = vix.join(cboe("VVIX", "VVIX"), on="day", how="left")
regime = vix.select(
    "day",
    pl.col("VIX_prev").alias("vix"),
    (pl.col("VIX9D_prev") / pl.col("VIX_prev")).alias("short_over_vix"),
    (pl.col("VIX_prev") / pl.col("VIX3M_prev")).alias("vix_over_3m"),
    pl.col("VVIX_prev").alias("vvix"),
)
events = pl.read_csv(EVENTS).with_columns(pl.col("day").str.to_date())
sessions_days = days["day"].to_list()
month_last = days.group_by(pl.col("day").dt.strftime("%Y-%m").alias("month")).agg(
    pl.col("day").max()
)["day"]
opex = set()
for y in range(2016, LAST_YEAR + 1):
    for m in range(1, 13):
        f = third_friday(y, m)
        while f not in set(sessions_days) and f.year == y:
            f -= timedelta(days=1)  # a holiday Friday: Thursday's session
        opex.add(f)
quarterly = {f for f in opex if f.month in (3, 6, 9, 12)}
kinds = {
    "FOMC announcement": set(events.filter(pl.col("event") == "fomc")["day"]),
    "CPI release": set(events.filter(pl.col("event") == "cpi")["day"]),
    "payrolls release": set(events.filter(pl.col("event") == "nfp")["day"]),
    "monthly options expiry": opex - quarterly,
    "quarterly expiry": quarterly,
    "last session of a month": set(month_last),
}
days = days.join(regime, on="day", how="left")


def stats(values: np.ndarray) -> str:
    n = len(values)
    if n < 3:
        return f"n {n:5}"
    m = values.mean() * 1e4
    t = values.mean() / values.std(ddof=1) * math.sqrt(n)
    return f"n {n:5}, mean {m:+7.2f} bps, t {t:+5.2f}"


print(f"QQQ, the rule's own results after costs, 2016-{LAST_YEAR}")
print(f"\nall days: {stats(days['pnl'].to_numpy())}")

print("\n1. by the decision's time (each day's result from that decision):")
by_slot = d.group_by("t").agg(pl.len()).sort("t")
for (t,) in by_slot.select("t").iter_rows():
    g = d.filter(pl.col("t") == t)
    held = g.filter(pl.col("rule") != 0)
    hour, minute = divmod(570 + t, 60)
    print(
        f"  {hour:02}:{minute:02}  in the market {len(held) / max(len(g), 1):4.0%}  "
        f"{stats(g['net'].to_numpy())}"
    )

print("\n2. by kind of day (the day's result):")
for name, chosen in kinds.items():
    on = days.filter(pl.col("day").is_in(list(chosen)))["pnl"].to_numpy()
    off = days.filter(~pl.col("day").is_in(list(chosen)))["pnl"].to_numpy()
    diff = on.mean() - off.mean()
    se = math.sqrt(on.var(ddof=1) / len(on) + off.var(ddof=1) / len(off))
    print(
        f"  {name:26} {stats(on)}; others {off.mean() * 1e4:+6.2f} bps; "
        f"difference t {diff / se:+5.2f}"
    )

print("\n3. by the volatility regime at the last close (the day's result):")
for column, label in (
    ("vix", "VIX"),
    ("short_over_vix", "VIX9D / VIX"),
    ("vix_over_3m", "VIX / VIX3M"),
    ("vvix", "VVIX"),
):
    g = days.drop_nulls(column)
    edges = np.quantile(g[column].to_numpy(), [1 / 3, 2 / 3])
    print(f"  {label}: tercile edges {edges[0]:.3f}, {edges[1]:.3f}")
    for i, (lo, hi) in enumerate(
        ((-np.inf, edges[0]), (edges[0], edges[1]), (edges[1], np.inf))
    ):
        part = g.filter((pl.col(column) > lo) & (pl.col(column) <= hi))
        print(f"    tercile {i + 1}: {stats(part['pnl'].to_numpy())}")

print("\n4. the last two decisions, by the day's move so far (|from the open|):")
late = d.filter(pl.col("t") >= 330).with_columns(
    pl.col("from_open").abs().alias("moved")
)
edges = np.quantile(late["moved"].to_numpy(), [1 / 3, 2 / 3])
for i, (lo, hi) in enumerate(
    ((-np.inf, edges[0]), (edges[0], edges[1]), (edges[1], np.inf))
):
    part = late.filter((pl.col("moved") > lo) & (pl.col("moved") <= hi))
    print(
        f"  tercile {i + 1} (|move| in 14-day vols up to {hi:.2f}): {stats(part['net'].to_numpy())}"
    )
print("  the top tercile by year:")
top = late.filter(pl.col("moved") > edges[1])
for y in range(2016, LAST_YEAR + 1):
    part = top.filter(pl.col("day").dt.year() == y)
    print(f"    {y}: {stats(part['net'].to_numpy())}")
