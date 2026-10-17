"""Pre-registration F1: the QQQ rule, unchanged, on CME Nikkei 225 futures (NIY) in Tokyo.

The Tokyo Stock Exchange's sessions (exchange_calendars XTKS): from 09:00 Tokyo time to the
close (15:00, 15:30 from 5 November 2024), the lunch break's CME minutes kept, no decision at
11:30 or 12:00. session_rule.py runs momentum.py's rule on them, exiting at the last minute's
open; a session on a new contract is not traded. Costs: one tick (5 index points) a fill plus
0.1 bps; half a tick reported too.

Two modes:
- implementation: 2016 only, the implementation year, with checks of the sessions' bars;
- test: 3 January 2017 to 9 October 2026, once. The result goes to OUT_DIR/F1_result.json,
  and the run refuses if that file exists. Also reported: each year, half-tick costs, the
  correlation with the QQQ rule's daily returns (RULE_DAYS, the engine's) and the two rules
  together at equal risk, each scaled by its volatility over the year before.

Usage: nikkei_f1.py implementation|test DBN_FILE RULE_DAYS_PARQUET OUT_DIR
"""

import json
import math
import subprocess
import sys
from datetime import date
from pathlib import Path

import databento as db
import exchange_calendars as xc
import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from session_rule import Session, simulate

MODE, DBN, RULE_DAYS, OUT = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4])
FIRST, TEST_FROM, LAST = date(2016, 1, 4), date(2017, 1, 3), date(2026, 10, 9)
TICK = 5.0
FEES = 0.1e-4
if MODE not in ("implementation", "test"):
    sys.exit("mode must be implementation or test")
lock = OUT / "F1_result.json"
if MODE == "test" and lock.exists():
    sys.exit(f"F1 has been run: {lock}")

bars = db.DBNStore.from_file(DBN).to_df()  # indexed by each minute's start, UTC
calendar = xc.get_calendar("XTKS", start=str(FIRST), end=str(LAST))
times = bars.index.asi8  # nanoseconds since 1970, UTC
sessions, checks = [], []
for day in calendar.sessions_in_range(str(FIRST), str(LAST)):
    open_, close = calendar.session_open(day), calendar.session_close(day)
    lunch = (calendar.session_break_start(day), calendar.session_break_end(day))
    a, b = (
        np.searchsorted(times, open_.value),
        np.searchsorted(times, close.value),
    )
    g = bars.iloc[a:b]
    length = int((close - open_) / pd.Timedelta(minutes=1))
    if len(g) == 0:
        checks.append((day.date(), length, 0, None))
        continue
    contracts = g["instrument_id"].value_counts()
    minute = ((g.index - open_) / pd.Timedelta(minutes=1)).to_numpy().astype(int)
    decisions = [
        t
        for t in range(30, length, 30)
        if not (lunch[0] <= open_ + pd.Timedelta(minutes=t) < lunch[1])
    ]
    sessions.append(
        Session(
            day=day.date(),
            minute=minute,
            open=g["open"].to_numpy(float),
            high=g["high"].to_numpy(float),
            low=g["low"].to_numpy(float),
            close=g["close"].to_numpy(float),
            volume=g["volume"].to_numpy(float),
            length=length,
            decisions=decisions,
            contract=int(contracts.index[0]),
        )
    )
    checks.append((day.date(), length, len(g), len(contracts)))

result = pl.DataFrame(
    [
        {
            "day": d.day,
            "gross": d.gross,
            "traded": d.traded,
            "size": d.size,
            "price": d.price,
            "held": d.traded_today,
        }
        for d in simulate(sessions, exit_at="last_open")
    ]
).with_columns(
    net=pl.col("gross") * pl.col("size")
    - pl.col("traded") * pl.col("size") * (TICK / pl.col("price") + FEES),
    net_half_tick=pl.col("gross") * pl.col("size")
    - pl.col("traded") * pl.col("size") * (TICK / 2 / pl.col("price") + FEES),
)


def summary(frame: pl.DataFrame, column: str = "net") -> dict:
    r = frame[column].to_numpy()
    return {
        "days": len(r),
        "traded_days": int(frame["held"].sum()),
        "annual": float(r.mean() * 252),
        "sharpe": float(r.mean() / r.std(ddof=1) * math.sqrt(252)),
        "t": float(r.mean() / r.std(ddof=1) * math.sqrt(len(r))),
        "mean_bps": float(r.mean() * 1e4),
    }


check = pl.DataFrame(
    checks, schema=["day", "minutes", "bars", "contracts"], orient="row"
)
if MODE == "implementation":
    year = check.filter(pl.col("day").dt.year() == 2016)
    print(
        f"2016: {year.height} Tokyo sessions; minutes with a bar: "
        f"{year['bars'].sum() / year['minutes'].sum():.0%}; sessions with no bar: "
        f"{(year['bars'] == 0).sum()}; with two contracts: {(year['contracts'] > 1).sum()}"
    )
    print(f"session lengths (minutes): {sorted(set(year['minutes'].to_list()))}")
    print(f"decision minutes: {sessions[20].decisions}")
    r2016 = result.filter(pl.col("day").dt.year() == 2016)
    print(f"2016 from its 16th session: {summary(r2016)}")
    print(f"rolls not traded in 2016: {int((r2016['size'] == 0).sum())}")
    print(
        f"sizes: {r2016.filter(pl.col('size') > 0)['size'].min():.2f} to "
        f"{r2016['size'].max():.2f}; prices {r2016['price'].min():.0f} to {r2016['price'].max():.0f}"
    )
    sys.exit(0)

test = result.filter(pl.col("day") >= TEST_FROM)
out = {
    "test": "F1",
    "period": [str(TEST_FROM), str(LAST)],
    "one_tick": summary(test),
    "half_tick": summary(test, "net_half_tick"),
    "passes": summary(test)["t"] >= 2.0,
    "years": {
        str(y): summary(test.filter(pl.col("day").dt.year() == y))
        for y in sorted(set(test["day"].dt.year().to_list()))
    },
    "implementation_year_2016": summary(result.filter(pl.col("day") < TEST_FROM)),
}
qqq = pl.read_parquet(RULE_DAYS).select("day", pl.col("rule").alias("qqq"))
both = (
    test.select("day", "net")
    .join(qqq, on="day", how="full", coalesce=True)
    .fill_null(0.0)
)
both = both.filter(pl.col("day").is_between(TEST_FROM, LAST)).sort("day")
out["correlation_with_qqq_rule"] = float(np.corrcoef(both["net"], both["qqq"])[0, 1])
combined = []
for y in sorted(set(both["day"].dt.year().to_list())):
    prior = both.filter(pl.col("day").dt.year() == y - 1)
    if prior.height == 0:
        prior = pl.concat(
            [result.filter(pl.col("day").dt.year() == y - 1).select("day", "net")]
        )
        prior = prior.join(qqq, on="day", how="full", coalesce=True).fill_null(0.0)
    sn, sq = prior["net"].std(), prior["qqq"].std()
    this = both.filter(pl.col("day").dt.year() == y)
    combined.append((this["net"] / sn + this["qqq"] / sq).to_numpy() / 2)
c = np.concatenate(combined)
q = both["qqq"].to_numpy()
out["sharpe_qqq_rule_alone"] = float(q.mean() / q.std(ddof=1) * math.sqrt(252))
out["sharpe_together_equal_risk"] = float(c.mean() / c.std(ddof=1) * math.sqrt(252))
out["commit"] = subprocess.run(
    ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
).stdout.strip()
OUT.mkdir(parents=True, exist_ok=True)
lock.write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
