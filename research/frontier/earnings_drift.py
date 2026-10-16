"""Does a stock keep moving after its earnings? S&P 500 members, 2016-2026.

"The company beat its earnings, so the stock will rise": the post-earnings announcement drift
(Ball and Brown 1968; Bernard and Thomas 1989), said to have vanished in large stocks since
about 2006 (Martineau 2022). This measures it on our members, with the release times of
edgar_earnings.py (each filing's own header) and the stored daily bars, which are adjusted for
splits and dividends.

- **The reaction day** is the first session whose close comes after the release: a release
  before 16:00 New York time on a session day reacts that day, one after it the next session.
  A second release within 20 sessions of the first is left out.
- **The surprise** is the reaction: the stock's return from the close before the reaction day
  to its close, less SPY's (as Chan, Jegadeesh and Lakonishok 1996 and Brandt et al. 2008,
  since analysts' forecasts are not in our data). Each event is ranked against the events of
  the 365 days before it, so no rank uses the future. Top fifth: the best reactions.
- **The drift** is what comes after: the stock's return less SPY's from the reaction day's
  close to 1, 5, 20 and 60 sessions later, the trade a reader of the news could still make.
- **As a strategy:** each day, hold equally the stocks whose reaction in the top fifth ended
  within the last 60 sessions, against those in the bottom fifth (and the top fifth alone
  against SPY), the members of the S&P 500 on the reaction day only. The daily result, its
  t-statistic, by period, and what a 10 bps round trip costs.

Not an exact test: companies that left the market under their ticker are missing from
EDGAR's ticker list, so the events lean towards survivors.

Usage: earnings_drift.py PRICES_PARQUET SPY_PARQUET EARNINGS_PARQUET
"""

import math
import sys
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

PRICES, SPY, EARNINGS = sys.argv[1], sys.argv[2], sys.argv[3]
NEW_YORK = ZoneInfo("America/New_York")
HORIZONS = (1, 5, 20, 60)
HOLD = 60

prices = pl.read_parquet(PRICES, columns=["date", "ticker", "close", "sp500"]).sort(
    "ticker", "date"
)
spy = pl.read_parquet(SPY, columns=["date", "close"]).sort("date")
days = spy.filter(pl.col("date") >= date(2015, 1, 1))["date"].to_list()
day_index = {d: i for i, d in enumerate(days)}
spy_close = dict(zip(spy["date"].to_list(), spy["close"].to_list(), strict=True))

# Each stock's closes on SPY's sessions, as an array per ticker (NaN where it did not trade).
closes, members = {}, {}
for (ticker,), group in prices.group_by(["ticker"]):
    row = np.full(len(days), np.nan)
    member = np.zeros(len(days), bool)
    for d, c, m in zip(group["date"], group["close"], group["sp500"], strict=True):
        if d in day_index:
            row[day_index[d]] = c
            member[day_index[d]] = m
    closes[ticker], members[ticker] = row, member
spy_row = np.array([spy_close[d] for d in days])


def reaction_index(accepted: datetime) -> int | None:
    """The index of the first session whose close (16:00 New York) is after `accepted`."""
    local = accepted.astimezone(NEW_YORK)
    d = local.date()
    if local.time() >= time(16, 0):
        d += timedelta(days=1)
    while d not in day_index:
        d += timedelta(days=1)
        if d > days[-1]:
            return None
    return day_index[d]


events = []
earnings = pl.read_parquet(EARNINGS).sort("ticker", "accepted")
for (ticker,), group in earnings.group_by(["ticker"]):
    if ticker not in closes:
        continue
    row, member, last = closes[ticker], members[ticker], -10_000
    for accepted in group["accepted"].to_list():
        i = reaction_index(
            accepted.replace(tzinfo=UTC) if accepted.tzinfo is None else accepted
        )
        if i is None or i < 1 or i - last < 20:
            continue
        last = i
        if not member[i] or np.isnan(row[i]) or np.isnan(row[i - 1]):
            continue
        reaction = row[i] / row[i - 1] - spy_row[i] / spy_row[i - 1]
        after = {}
        for h in HORIZONS:
            j = i + h
            after[h] = (
                row[j] / row[i] - spy_row[j] / spy_row[i]
                if j < len(days) and not np.isnan(row[j])
                else np.nan
            )
        events.append(
            {"ticker": ticker, "i": i, "day": days[i], "reaction": reaction}
            | {f"after_{h}": v for h, v in after.items()}
        )

events = pl.DataFrame(events).sort("i", "ticker")
# Rank each event among the events of the 365 days before its reaction day.
day_arr = events["day"].to_numpy()
react = events["reaction"].to_numpy()
ranks = np.full(len(events), np.nan)
for k in range(len(events)):
    window = (day_arr < day_arr[k]) & (day_arr >= day_arr[k] - np.timedelta64(365, "D"))
    if window.sum() >= 500:
        ranks[k] = (react[window] < react[k]).mean()
events = events.with_columns(rank=pl.Series(ranks)).filter(pl.col("rank").is_not_nan())
events = events.with_columns(
    group=pl.when(pl.col("rank") >= 0.8)
    .then(pl.lit("top"))
    .when(pl.col("rank") < 0.2)
    .then(pl.lit("bottom"))
    .otherwise(pl.lit("middle"))
)
print(
    f"{events.height:,} ranked earnings events of {events['ticker'].n_unique()} members, "
    f"{events['day'].min()} to {events['day'].max()}"
)

print(
    "Reaction and drift after it, less SPY, mean in bps (t by event; events overlap in time)"
)
for name in ("top", "middle", "bottom"):
    g = events.filter(pl.col("group") == name)
    cells = [f"reaction {g['reaction'].mean() * 1e4:+7.0f}"]
    for h in HORIZONS:
        v = g[f"after_{h}"].drop_nans().to_numpy()
        cells.append(
            f"{h:>2}d {v.mean() * 1e4:+6.1f} (t {v.mean() / v.std(ddof=1) * math.sqrt(len(v)):+.1f})"
        )
    print(f"   {name:6} {g.height:>6,}: " + ", ".join(cells))

# The strategy, in calendar time.
long_days, short_days = (
    [[] for _ in days],
    [[] for _ in days],
)
for row in events.filter(pl.col("group") != "middle").iter_rows(named=True):
    target = long_days if row["group"] == "top" else short_days
    for j in range(row["i"] + 1, min(row["i"] + HOLD + 1, len(days))):
        target[j].append(row["ticker"])


def day_return(tickers, j):
    values = [
        closes[t][j] / closes[t][j - 1] - 1
        for t in tickers
        if not (np.isnan(closes[t][j]) or np.isnan(closes[t][j - 1]))
    ]
    return float(np.mean(values)) if values else np.nan


series = []
for j in range(1, len(days)):
    market = spy_row[j] / spy_row[j - 1] - 1
    long_r, short_r = day_return(long_days[j], j), day_return(short_days[j], j)
    series.append(
        {
            "day": days[j],
            "long_short": long_r - short_r,
            "long_vs_spy": long_r - market,
            "n_long": len(long_days[j]),
        }
    )
series = pl.DataFrame(series).drop_nans().filter(pl.col("day") >= events["day"].min())


def report(values, label):
    v = np.asarray(values)
    t = v.mean() / v.std(ddof=1) * math.sqrt(len(v))
    return (
        f"{label}: {v.mean() * 252:+.1%} a year, Sharpe {v.mean() / v.std(ddof=1) * math.sqrt(252):+.2f}, "
        f"t {t:+.2f}"
    )


per_day = events.filter(pl.col("group") == "top").height / len(series)
print(
    f"Holding {HOLD} sessions after the reaction; about {series['n_long'].mean():.0f} stocks long"
)
for first, last, label in (
    (date(2016, 1, 1), date(2026, 12, 31), "all"),
    (date(2016, 1, 1), date(2020, 12, 31), "2016-2020"),
    (date(2021, 1, 1), date(2026, 12, 31), "2021-2026"),
):
    part = series.filter(pl.col("day").is_between(first, last))
    print(f"   {label}:")
    print("      " + report(part["long_short"], "top fifth less bottom fifth"))
    print("      " + report(part["long_vs_spy"], "top fifth less SPY        "))
cost = per_day * 2 * 2 * 0.0005 / max(series["n_long"].mean(), 1) * 252
print(
    f"A 10 bps round trip on each leg costs about {cost:.2%} a year for the long-short "
    f"({per_day:.1f} new top events a day)"
)
