"""Independent re-check of the overnight leg: EODHD daily bars (dividend-adjusted) instead of
the Alpaca minutes, and a weight set only from the past (equal risk, trailing 250 sessions),
instead of the 0.5 chosen after seeing results."""

import math

import numpy as np
import polars as pl

q = (
    pl.read_parquet("data/raw/etf_daily/QQQ.parquet")
    .sort("date")
    .select(pl.col("date").alias("day"), "open", "close")
)
q = q.with_columns(
    (pl.col("open") / pl.col("close").shift(1) - 1).alias("overnight")
).drop_nulls("overnight")
COST_NIGHT = 2 * 0.3e-4 + 0.0000206  # buy near the close, sell near the open
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
rule = d.group_by("day").agg(pl.col("net").sum().alias("rule")).sort("day")
# the night INTO day t (close t-1 -> open t) belongs with day t's account
j = rule.join(q.select("day", "overnight"), on="day").sort("day")
assert j["day"].is_sorted()
r, on = j["rule"].to_numpy(), j["overnight"].to_numpy() - COST_NIGHT
# equal-risk weight from the past only: rule vol / overnight vol over the 250 sessions before
w = np.full(len(r), np.nan)
for i in range(250, len(r)):
    w[i] = r[i - 250 : i].std(ddof=1) / on[i - 250 : i].std(ddof=1)
ok = ~np.isnan(w)
days = j["day"].to_numpy()


def stats(name, x):
    eq = np.cumprod(1 + x)
    dd = (eq / np.maximum.accumulate(eq) - 1).min()
    print(
        f"  {name:36} {np.prod(1 + x) ** (252 / len(x)) - 1:+6.1%} a year, Sharpe {x.mean() / x.std(ddof=1) * math.sqrt(252):5.2f}, worst fall {dd:6.1%}, worst day {x.min():+.1%}"
    )


print(
    f"EODHD daily bars, {days[ok][0]}..{days[ok][-1]}; mean equal-risk weight {w[ok].mean():.2f} (10-90%: {np.nanpercentile(w, 10):.2f}-{np.nanpercentile(w, 90):.2f}); corr(rule, overnight) {np.corrcoef(r[ok], on[ok])[0, 1]:+.3f}"
)
for label, m in (
    ("2017-2026", ok),
    ("2017-2019", ok & (days < np.datetime64("2020-01-01"))),
    (
        "2020-2022",
        ok
        & (days >= np.datetime64("2020-01-01"))
        & (days < np.datetime64("2023-01-01")),
    ),
    ("2023-2026", ok & (days >= np.datetime64("2023-01-01"))),
):
    print(label)
    stats("rule", r[m])
    stats("overnight at 1x", on[m])
    stats("rule + overnight, equal-risk weight", r[m] + w[m] * on[m])
    stats("rule + overnight at 0.5x", r[m] + 0.5 * on[m])
