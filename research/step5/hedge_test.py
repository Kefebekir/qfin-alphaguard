"""Online expert allocation over the rule and its challengers, 2019-2026.

Every expert's daily result is out of sample: the learned policies of
policy_nets.py were fitted walk-forward, year by year; the gamma sizings use
trailing ranks only. Settings fixed from theory before the run:

- Hedge (Freund and Schapire 1997): weights proportional to
  exp(-eta x cumulative loss), eta = sqrt(8 ln N / T);
- Fixed-Share (Herbster and Warmuth 1998): the same, then a share alpha = 0.01
  of the weight spread evenly each day, so it can follow a change of regime;
- a day's loss: minus the expert's result over 3 x the rule's daily volatility
  over the previous 250 days, clipped to [-1, 1] and mapped to [0, 1].

The weights for a day use the results up to the day before only.
"""

import math
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from qfin_alphaguard.data.intraday import intraday_path
from qfin_alphaguard.sessions import sessions

COST = 0.3e-4 + 0.0000206 / 2
t = pl.concat(
    [
        pl.read_parquet(f"data/intraday_pilot/policy/{tag}/daily_{tag}.parquet")
        for tag in ("selection", "holdout")
    ]
).with_columns((pl.col("model") + "_" + pl.col("pooled")).alias("expert"))
wide = t.pivot(on="expert", index="day", values="pnl").sort("day")

# The gamma-state score, as gamma_test.py computes it.
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
    k, c = k[keep], g["close"].to_numpy()[keep]
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


def trailing_rank(x, n=250):
    r = np.full(len(x), np.nan)
    for i in range(n, len(x)):
        past = x[i - n : i]
        past = past[~np.isnan(past)]
        if len(past) > 50 and not np.isnan(x[i]):
            r[i] = (past < x[i]).mean()
    return r


f = f.with_columns(
    pl.Series(
        "score",
        (trailing_rank(f["vr_y"].to_numpy()) + trailing_rank(f["rviv_y"].to_numpy()))
        / 2,
    )
)
size = (
    pl.read_parquet("data/intraday_pilot/policy/decisions.parquet")
    .filter(pl.col("ticker") == "QQQ")
    .group_by("day")
    .agg(pl.col("size").first())
)
ev = pl.read_csv("data/raw/calendar/events.csv").with_columns(
    pl.col("day").str.to_date()
)
fomc = set(ev.filter(pl.col("event") == "fomc")["day"])
wide = wide.join(f.select("day", "score"), on="day").join(size, on="day").sort("day")
assert wide["day"].is_sorted()
rule = wide["rule_QQQ"].to_numpy()
score = wide["score"].to_numpy()
room = 2.0 / wide["size"].to_numpy()
third = np.where(score < 1 / 3, 0, np.where(score < 2 / 3, 1, 2))
extra = {
    "gamma_H1": rule * np.minimum(np.array([0.0, 1.0, 1.5])[third], room),
    "gamma_inverse": rule * np.minimum(np.array([1.5, 1.0, 0.0])[third], room),
    "no_fomc": np.where(np.isin(wide["day"].to_numpy(), list(fomc)), 0.0, rule),
    "half_rule": rule * 0.5,
    "flat": np.zeros_like(rule),
}
names = [c for c in wide.columns if c not in ("day", "score", "size")] + list(extra)
R = np.column_stack(
    [wide[c].to_numpy() if c in wide.columns else extra[c] for c in names]
)
days = wide["day"].to_numpy()
N, T = R.shape[1], R.shape[0]
eta = math.sqrt(8 * math.log(N) / T)
vol = np.array(
    [
        rule[max(0, i - 250) : i].std(ddof=1) if i > 20 else rule[:250].std(ddof=1)
        for i in range(T)
    ]
)
loss = (np.clip(-R / (3 * vol[:, None]), -1, 1) + 1) / 2


def allocate(alpha):
    w = np.ones(N) / N
    out = np.zeros(T)
    weights = np.zeros((T, N))
    for i in range(T):
        weights[i] = w
        out[i] = w @ R[i]
        w = w * np.exp(-eta * loss[i])
        w /= w.sum()
        w = (1 - alpha) * w + alpha / N
    return out, weights


def line(name, r, base):
    g = np.prod(1 + r) ** (252 / len(r)) - 1
    s = r.mean() / r.std(ddof=1) * math.sqrt(252)
    diff = r - base
    tt = (
        diff.mean() / diff.std(ddof=1) * math.sqrt(len(diff)) if diff.std() > 0 else 0.0
    )
    print(
        f"  {name:30} {g:+6.1%} a year, Sharpe {s:5.2f}; minus the rule {diff.mean() * 252:+6.2%}, t {tt:+5.2f}"
    )


print(f"{N} experts, {T} days {days[0]} .. {days[-1]}, eta {eta:.4f}")
hedge, wh = allocate(0.0)
share, ws = allocate(0.01)
equal = R.mean(1)
for label, on in (
    ("2019-2026", np.ones(T, bool)),
    ("2019-2022", days < np.datetime64("2023-01-01")),
    ("2023-2026", days >= np.datetime64("2023-01-01")),
):
    print(label)
    line("rule", rule[on], rule[on])
    line("Hedge", hedge[on], rule[on])
    line("Fixed-Share", share[on], rule[on])
    line("equal weight of all experts", equal[on], rule[on])
    sh = [
        R[on, j].mean() / R[on, j].std(ddof=1) * math.sqrt(252)
        if R[on, j].std() > 0
        else -9
        for j in range(N)
    ]
    best = int(np.argmax(sh))
    line(f"best in hindsight: {names[best]}", R[on, best], rule[on])
print("\nthe largest weights at the end (Hedge | Fixed-Share):")
for j in np.argsort(-wh[-1])[:6]:
    print(f"  {names[j]:20} {wh[-1, j]:.3f} | {ws[-1, j]:.3f}")
print("the rule's weight at the start of each year (Hedge, Fixed-Share):")
for y in range(2019, 2027):
    i = int(np.argmax(days >= np.datetime64(f"{y}-01-01")))
    j = names.index("rule_QQQ")
    print(f"  {y}: {wh[i, j]:.3f}, {ws[i, j]:.3f}")
