"""Time-series momentum (Moskowitz, Ooi and Pedersen 2012) on the stored ETFs.
Month-end rebalance; each ETF long if its last 12 months' return beat the T-bill, short
otherwise; sized to 10% a year of its own volatility (60-day, known at the rebalance);
1 bp of cost on every unit traded. Total returns (dividend-adjusted closes)."""

import math

import numpy as np
import polars as pl

ETFS = [
    "QQQ",
    "SPY",
    "IWM",
    "DIA",
    "TLT",
    "GLD",
    "HYG",
    "XLB",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLU",
    "XLV",
    "XLY",
]
px = None
for t in ETFS:
    d = pl.read_parquet(f"data/raw/etf_daily/{t}.parquet").select(
        "date", pl.col("close").alias(t)
    )
    px = d if px is None else px.join(d, on="date", how="full", coalesce=True)
px = px.sort("date")
tb = pl.read_csv("data/raw/fred/DTB3.csv", null_values=["."]).select(
    pl.col("observation_date").str.to_date().alias("date"),
    (pl.col("DTB3") / 100).alias("rf"),
)
px = (
    px.join(tb, on="date", how="left")
    .sort("date")
    .with_columns(pl.col("rf").fill_null(strategy="forward"))
)
dates = px["date"].to_numpy()
P = px.select(ETFS).to_numpy()
R = P[1:] / P[:-1] - 1
dates = dates[1:]
rf_daily = px["rf"].to_numpy()[1:] / 252
month = np.array([str(d)[:7] for d in dates])
ends = np.where(month[:-1] != month[1:])[0]  # last day of each month


def run(cols, long_only=False, flat_short=False):
    k = [ETFS.index(c) for c in cols]
    w = np.zeros(len(k))
    out = np.zeros(len(dates))
    for i in range(len(dates)):
        r = np.nan_to_num(R[i, k])
        out[i] = w @ r
        if i in ends_set:
            look = i - 252
            if look < 60:
                continue
            past = np.prod(1 + np.nan_to_num(R[look : i + 1, k]), axis=0) - 1
            rf12 = np.prod(1 + rf_daily[look : i + 1]) - 1
            vol = np.nanstd(R[i - 59 : i + 1, k], axis=0, ddof=1) * math.sqrt(252)
            ok = ~np.isnan(R[look, k]) & (vol > 0)
            sig = np.where(past > rf12, 1.0, -1.0)
            if long_only:
                sig = np.where(sig > 0, 1.0, 0.0)
            new = np.where(
                ok, sig * 0.10 / np.where(vol > 0, vol, 1) / max(ok.sum(), 1), 0.0
            )
            out[i] -= np.abs(new - w).sum() * 1e-4
            w = new
    return out


ends_set = set(ends.tolist())


def stats(name, r, lo, hi):
    m = (dates >= np.datetime64(lo)) & (dates <= np.datetime64(hi))
    x = r[m]
    s = x.mean() / x.std(ddof=1) * math.sqrt(252)
    g = np.prod(1 + x) ** (252 / len(x)) - 1
    dd = (np.cumprod(1 + x) / np.maximum.accumulate(np.cumprod(1 + x)) - 1).min()
    return f"{name:34} {g:+6.1%} a year, vol {x.std(ddof=1) * math.sqrt(252):5.1%}, Sharpe {s:5.2f}, worst fall {dd:6.1%}"


uni_all = ETFS
uni_4 = ["SPY", "TLT", "GLD", "HYG"]
res = {
    "TSMOM, 16 ETFs": run(uni_all),
    "TSMOM, SPY TLT GLD HYG": run(uni_4),
    "same, long or flat": run(uni_4, long_only=True),
}
for lo, hi in (
    ("2008-01-01", "2026-10-14"),
    ("2016-01-26", "2026-10-09"),
    ("2023-01-01", "2026-10-09"),
):
    print(f"-- {lo[:4]} to {hi[:4]}")
    for n, r in res.items():
        print("  " + stats(n, r, lo, hi))
    spy = np.nan_to_num(R[:, ETFS.index("SPY")])
    print("  " + stats("SPY buy and hold", spy, lo, hi))
# correlation with the QQQ rule's day
COST = 0.3e-4 + 0.0000206 / 2
q = (
    pl.read_parquet("data/intraday_pilot/policy/decisions.parquet")
    .filter(pl.col("ticker") == "QQQ")
    .sort("day", "t")
)
q = q.with_columns(pl.col("rule").shift(1).over("day").fill_null(0).alias("before"))
last = pl.col("t") == pl.col("t").max().over("day")
q = q.with_columns(
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
rule = q.group_by("day").agg(pl.col("net").sum().alias("rule")).sort("day")
ts = pl.DataFrame(
    {
        "day": dates.astype("datetime64[D]"),
        "ts4": res["TSMOM, SPY TLT GLD HYG"],
        "ts16": res["TSMOM, 16 ETFs"],
    }
).with_columns(pl.col("day").cast(pl.Date))
j = rule.join(ts, on="day").sort("day")
print(
    "\ncorrelation with the QQQ rule's day, 2016-2026: TSMOM 4 ETFs",
    round(np.corrcoef(j["rule"], j["ts4"])[0, 1], 3),
    " 16 ETFs",
    round(np.corrcoef(j["rule"], j["ts16"])[0, 1], 3),
)
for wts in (0.5, 1.0):
    comb = j["rule"].to_numpy() + wts * j["ts4"].to_numpy()
    print(
        f"  rule + {wts} x TSMOM(4): Sharpe {comb.mean() / comb.std(ddof=1) * math.sqrt(252):.2f}  (rule alone {j['rule'].mean() / j['rule'].std() * math.sqrt(252):.2f})"
    )
