"""B5: two-state Gaussian HMM on QQQ daily returns, fitted each year on the years before only;
each day's state probability from the forward filter on information up to the day before."""

import math

import numpy as np
import polars as pl
from hmmlearn.hmm import GaussianHMM
from scipy.stats import norm

q = (
    pl.read_parquet("data/raw/etf_daily/QQQ.parquet")
    .sort("date")
    .with_columns((pl.col("close") / pl.col("close").shift(1) - 1).alias("r"))
    .drop_nulls("r")
)
dates, r = q["date"].to_numpy(), q["r"].to_numpy()
years = q["date"].dt.year().to_numpy()
prob_high = np.full(len(r), np.nan)
for Y in range(2016, 2027):
    train = years < Y
    m = GaussianHMM(
        n_components=2, covariance_type="diag", n_iter=200, random_state=0
    ).fit(r[train].reshape(-1, 1))
    mu, sd = m.means_.ravel(), np.sqrt(m.covars_.ravel())
    high = int(np.argmax(sd))
    A = m.transmat_
    # forward filter over the training years, then on through year Y, one day at a time
    alpha = m.startprob_.copy()
    for i in range(len(r)):
        if years[i] > Y:
            break
        pred = (
            alpha @ A if i > 0 else alpha
        )  # the state today, from information up to yesterday
        if years[i] == Y:
            prob_high[i] = pred[high]
        lik = norm.pdf(r[i], mu, sd)
        alpha = pred * lik
        alpha /= alpha.sum()
hmm = (
    pl.DataFrame({"day": dates, "p_high": prob_high})
    .with_columns(pl.col("day").cast(pl.Date))
    .drop_nans("p_high")
)
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
rule = d.group_by("day").agg(pl.col("net").sum().alias("pnl"))
j = rule.join(hmm, on="day").sort("day")
hi = j.filter(pl.col("p_high") > 0.5)["pnl"].to_numpy()
lo = j.filter(pl.col("p_high") <= 0.5)["pnl"].to_numpy()
diff = hi.mean() - lo.mean()
t = diff / math.sqrt(hi.var(ddof=1) / len(hi) + lo.var(ddof=1) / len(lo))
sh = lambda x: x.mean() / x.std(ddof=1) * math.sqrt(252)
print(
    f"B5: rule in the HMM's high-volatility state {hi.mean() * 1e4:+.2f} bps ({len(hi)} days, Sharpe {sh(hi):.2f}) vs low {lo.mean() * 1e4:+.2f} bps ({len(lo)}, Sharpe {sh(lo):.2f}); t {t:+.2f} -> {'PASS' if t >= 2.64 else 'fail'}"
)
