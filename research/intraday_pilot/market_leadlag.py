"""Intraday pilot: predict QQQ's next 15 minutes from the last 15 minutes of
many assets, with machine learning (Aleti, Bollerslev and Siggaard, 2025;
Huddleston, Liu and Stentoft, 2023).

Both papers find the market's next few minutes predictable from the lagged
intraday returns of many portfolios or stocks, with regularised and
tree-based models and an equal-weighted ensemble of them, and trade it on
SPY after half-spread costs: a Sharpe ratio of 1.37 in 2004-2020 (15-minute
returns of 272 factor portfolios), and 0.98 in their sample (5-minute
returns of the S&P 500's members). Both say it was stronger in crises.

Here the predictors are what we store: the stocks in the trading universe in
every year since 2016 (20 of them), QQQ, and the ETFs whose 1-minute bars
are stored for every year. For each asset at the end of each 15-minute slot:
its return over that slot, over the last hour, since the open, and the
opening gap (for an asset with daily bars in PRICES, adjusted for its
splits and dividends). The target is QQQ's return from the next minute's
open to the open 15 minutes later (to the close for the day's last slot);
for training it is clipped at 4 standard deviations of the training years,
to leave out jumps, which the papers find unpredictable.

Models, walk-forward by year (hyper-parameters chosen on the last 20% of the
years before, then refitted on all of them): ridge, lasso, elastic net,
principal-component and partial-least-squares regressions, gradient-boosted
trees, a small neural network, and their equal-weighted average (ensemble).
Strategies, as in the 2025 paper, at 1x and flat at every close:
- sign: long if the forecast is positive, short otherwise;
- s_sign: change side only when the forecast is beyond one side's cost,
  otherwise keep the position; s_long: the same, long or flat.
Costs: 0.15 bps of value and 0.006 USD a share (fees and half a cent of
spread) on every change.

Usage: market_leadlag.py ROOT PRICES OUT [TARGET]
"""

import math
import sys
import time
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor

from qfin_alphaguard.data.intraday import intraday_path
from qfin_alphaguard.sessions import sessions

warnings.filterwarnings("ignore", category=ConvergenceWarning)
ROOT, PRICES, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
TARGET = sys.argv[4] if len(sys.argv) > 4 else "QQQ"
STOCKS = [
    "AAPL",
    "AMZN",
    "BA",
    "BAC",
    "CSCO",
    "CVX",
    "GOOGL",
    "HD",
    "INTC",
    "JNJ",
    "JPM",
    "MRK",
    "MSFT",
    "NFLX",
    "PFE",
    "PG",
    "QCOM",
    "V",
    "WMT",
    "XOM",
]
ETFS = [
    "QQQ",
    "SPY",
    "IWM",
    "DIA",
    "TLT",
    "GLD",
    "HYG",
    "XLK",
    "XLF",
    "XLE",
    "XLV",
    "XLI",
    "XLY",
    "XLP",
    "XLU",
    "XLB",
]
YEARS = range(2016, 2027)
SLOT = 15
PER_VALUE, PER_SHARE = 0.15e-4, 0.006
assets = [
    a
    for a in [TARGET, *ETFS, *STOCKS]
    if all(intraday_path(ROOT, a, y).exists() for y in YEARS)
]
assets = list(dict.fromkeys(assets))
print("assets:", len(assets), assets)

schedule = pl.DataFrame(
    [
        {
            "day": s.day,
            "open_at": s.open,
            "slots": int((s.close - s.open).total_seconds() // 60) // SLOT,
        }
        for s in sessions(date(YEARS.start, 1, 1), date(YEARS.stop - 1, 12, 31))
    ],
    schema={"day": pl.Date, "open_at": pl.Datetime("us", "UTC"), "slots": pl.Int64},
)


def grid(asset: str) -> pl.DataFrame:
    """One row per day and slot: the price at the slot's end, and the fill
    price at its start (the open of its first minute)."""
    m = (
        pl.concat(
            [
                pl.read_parquet(intraday_path(ROOT, asset, y)).select(
                    "start", "open", "close"
                )
                for y in YEARS
            ]
        )
        .with_columns(pl.col("start").dt.date().alias("day"))
        .join(schedule, on="day")
        .with_columns(
            ((pl.col("start") - pl.col("open_at")).dt.total_minutes()).alias("k")
        )
        .filter((pl.col("k") >= 0) & (pl.col("k") < pl.col("slots") * SLOT))
        .sort("start")
        .with_columns((pl.col("k") // SLOT).alias("slot"))
    )
    by_slot = m.group_by("day", "slot").agg(
        pl.col("close").last().alias("end"),
        pl.col("open")
        .filter(pl.col("k") == pl.col("slot") * SLOT)
        .first()
        .alias("start_open"),
    )
    first = m.group_by("day").agg(pl.col("open").first().alias("day_open"))
    full = schedule.select("day", pl.int_ranges(0, "slots").alias("slot")).explode(
        "slot", empty_as_null=True
    )
    return (
        full.join(by_slot, on=["day", "slot"], how="left")
        .join(first, on="day", how="left")
        .sort("day", "slot")
        .with_columns(pl.col("end").forward_fill().over("day"))
        .with_columns(
            pl.coalesce(
                "start_open", pl.col("end").shift(1).over("day"), "day_open"
            ).alias("fill")
        )
        .select("day", "slot", "end", "fill", "day_open")
    )


# adjustment_t / adjustment_t-1: what turns an overnight return as traded
# into one adjusted for that night's split or dividend
adjustments = (
    pl.read_parquet(PRICES)
    .sort("ticker", "date")
    .select(
        "ticker",
        pl.col("date").alias("day"),
        (pl.col("adjustment") / pl.col("adjustment").shift(1))
        .over("ticker")
        .alias("adjust"),
    )
)
frames = []
for a in assets:
    g = grid(a).join(
        adjustments.filter(pl.col("ticker") == a).drop("ticker"), on="day", how="left"
    )
    prev_close = (
        g.group_by("day")
        .agg(pl.col("end").last().alias("close"))
        .sort("day")
        .with_columns(pl.col("close").shift(1).alias("prev_close"))
    )
    g = g.join(
        prev_close.select("day", "prev_close"), on="day", how="left"
    ).with_columns(
        (pl.col("end") / pl.coalesce(pl.col("end").shift(1).over("day"), "day_open"))
        .log()
        .alias("r15"),
    )
    g = g.with_columns(
        pl.col("r15").rolling_sum(4, min_samples=1).over("day").alias("r60"),
        (pl.col("end") / pl.col("day_open")).log().alias("today"),
        (pl.col("day_open") / pl.col("prev_close") * pl.col("adjust").fill_null(1.0))
        .log()
        .alias("gap"),
    )
    cols = ["r15", "r60", "today", "gap"]
    if a == TARGET:
        # Each decision at a slot's end trades at the next slot's fill and is
        # held to the fill after it (the close for the last slot).
        day_close = g.group_by("day").agg(pl.col("end").last().alias("day_close"))
        g = (
            g.join(day_close, on="day")
            .with_columns(
                pl.col("fill").shift(-1).over("day").alias("entry"),
                pl.coalesce(pl.col("fill").shift(-2).over("day"), "day_close").alias(
                    "exit"
                ),
            )
            .with_columns((pl.col("exit") / pl.col("entry") - 1).alias("target"))
        )
        cols += ["target", "entry"]
    frames.append(g.select("day", "slot", *[pl.col(c).alias(f"{a}:{c}") for c in cols]))

d = frames[0]
for f in frames[1:]:
    d = d.join(f, on=["day", "slot"], how="left")
features = [
    c
    for c in d.columns
    if ":" in c and c.split(":")[1] in ("r15", "r60", "today", "gap")
]
d = (
    d.with_columns(pl.col("slot").alias("time"))
    .filter(pl.col(f"{TARGET}:entry").is_not_null())
    .fill_nan(None)
    .drop_nulls([*features, f"{TARGET}:target"])
    .sort("day", "slot")
)
features.append("time")
print(f"{d.height} decisions on {d['day'].n_unique()} days")
X = d.select(features).to_numpy()
ret = d[f"{TARGET}:target"].to_numpy()
years = d["day"].dt.year().to_numpy()
cost = PER_VALUE + PER_SHARE / d[f"{TARGET}:entry"].to_numpy()

own = [f"{TARGET}:{c}" for c in ("r15", "r60", "today", "gap")] + ["time"]
own_index = [features.index(c) for c in own]


class Pcr:
    def __init__(self, k: int) -> None:
        self.k = k

    def fit(self, x, t):
        self.pca = PCA(self.k).fit(x)
        self.ols = LinearRegression().fit(self.pca.transform(x), t)
        return self

    def predict(self, x):
        return self.ols.predict(self.pca.transform(x))


class Pls:
    def __init__(self, k: int) -> None:
        self.model = PLSRegression(k, scale=False)

    def fit(self, x, t):
        self.model.fit(x, t)
        return self

    def predict(self, x):
        return self.model.predict(x).ravel()


GRIDS = {
    "ridge": [lambda a=a: Ridge(alpha=a) for a in (1e2, 1e3, 1e4, 1e5)],
    "lasso": [
        lambda a=a: Lasso(alpha=a, max_iter=2000) for a in (3e-4, 1e-3, 3e-3, 1e-2)
    ],
    "enet": [
        lambda a=a: ElasticNet(alpha=a, l1_ratio=0.5, max_iter=2000)
        for a in (6e-4, 2e-3, 6e-3, 2e-2)
    ],
    "pcr": [lambda k=k: Pcr(k) for k in (1, 3, 5, 10)],
    "pls": [lambda k=k: Pls(k) for k in (1, 2, 3)],
    "gbrt": [
        lambda n=n: HistGradientBoostingRegressor(
            max_iter=n,
            learning_rate=0.02,
            max_leaf_nodes=7,
            min_samples_leaf=500,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=0,
        )
        for n in (50, 150, 400)
    ],
    "fnn": [
        lambda a=a: MLPRegressor(
            hidden_layer_sizes=(16,),
            alpha=a,
            learning_rate_init=1e-3,
            max_iter=50,
            early_stopping=True,
            random_state=0,
        )
        for a in (0.1, 1.0, 10.0)
    ],
}
names = [*GRIDS, "ensemble", "ar_own"]
pred = {n: np.full(len(d), np.nan) for n in names}
for year in range(2018, YEARS.stop):
    train, test = years < year, years == year
    if not test.any():
        continue
    cut = np.quantile(np.nonzero(train)[0], 0.8)
    fit_part, valid = (
        train & (np.arange(len(d)) < cut),
        train & (np.arange(len(d)) >= cut),
    )
    mean, sd = X[train].mean(0), X[train].std(0) + 1e-12
    Z = (X - mean) / sd
    bound = 4 * ret[train].std()  # from the training years only
    y = np.clip(ret, -bound, bound)
    scale = y[train].std()
    seconds = {}
    for name, grid_models in GRIDS.items():
        started = time.perf_counter()
        best, best_mse = None, np.inf
        for make in grid_models:
            model = make().fit(Z[fit_part], y[fit_part] / scale)
            mse = np.mean((model.predict(Z[valid]) * scale - y[valid]) ** 2)
            if mse < best_mse:
                best, best_mse = make, mse
        pred[name][test] = (
            best().fit(Z[train], y[train] / scale).predict(Z[test]) * scale
        )
        seconds[name] = round(time.perf_counter() - started)
    pred["ensemble"][test] = np.mean([pred[n][test] for n in GRIDS], axis=0)
    own_model = Ridge(alpha=1e3).fit(Z[train][:, own_index], y[train] / scale)
    pred["ar_own"][test] = own_model.predict(Z[test][:, own_index]) * scale
    print(f"  {year} done; seconds per model: {seconds}", flush=True)

tested = years >= 2018
days = d["day"].to_numpy()
new_day = np.concatenate([[True], days[1:] != days[:-1]])
last_of_day = np.concatenate([days[1:] != days[:-1], [True]])


def run(forecast: np.ndarray, kind: str) -> np.ndarray:
    pos = np.zeros(len(forecast))
    held = 0.0
    for i in range(len(forecast)):
        if new_day[i]:
            held = 0.0
        f = forecast[i]
        if kind == "sign":
            held = 1.0 if f >= 0 else -1.0
        elif kind == "s_sign":
            held = 1.0 if f > cost[i] else (-1.0 if f < -cost[i] else held)
        elif kind == "s_long":
            held = 1.0 if f > cost[i] else (0.0 if f < -cost[i] else held)
        pos[i] = held
    prev = np.concatenate([[0.0], pos[:-1]])
    prev[new_day] = 0.0
    traded = np.abs(pos - prev) + np.where(last_of_day, np.abs(pos), 0.0)
    return pos * ret - traded * cost


frame = pl.DataFrame({"day": days, "tested": tested})
print(
    f"{TARGET}, 15-minute decisions, out of sample 2018-{YEARS.stop - 1}, 1x, after costs"
)
for name in names:
    f = pred[name][tested]
    r2 = 1 - np.sum((ret[tested] - f) ** 2) / np.sum(ret[tested] ** 2)
    hit = np.mean(np.sign(f) == np.sign(ret[tested]))
    line = f"{name:9} R2 {r2 * 100:+.3f}%, sign right {hit:.1%}"
    for kind in ("sign", "s_sign", "s_long"):
        pnl = run(np.where(tested, pred[name], 0.0), kind)
        r = (
            frame.with_columns(pl.Series("pnl", pnl))
            .filter("tested")
            .group_by("day")
            .agg(pl.col("pnl").sum())
            .sort("day")["pnl"]
            .to_numpy()
        )
        line += f" | {kind} {np.mean(r) * 252:+.1%} a year, Sharpe {r.mean() / r.std(ddof=1) * math.sqrt(252):+.2f}"
    print(line)
hold = (
    frame.with_columns(pl.Series("r", ret))
    .filter("tested")
    .group_by("day")
    .agg(pl.col("r").sum())["r"]
    .to_numpy()
)
print(
    f"holding {TARGET} from the first decision's fill to the close, before costs: "
    f"{hold.mean() * 252:+.1%} a year, Sharpe {hold.mean() / hold.std(ddof=1) * math.sqrt(252):+.2f}"
)
out = d.select("day", "slot").with_columns(
    pl.Series("ret", ret), *[pl.Series(n, pred[n]) for n in names]
)
out.write_parquet(OUT)
