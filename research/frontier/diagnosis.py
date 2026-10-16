"""Why the ML does not beat the rule: what the price data can and cannot tell, measured.

The half-hourly decisions of policy_data.py (16 ETFs, 2016-2026): at each HH:00 and HH:30
from 10:00, what is known at the end of the minute before, and the return from that decision's
fill to the next one's. Walk-forward: each test year's model is fitted on the years before it;
the test years are 2018-2026. Returns are divided by the 14-day daily volatility, so that the
16 ETFs can be pooled.

1. Direction against size. Boosted trees and ridge predict the next half hour's return (its
   sign and size together) and, separately, its size alone (the absolute return). Out of
   sample R^2: for the return against a forecast of zero, for the size against the training
   mean. Also the rank correlation of forecast and outcome (IC), and how often the sign is
   right.
2. More parameters. On QQQ alone, trees from 2 leaves and 50 rounds to 1,024 leaves and 2,000
   rounds: R^2 on the training years and on the test years.
3. More data. The same model trained on 2, 4 and all earlier years, on QQQ alone and on all 16
   ETFs pooled (about 16 times the rows): R^2 on 2022-2026, which every variant tests on.
4. How much data a real edge would need. With n independent half hours, an IC of c gives a
   t-statistic of about c * sqrt(n).

Usage: diagnosis.py DECISIONS_PARQUET
"""

import math
import sys

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "rule",
    "above_upper",
    "below_lower",
    "from_vwap",
    "from_open",
    "last_30",
    "gap",
    "time",
    "minutes_left",
    "volume_so_far",
    "vol_so_far",
    "sigma",
    "yesterday",
    "vol14",
]
TEST_YEARS = range(2018, 2027)

data = (
    pl.read_parquet(sys.argv[1])
    .filter(pl.col("vol14") > 0)
    .with_columns(
        year=pl.col("day").dt.year(),
        z=pl.col("ret") / pl.col("vol14"),
    )
    .drop_nulls(FEATURES + ["z"])
    .filter(pl.all_horizontal([pl.col(c).is_finite() for c in FEATURES + ["z"]]))
    .sort("ticker", "day", "t")
)


def trees(leaves=16, rounds=200):
    return HistGradientBoostingRegressor(
        max_leaf_nodes=leaves,
        max_iter=rounds,
        learning_rate=0.05,
        min_samples_leaf=20,
        early_stopping=False,
        random_state=0,
    )


class Linear:
    def __init__(self):
        self.scaler, self.model = StandardScaler(), Ridge(alpha=10.0)

    def fit(self, x, y):
        self.model.fit(self.scaler.fit_transform(x), y)
        return self

    def predict(self, x):
        return self.model.predict(self.scaler.transform(x))


def r2(y, forecast, benchmark):
    return 1 - np.sum((y - forecast) ** 2) / np.sum((y - benchmark) ** 2)


def rank_ic(y, forecast):
    a = np.argsort(np.argsort(y))
    b = np.argsort(np.argsort(forecast))
    return float(np.corrcoef(a, b)[0, 1])


def walk_forward(frame, target, make, first_train_year=None):
    """Forecasts for every test year from a model fitted on the years before it."""
    ys, fs, means = [], [], []
    for year in TEST_YEARS:
        train = frame.filter(pl.col("year") < year)
        if first_train_year is not None:
            train = train.filter(pl.col("year") >= first_train_year(year))
        test = frame.filter(pl.col("year") == year)
        if len(train) == 0 or len(test) == 0:
            continue
        y_train = train[target].to_numpy()
        model = make().fit(train.select(FEATURES).to_numpy(), y_train)
        ys.append(test[target].to_numpy())
        fs.append(model.predict(test.select(FEATURES).to_numpy()))
        means.append(np.full(len(test), y_train.mean()))
    return np.concatenate(ys), np.concatenate(fs), np.concatenate(means)


data = data.with_columns(size=pl.col("z").abs())
qqq = data.filter(pl.col("ticker") == "QQQ")

print("1. Direction against size, out of sample 2018-2026")
for name, frame in (("QQQ", qqq), ("16 ETFs pooled", data)):
    print(f"   {name}, {len(frame.filter(pl.col('year') >= 2018)):,} test half hours")
    for model_name, make in (("ridge", Linear), ("trees", trees)):
        y, f, _ = walk_forward(frame, "z", make)
        s, g, m = walk_forward(frame, "size", make)
        print(
            f"      {model_name:5}: direction R2 {r2(y, f, 0) * 100:+.2f}%, IC {rank_ic(y, f):+.3f}, "
            f"sign right {np.mean(np.sign(f) == np.sign(y)):.1%}  |  "
            f"size R2 {r2(s, g, m) * 100:+.1f}%, IC {rank_ic(s, g):+.3f}"
        )

print(
    "2. More parameters, QQQ: R2 of the direction on the training years / the test years"
)
for leaves, rounds in (
    (2, 50),
    (8, 200),
    (32, 500),
    (128, 1000),
    (512, 2000),
    (1024, 2000),
):
    in_sample, out_sample = [], []
    for year in TEST_YEARS:
        train, test = (
            qqq.filter(pl.col("year") < year),
            qqq.filter(pl.col("year") == year),
        )
        model = trees(leaves, rounds).fit(
            train.select(FEATURES).to_numpy(), train["z"].to_numpy()
        )
        in_sample.append(
            r2(
                train["z"].to_numpy(),
                model.predict(train.select(FEATURES).to_numpy()),
                0,
            )
        )
        out_sample.append(
            (test["z"].to_numpy(), model.predict(test.select(FEATURES).to_numpy()))
        )
    y = np.concatenate([a for a, _ in out_sample])
    f = np.concatenate([b for _, b in out_sample])
    print(
        f"   {leaves:>5} leaves x {rounds:>5} rounds: training {np.mean(in_sample) * 100:+6.2f}%, "
        f"test {r2(y, f, 0) * 100:+6.2f}%"
    )

print("3. More data: trees (16 leaves, 200 rounds), direction R2 on 2022-2026")
late = [y for y in TEST_YEARS if y >= 2022]
for name, frame in (("QQQ", qqq), ("16 ETFs pooled", data)):
    for label, first in (
        ("last 2 years", lambda year: year - 2),
        ("last 4 years", lambda year: year - 4),
        ("all years before", None),
    ):
        ys, fs = [], []
        for year in late:
            train = frame.filter(pl.col("year") < year)
            if first is not None:
                train = train.filter(pl.col("year") >= first(year))
            test = frame.filter((pl.col("year") == year) & (pl.col("ticker") == "QQQ"))
            model = trees().fit(
                train.select(FEATURES).to_numpy(), train["z"].to_numpy()
            )
            ys.append(test["z"].to_numpy())
            fs.append(model.predict(test.select(FEATURES).to_numpy()))
        y, f = np.concatenate(ys), np.concatenate(fs)
        rows = len(frame.filter(pl.col("year") < 2026))
        print(
            f"   trained on {name}, {label:16}: QQQ test R2 {r2(y, f, 0) * 100:+.2f}%, "
            f"IC {rank_ic(y, f):+.3f} (up to {rows:,} training rows)"
        )

n = len(qqq)
print("4. How much data an edge needs: t ~ IC x sqrt(n)")
print(f"   QQQ has {n:,} half hours; days {qqq['day'].n_unique():,}")
for ic in (0.01, 0.02, 0.05):
    print(
        f"   IC {ic}: t {ic * math.sqrt(n):.1f} on every half hour, "
        f"{ic * math.sqrt(qqq['day'].n_unique()):.1f} on one decision a day; "
        f"t = 2 needs {(2 / ic) ** 2:,.0f} independent bets"
    )
