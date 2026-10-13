"""Pilot for decision 0017: do price and volume features predict S&P 500 returns
out of sample, and does a long-only portfolio keep anything after costs?

Walk-forward by calendar year: each test year's models are fitted only on
earlier rows, purged of the last 1 + h sessions so that no training label
overlaps the test year. Fixed hyperparameters, chosen before looking at any
test result, so the only choices tried are the ones reported here.
"""

import sys
from pathlib import Path

import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

PANEL = Path(sys.argv[1])
OUT = Path(sys.argv[2])
H = int(sys.argv[3])
TEST_YEARS = range(2018, 2027)

panel = pl.read_parquet(PANEL).filter(pl.col(f"fwd_{H}").is_not_null())
features = [c for c in panel.columns if c.startswith("z_")]
label = f"fwd_{H}"
# The label each model learns: the day's cross-sectional rank of the return.
panel = panel.with_columns(
    (
        (pl.col(label).rank("average").over("date") - 1) / (pl.len().over("date") - 1)
        - 0.5
    ).alias("y")
)
dates = panel["date"].unique().sort()


def models():
    return {
        "ridge": Ridge(alpha=10.0),
        "gbt": HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=2000,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=0,
        ),
    }


predictions = []
for year in TEST_YEARS:
    test = panel.filter(pl.col("date").dt.year() == year)
    if test.is_empty():
        continue
    first_test = test["date"].min()
    earlier = dates.filter(dates < first_test)
    cutoff = earlier[-(H + 2)]  # purge: the last 1 + h sessions before the test year
    train = panel.filter(pl.col("date") <= cutoff)
    X, y = train.select(features).to_numpy(), train["y"].to_numpy()
    out = test.select("date", "ticker", label)
    for name, model in models().items():
        model.fit(X, y)
        out = out.with_columns(
            pl.Series(name, model.predict(test.select(features).to_numpy()))
        )
    # Single signals, signed as the literature uses them.
    out = out.with_columns(
        test["z_mom_12_1"].alias("momentum"),
        (-test["z_ret_5"]).alias("reversal"),
        (-test["z_vol_63"]).alias("low_vol"),
    )
    predictions.append(out)
    print(year, "trained on", train.height, "rows to", cutoff, flush=True)

pred = pl.concat(predictions)
pred = pred.with_columns(
    (
        (pl.col("ridge").rank().over("date") + pl.col("gbt").rank().over("date")) / 2
    ).alias("ensemble")
)
pred.write_parquet(OUT)
print("wrote", OUT, pred.height)
