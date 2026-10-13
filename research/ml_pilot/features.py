"""Pilot for decision 0017: price and volume features of every S&P 500 member.

Every feature of day t uses only bars up to the close of t. The labels are
log returns from the close of t+1 to the close of t+1+h: the plan made after
the close of t trades during t+1, so the return of t+1 itself is not counted.
"""

import sys
from pathlib import Path

import polars as pl

PRICES = Path(sys.argv[1])
OUT = Path(sys.argv[2])
HORIZONS = (1, 5, 21)

daily = pl.read_parquet(PRICES)
spy = (
    daily.filter(pl.col("ticker") == "SPY")
    .sort("date")
    .select("date", pl.col("close").log().diff().alias("mkt"))
)
stocks = daily.filter(pl.col("ticker") != "SPY").sort("ticker", "date")

t = "ticker"
r = pl.col("r")
frame = (
    stocks.with_columns(
        pl.col("close").log().diff().over(t).alias("r"),
        (pl.col("open") / pl.col("close").shift(1).over(t)).log().alias("overnight"),
        (pl.col("close") / pl.col("open")).log().alias("intraday"),
        (pl.col("high") / pl.col("low")).log().alias("range"),
        (
            pl.col("close")
            / pl.col("adjustment")
            * pl.col("volume")
            / pl.col("split_factor")
        ).alias("dvol"),
    )
    .join(spy, on="date", how="left")
    .with_columns(
        # Returns over past windows, the latest day included.
        r.rolling_sum(1).over(t).alias("ret_1"),
        r.rolling_sum(5).over(t).alias("ret_5"),
        r.rolling_sum(21).over(t).alias("ret_21"),
        r.rolling_sum(63).over(t).alias("ret_63"),
        r.rolling_sum(126).over(t).alias("ret_126"),
        (r.rolling_sum(252).over(t) - r.rolling_sum(21).over(t)).alias("mom_12_1"),
        r.rolling_std(21).over(t).alias("vol_21"),
        r.rolling_std(63).over(t).alias("vol_63"),
        r.rolling_std(252).over(t).alias("vol_252"),
        r.rolling_max(21).over(t).alias("max_21"),
        r.rolling_skew(63).over(t).alias("skew_63"),
        pl.col("range").rolling_mean(21).over(t).alias("range_21"),
        pl.col("overnight").rolling_sum(21).over(t).alias("overnight_21"),
        pl.col("intraday").rolling_sum(21).over(t).alias("intraday_21"),
        pl.col("dvol").rolling_mean(21).over(t).log().alias("dvol_21"),
        (
            pl.col("dvol").rolling_mean(5).over(t)
            / pl.col("dvol").rolling_mean(63).over(t)
        )
        .log()
        .alias("dvol_shock"),
        (r.abs() / pl.col("dvol")).rolling_mean(21).over(t).log().alias("amihud_21"),
        (pl.col("close") / pl.col("close").rolling_max(252).over(t) - 1).alias(
            "from_high"
        ),
        pl.rolling_cov(r, pl.col("mkt"), window_size=252).over(t).alias("cov_252"),
        pl.col("mkt").rolling_var(252).over(t).alias("mvar_252"),
    )
    .with_columns((pl.col("cov_252") / pl.col("mvar_252")).alias("beta_252"))
    .with_columns(
        (r - pl.col("beta_252") * pl.col("mkt"))
        .rolling_std(63)
        .over(t)
        .alias("idio_63"),
        *[
            (
                pl.col("close").shift(-(1 + h)).over(t)
                / pl.col("close").shift(-1).over(t)
            )
            .log()
            .alias(f"fwd_{h}")
            for h in HORIZONS
        ],
    )
)
FEATURES = [
    "ret_1",
    "ret_5",
    "ret_21",
    "ret_63",
    "ret_126",
    "mom_12_1",
    "vol_21",
    "vol_63",
    "vol_252",
    "max_21",
    "skew_63",
    "range_21",
    "overnight_21",
    "intraday_21",
    "dvol_21",
    "dvol_shock",
    "amihud_21",
    "from_high",
    "beta_252",
    "idio_63",
]
panel = (
    frame.filter(pl.col("sp500"))
    .select("date", "ticker", "close", *FEATURES, *[f"fwd_{h}" for h in HORIZONS])
    .drop_nulls(FEATURES)
    .filter(pl.all_horizontal([pl.col(f).is_finite() for f in FEATURES]))
)
# Cross-sectional ranks in [-0.5, 0.5] each day: robust to outliers and to
# the level of a feature drifting over the years.
panel = panel.with_columns(
    [
        (
            (pl.col(f).rank("average").over("date") - 1) / (pl.len().over("date") - 1)
            - 0.5
        ).alias(f"z_{f}")
        for f in FEATURES
    ]
)
panel.write_parquet(OUT)
print(
    panel.height,
    "rows",
    panel["date"].min(),
    "..",
    panel["date"].max(),
    "stocks a day:",
    panel.group_by("date").len()["len"].median(),
)
