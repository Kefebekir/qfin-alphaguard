"""N4's volatilities and statistical sectors, from the daily closes (pre-registration N4).

- A member's volatility on a day: the standard deviation of its daily close-to-close returns
  over the 20 sessions ending the session before the day, so it is known before the day's
  trades. Null with fewer than 20 returns; N4 then takes the day's median.
- Sectors, one set a calendar year: the S&P 500 members on the year's first session with at
  least 200 daily returns in the 252 sessions before it, grouped into 11 by Ward's
  hierarchical clustering on the distance sqrt(2 (1 - correlation)) of those returns. Group 0
  holds a member without that history or new in the year. Nothing after the year's first
  session is used.

The closes are the stored ones, adjusted for splits and dividends, so a return is the
holder's.

Output: OUT/risk.parquet (day, ticker, vol, sector) for every member-day from 2016 on.
Usage: risk.py PRICES OUT
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

WINDOW = 20
HISTORY = 252
ENOUGH = 200
GROUPS = 11


def returns(prices: pl.DataFrame) -> pl.DataFrame:
    """(date, ticker, r, sp500): each day's close over the ticker's close before it, less one."""
    return prices.sort("ticker", "date").with_columns(
        (pl.col("close") / pl.col("close").shift(1).over("ticker") - 1).alias("r")
    )


def volatility(prices: pl.DataFrame) -> pl.DataFrame:
    """(day, ticker, vol): the 20 returns ending the session before each day."""
    r = returns(prices)
    return r.select(
        pl.col("date").alias("day"),
        "ticker",
        pl.col("r").rolling_std(WINDOW).shift(1).over("ticker").alias("vol"),
    )


def sectors(prices: pl.DataFrame, year: int) -> dict[str, int]:
    """Each member's group (1 to 11) for `year`, from the 252 sessions before its first."""
    days = prices["date"].unique().sort()
    first = days.filter(days.dt.year() == year).min()
    before = days.filter(days < first).tail(
        HISTORY + 1
    )  # one more, for the first return
    members = (
        prices.filter((pl.col("date") == first) & pl.col("sp500"))["ticker"]
        .unique()
        .sort()
    )
    window = returns(
        prices.filter(
            pl.col("ticker").is_in(members.to_list())
            & pl.col("date").is_in(before.to_list())
        )
    ).filter(pl.col("date") > before.min())
    wide = window.pivot(on="ticker", index="date", values="r").sort("date")
    names = [t for t in wide.columns[1:] if wide[t].drop_nulls().len() >= ENOUGH]
    matrix = np.ma.masked_invalid(wide.select(names).to_numpy().astype(float))
    corr = np.ma.corrcoef(matrix, rowvar=False).filled(0.0)
    distance = np.sqrt(np.clip(2.0 * (1.0 - corr), 0.0, None))
    np.fill_diagonal(distance, 0.0)
    tree = linkage(squareform(distance, checks=False), method="ward")
    labels = fcluster(tree, GROUPS, criterion="maxclust")
    return dict(zip(names, (int(x) for x in labels), strict=True))


def table(prices: pl.DataFrame, first_year: int = 2016) -> pl.DataFrame:
    """(day, ticker, vol, sector) for each member-day from `first_year` on."""
    vol = volatility(prices)
    members = prices.filter(pl.col("sp500") & (pl.col("date").dt.year() >= first_year))
    years = sorted(members["date"].dt.year().unique().to_list())
    groups = pl.DataFrame(
        [(y, t, g) for y in years for t, g in sectors(prices, y).items()],
        schema={"year": pl.Int32, "ticker": pl.String, "sector": pl.Int64},
        orient="row",
    )
    return (
        members.select(pl.col("date").alias("day"), "ticker")
        .join(vol, on=["day", "ticker"], how="left")
        .with_columns(pl.col("day").dt.year().cast(pl.Int32).alias("year"))
        .join(groups, on=["year", "ticker"], how="left")
        .with_columns(pl.col("sector").fill_null(0))
        .drop("year")
        .sort("day", "ticker")
    )


if __name__ == "__main__":
    PRICES, OUT = Path(sys.argv[1]), Path(sys.argv[2])
    OUT.mkdir(parents=True, exist_ok=True)
    prices = pl.read_parquet(PRICES, columns=["date", "ticker", "close", "sp500"])
    risk = table(prices)
    risk.write_parquet(OUT / "risk.parquet")
    print(
        f"{risk.height:,} member-days; volatility missing on {risk['vol'].null_count():,}"
    )
    sizes = risk.group_by(pl.col("day").dt.year().alias("year"), "sector").agg(
        pl.col("ticker").n_unique().alias("members")
    )
    for year, part in sizes.sort("year", "sector").group_by(
        "year", maintain_order=True
    ):
        print(year[0], dict(part.select("sector", "members").iter_rows()))
