"""The news reader's harness: each test year's data, the policies, the scores, the holdout lock.

Pre-registrations N1 and N2 (and amendment 1) fix everything here; Efe's part is the model.
A model is anything with fit(X, y) and predict(X), as in scikit-learn; `make` builds a fresh
one for each test year.

- Test year Y reads the features of ChronoBERT vintage min(Y - 1, 2024) (features.py) and is
  fitted on the years from 2016 to Y - 1.
- Development: test years 2018-2021, run as often as Efe likes.
- Holdout: 2022 to 9 October 2026, run once. It needs the chosen variant committed in
  variants/<test>_chosen.json, with nothing uncommitted; it writes its result to
  data/news_reader/holdout_<test>.json and refuses to run again.

N1 (the QQQ rule's day): X is the mean embedding of the morning's headlines and log(1 + their
number); y is the rule's day return over QQQ's 14-day volatility. A test day's size is the
rule's times 1.5 if the forecast is above the top third of the training forecasts, 0.5 if
below the bottom third, 1 otherwise; the day's return scales with it. The score is the daily
difference between the sized rule at the rule's risk (the scale from the training years) and
the rule.

N2 (members' next day): X likewise for a member's headlines to 15:30; y is the return from
the day's close to the next session's close less the mean of the members with news that day.
Each test day with at least 20 such members: equal weights long the top tenth of forecasts,
short the bottom tenth, less 10 bps a round trip on each side.
"""

import json
import math
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from itertools import pairwise
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression

from qfin_alphaguard.sessions import sessions

EMB = Path("data/news_reader/emb")
RULE_DAYS = Path("data/news_reader/rule_days.parquet")
PRICES = Path("data/raw/prices.parquet")
VARIANTS = Path("research/news_reader/variants")
LOCKS = Path("data/news_reader")
FIRST_TRAIN = 2016
DEVELOPMENT = (2018, 2019, 2020, 2021)
HOLDOUT = (2022, 2023, 2024, 2025, 2026)
LAST_DAY = date(2026, 10, 9)
THRESHOLD = 2.24  # one-sided, Bonferroni over N1 and N2
COST = 0.0010  # a round trip, each side (N2)
MIN_MEMBERS = 20


def vintage_for(year: int) -> int:
    """The ChronoBERT vintage test year `year` may use (amendment 1: none after 2024)."""
    return min(year - 1, 2024)


def features(test: str, vintage: int) -> tuple[pl.DataFrame, np.ndarray]:
    """The rows (day, and ticker for N2) and X: mean embedding and log(1 + count)."""
    rows = pl.read_parquet(EMB / f"{test.lower()}_v{vintage}.parquet")
    embedding = np.load(EMB / f"{test.lower()}_v{vintage}.npy").astype(np.float32)
    x = np.hstack([embedding, np.log1p(rows["count"].to_numpy())[:, None]])
    return rows, x


@dataclass(frozen=True)
class Year:
    """One test year's data: training rows, test rows, and the target and its use."""

    year: int
    train_x: np.ndarray
    train_y: np.ndarray
    test_x: np.ndarray
    train: pl.DataFrame  # the training rows' days (and tickers), targets, returns
    test: pl.DataFrame


def _split(rows: pl.DataFrame, x: np.ndarray, year: int) -> Year:
    years = rows["day"].dt.year().to_numpy()
    train = (years >= FIRST_TRAIN) & (years < year)
    test = (years == year) & (rows["day"].to_numpy() <= np.datetime64(LAST_DAY))
    return Year(
        year,
        x[train],
        rows["y"].to_numpy()[train],
        x[test],
        rows.filter(pl.Series(train)),
        rows.filter(pl.Series(test)),
    )


def n1_year(year: int, count_only: bool = False) -> Year:
    rows, x = features("N1", vintage_for(year))
    rule = pl.read_parquet(RULE_DAYS)
    joined = rows.with_row_index("row").join(rule, on="day", how="inner").sort("day")
    x = x[joined["row"].to_numpy()]
    if count_only:
        x = x[:, -1:]
    joined = joined.with_columns(y=pl.col("rule") / pl.col("vol14"))
    return _split(joined, x, year)


def next_day_returns() -> pl.DataFrame:
    """Each member's return from a session's close to the next session's close."""
    days = [s.day for s in sessions(date(2015, 12, 1), LAST_DAY + timedelta(days=10))]
    following = dict(pairwise(days))
    prices = pl.read_parquet(PRICES, columns=["date", "ticker", "close"])
    nxt = prices.with_columns(
        pl.col("date").replace_strict(following, default=None).alias("next_day")
    )
    return nxt.join(
        prices.rename({"date": "next_day", "close": "next_close"}),
        on=["ticker", "next_day"],
        how="inner",
    ).select(
        pl.col("date").alias("day"),
        "ticker",
        (pl.col("next_close") / pl.col("close") - 1).alias("next_return"),
    )


def n2_year(year: int, returns: pl.DataFrame, count_only: bool = False) -> Year:
    rows, x = features("N2", vintage_for(year))
    joined = (
        rows.with_row_index("row")
        .join(returns, on=["ticker", "day"], how="inner")
        .with_columns(
            y=pl.col("next_return") - pl.col("next_return").mean().over("day")
        )
        .sort("day", "ticker")
    )
    x = x[joined["row"].to_numpy()]
    if count_only:
        x = x[:, -1:]
    return _split(joined, x, year)


def t_stat(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    return float(v.mean() / v.std(ddof=1) * math.sqrt(len(v)))


def n1_sizes(train_forecasts: np.ndarray, forecasts: np.ndarray) -> np.ndarray:
    low, high = np.quantile(train_forecasts, [1 / 3, 2 / 3])
    return np.where(forecasts > high, 1.5, np.where(forecasts < low, 0.5, 1.0))


def n1_score(
    data: Year, train_forecasts: np.ndarray, forecasts: np.ndarray
) -> np.ndarray:
    """The daily difference: the sized rule at the rule's risk, less the rule."""
    r_train, r_test = data.train["rule"].to_numpy(), data.test["rule"].to_numpy()
    sized_train = n1_sizes(train_forecasts, train_forecasts) * r_train
    scale = r_train.std(ddof=1) / sized_train.std(ddof=1)
    return scale * n1_sizes(train_forecasts, forecasts) * r_test - r_test


def n2_score(data: Year, forecasts: np.ndarray) -> np.ndarray:
    """Each test day's long-short return after costs, on days with enough members."""
    frame = data.test.select("day", "next_return").with_columns(
        forecast=pl.Series(forecasts)
    )
    daily = []
    for _, day in frame.group_by("day", maintain_order=True):
        if day.height < MIN_MEMBERS:
            continue
        ranked = day.sort("forecast")
        n = day.height // 10
        short = ranked["next_return"][:n].mean()
        long = ranked["next_return"][-n:].mean()
        daily.append(long - short - 2 * COST)
    return np.array(daily)


def run(
    test: str, make: Callable, split: str = "development", count_only: bool = False
) -> dict:
    """Fit a fresh model for each test year of `split` and score it.

    With count_only the features are log(1 + count) alone and the model a linear regression
    (the pre-registrations' comparison: does reading the headlines add to counting them?).
    """
    if count_only:
        make = LinearRegression
    if split not in ("development", "holdout"):
        raise ValueError(f"split must be development or holdout, got {split!r}")
    years = DEVELOPMENT if split == "development" else HOLDOUT
    returns = next_day_returns() if test == "N2" else None
    per_year, all_scores = {}, []
    for year in years:
        data = (
            n1_year(year, count_only)
            if test == "N1"
            else n2_year(year, returns, count_only)
        )
        model = make().fit(data.train_x, data.train_y)
        forecasts = model.predict(data.test_x)
        if test == "N1":
            scores = n1_score(data, model.predict(data.train_x), forecasts)
        else:
            scores = n2_score(data, forecasts)
        all_scores.append(scores)
        per_year[year] = {
            "days": len(scores),
            "mean_bps": float(scores.mean() * 1e4),
            "t": t_stat(scores),
        }
    scores = np.concatenate(all_scores)
    return {
        "test": test,
        "split": split,
        "count_only": count_only,
        "days": len(scores),
        "mean_bps": float(scores.mean() * 1e4),
        "annual": float(scores.mean() * 252),
        "t": t_stat(scores),
        "passes": bool(t_stat(scores) >= THRESHOLD),
        "years": per_year,
    }


def run_holdout(test: str, make: Callable) -> dict:
    """The holdout, once: the chosen variant must be committed, and no earlier run."""
    chosen = VARIANTS / f"{test}_chosen.json"
    lock = LOCKS / f"holdout_{test}.json"
    if lock.exists():
        raise RuntimeError(f"{test}'s holdout has been run: {lock}")
    if not chosen.exists():
        raise RuntimeError(f"write and commit {chosen} with the chosen variant first")
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", str(chosen), "research/news_reader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(chosen)],
        capture_output=True,
        check=False,
    ).returncode
    if status.strip() or tracked != 0:
        raise RuntimeError(
            "commit the chosen variant and the harness before the holdout"
        )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    result = run(test, make, "holdout") | {
        "count_only": run(test, make, "holdout", count_only=True),
        "commit": commit,
        "variant": json.loads(chosen.read_text()),
    }
    lock.write_text(json.dumps(result, indent=2))
    return result
