"""N6's harness: what N6's model sees, the years it is trained and judged on, and the score.

Built on n6_data.py's tables. Nothing here learns. It prepares the arrays Efe's model reads,
splits them into years, and scores a year's forecasts with N4's variant 2's portfolio, so that
N6 and N4 are judged alike.

- The target, as N4's: a member-day's next return less the day's mean over the member-days
  with news and a next return, over its vol20 (the day's median where it has none).
- The member's context: each number less the day's mean over the day's spread, across the
  day's member-days with news, clipped at 5, missing as 0. The market's: less its mean over
  its spread on the training days only, clipped at 5, missing as 0. The scheduled events as 0
  or 1; the weekday and the sector (9 where none) as numbers for the model to embed.
- Headlines: a member-day's last MAX_HEADLINES by the time they became known, each a row of
  N6's eyes with log(1 + its age in hours), whether it counts from an edit, and whether it
  tags two stocks; -1 pads. A member-day with more keeps its count in full.
- A development year: validated on its sessions to LAST_DAY, by a model trained on every
  member-day from 2016 that ends GAP sessions before the year's first.
- Placebos: the day's headlines and counts dealt again among its member-days, each member's
  context and return staying with it; and the context alone, without any headline or count.
- The score: harness.n4_days with N4's variant 2's levers (a tenth a side, equal weights,
  ties in, 20 members at least, 2 bps a day).
"""

import sys
import types
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
import harness
from n6_context import CALENDAR, MARKET, MEMBER

N6 = Path("data/news_reader/n6")
MAX_HEADLINES = 32
GAP = 5  # sessions between the last training day and the first validated
CLIP = 5.0
DEVELOPMENT = (2018, 2019, 2020, 2021)
SECOND_LOOK = (2022, 2023, 2024, 2025, 2026)
LAST_DAY = harness.LAST_DAY
LEVERS = {"target": "scaled", "side": "tenth", "weights": "equal", "sectors": "none"}
NUMBERS = tuple(c for c in MEMBER if c != "sector")
EVENTS = tuple(c for c in CALENDAR if c != "weekday")
NO_SECTOR = 9


@dataclass(frozen=True)
class Arrays:
    """Every member-day, row for row with `days`."""

    days: pl.DataFrame  # day, ticker, next_return, vol: what the score needs
    day: np.ndarray  # int64, the day's place among all days
    member: np.ndarray  # float32 [n, len(NUMBERS)], scaled within each day
    market: np.ndarray  # float32 [n, len(MARKET)], as stored: scaled by market_scaled
    events: np.ndarray  # float32 [n, len(EVENTS)]
    weekday: np.ndarray  # int64
    sector: np.ndarray  # int64, NO_SECTOR where none
    y: np.ndarray  # float32, nan without a next return
    count: np.ndarray  # float32, log(1 + headlines)
    heads: np.ndarray  # int32 [n, MAX_HEADLINES], rows of headlines.npy, -1 pads
    head_info: (
        np.ndarray
    )  # float32 [n, MAX_HEADLINES, 3]: log(1 + age), edited, two tags


def target(days: pl.DataFrame) -> np.ndarray:
    """N4's scaled target; nan where there is no next return."""
    frame = days.with_columns(
        pl.col("vol20").fill_null(pl.col("vol20").median().over("day")).alias("v")
    ).with_columns(
        y=(pl.col("next_return") - pl.col("next_return").mean().over("day"))
        / pl.col("v")
    )
    return frame["y"].fill_null(np.nan).to_numpy().astype(np.float32)


def within_day(days: pl.DataFrame, columns) -> np.ndarray:
    """Each column less the day's mean over its spread across the day's rows, clipped,
    missing (and a day with no spread) as 0."""
    scaled = days.select(
        [
            ((pl.col(c) - pl.col(c).mean().over("day")) / pl.col(c).std().over("day"))
            .clip(-CLIP, CLIP)
            .fill_nan(None)
            .fill_null(0.0)
            .alias(c)
            for c in columns
        ]
    )
    return scaled.to_numpy().astype(np.float32)


def market_scaled(market: np.ndarray, day: np.ndarray, train: np.ndarray) -> np.ndarray:
    """The market's numbers less their mean over their spread on the training days, one
    value a day, clipped; missing as 0."""
    _, first = np.unique(day[train], return_index=True)
    once = market[train][first].astype(np.float64)
    mean, spread = np.nanmean(once, 0), np.nanstd(once, 0)
    spread[~(spread > 0)] = 1.0
    out = np.clip((market - mean) / spread, -CLIP, CLIP)
    return np.nan_to_num(out, nan=0.0).astype(np.float32)


def headlines(
    days: pl.DataFrame, pairs: pl.DataFrame, most: int = MAX_HEADLINES
) -> tuple[np.ndarray, np.ndarray]:
    """Each member-day's last `most` headlines, as rows of the store and their info."""
    start, stop = days["start"].to_numpy(), days["stop"].to_numpy()
    row = pairs["row"].to_numpy()
    info = np.stack(
        [
            np.log1p(pairs["age_hours"].to_numpy()),
            pairs["edited"].to_numpy().astype(np.float32),
            (pairs["tags"].to_numpy() == 2).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    heads = np.full((days.height, most), -1, dtype=np.int32)
    head_info = np.zeros((days.height, most, 3), dtype=np.float32)
    for i, (a, b) in enumerate(zip(start, stop, strict=True)):
        a = max(a, b - most)
        heads[i, : b - a] = row[a:b]
        head_info[i, : b - a] = info[a:b]
    return heads, head_info


def build(days: pl.DataFrame, pairs: pl.DataFrame, most: int = MAX_HEADLINES) -> Arrays:
    """The arrays for every member-day of `days` (n6_data's table, sorted by day)."""
    order = days["day"].unique().sort()
    day = (
        days.select(pl.col("day").replace_strict(order, pl.Series(range(len(order)))))
        .to_series()
        .to_numpy()
        .astype(np.int64)
    )
    heads, head_info = headlines(days, pairs, most)
    vol = days["vol20"].fill_null(days["vol20"].median())
    return Arrays(
        days=days.select("day", "ticker", "next_return").with_columns(
            vol=vol, sector=pl.lit(0)
        ),
        day=day,
        member=within_day(days, NUMBERS),
        market=days.select(MARKET).to_numpy().astype(np.float32),
        events=days.select(EVENTS).to_numpy().astype(np.float32),
        weekday=days["weekday"].to_numpy().astype(np.int64),
        sector=days["sector"].fill_null(NO_SECTOR).to_numpy().astype(np.int64),
        y=target(days),
        count=np.log1p(days["count"].to_numpy()).astype(np.float32),
        heads=heads,
        head_info=head_info,
    )


def load(n6: Path = N6, most: int = MAX_HEADLINES) -> Arrays:
    return build(
        pl.read_parquet(n6 / "days.parquet"),
        pl.read_parquet(n6 / "pairs.parquet"),
        most,
    )


def fold(a: Arrays, year: int, gap: int = GAP) -> tuple[np.ndarray, np.ndarray]:
    """The training and validated member-days of development year `year`."""
    when = a.days["day"].to_numpy()
    first = np.datetime64(date(year, 1, 1))
    validated = (
        (when >= first)
        & (when < np.datetime64(date(year + 1, 1, 1)))
        & (when <= np.datetime64(LAST_DAY))
        & ~np.isnan(a.y)
    )
    first_day = a.day[when >= first].min()
    trained = (a.day < first_day - gap) & ~np.isnan(a.y)
    return trained, validated


def deal(a: Arrays, seed: int) -> Arrays:
    """The placebo: each day's headlines and counts dealt again among its member-days."""
    rng = np.random.default_rng(seed)
    order = np.arange(len(a.day))
    bounds = np.flatnonzero(np.diff(a.day)) + 1
    for part in np.split(order, bounds):
        order[part] = rng.permutation(part)
    return replace(
        a, heads=a.heads[order], head_info=a.head_info[order], count=a.count[order]
    )


def silent(a: Arrays) -> Arrays:
    """The placebo without news: no headline and no count."""
    return replace(
        a,
        heads=np.full_like(a.heads, -1),
        head_info=np.zeros_like(a.head_info),
        count=np.zeros_like(a.count),
    )


def score(a: Arrays, rows: np.ndarray, forecasts: np.ndarray) -> list[dict]:
    """The days of N4's variant 2's portfolio on these member-days and forecasts."""
    test = a.days.filter(pl.Series(rows))
    return harness.n4_days(types.SimpleNamespace(test=test), forecasts, LEVERS)


def summary(days: list[dict]) -> dict:
    """Bps a day after costs, its t, and the days."""
    s = np.array([d["score"] for d in days])
    return {
        "bps": float(s.mean() * 1e4) if len(s) else 0.0,
        "t": harness.t_stat(s) if len(s) > 1 else 0.0,
        "days": len(s),
    }
