"""1-minute bars for the trading universe, cut to the regular session.

For each year, the stocks of that year's trading universe (decision 0005) get
the year's 1-minute bars from EODHD: prices and volume as traded, and only the
minutes that start inside the regular session (sessions.py). One Parquet file
per stock and year, under data/raw/intraday/<ticker>/<year>.parquet.

EODHD files a renamed company's 1-minute history under one of its tickers,
not always the one in the daily data: FB_old's minutes are under META, and
DowDuPont's (DWDP) under DD. So a stock's own code is tried first, then the
codes that joined the index on a day it left (renamed to) and those that left
on a day it joined (renamed from). EODHD has no minutes under TICKER_old
codes at all. Every year of minutes is checked against the daily closes and
kept only if they match: a check on the data, and on the company.
"""

import re
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl

from qfin_alphaguard.data.eodhd import MINUTE_SCHEMA, EodhdClient, eodhd_code
from qfin_alphaguard.data.universe import IndexHistory
from qfin_alphaguard.events import Bar
from qfin_alphaguard.sessions import sessions

# A day matches when the last regular-session minute closes within this of the
# daily close as traded; the closing auction can differ a little from 15:59.
CLOSE_TOLERANCE = 0.01
# A code's minutes are taken for a stock when this share of the days match.
MATCHING_DAYS = 0.9

# The rules a Bar event enforces (events.py). A minute that breaks one is
# dropped when it is downloaded, rather than stopping a backtest halfway.
VALID_MINUTE = (
    (pl.min_horizontal("open", "high", "low", "close") > 0)
    & (pl.col("low") <= pl.min_horizontal("open", "close"))
    & (pl.max_horizontal("open", "close") <= pl.col("high"))
    & (pl.col("volume") >= 0)
)


@dataclass(frozen=True)
class YearOfMinutes:
    """What was stored for one stock and year."""

    ticker: str  # the daily code, such as FB_old
    year: int
    source: str | None  # where the minutes came from, such as META; None if nowhere
    rows: int
    sessions: int  # sessions in the year, up to the last day asked for
    matching: float  # share of days whose last minute matched the daily close
    error: str | None = None  # why the download failed, if it did
    dropped: int = 0  # minutes that broke the Bar rules


def intraday_path(root: Path, ticker: str, year: int) -> Path:
    return root / "intraday" / ticker / f"{year}.parquet"


def regular_session(minutes: pl.DataFrame) -> pl.DataFrame:
    """Keep the bars that start inside the regular session: open <= start < close."""
    if minutes.is_empty():
        return minutes
    found = sessions(minutes["start"].min().date(), minutes["start"].max().date())
    schedule = pl.DataFrame(
        {
            "day": [session.day for session in found],
            "session_open": [session.open for session in found],
            "session_close": [session.close for session in found],
        },
        schema={
            "day": pl.Date,
            "session_open": pl.Datetime("us", "UTC"),
            "session_close": pl.Datetime("us", "UTC"),
        },
    )
    return (
        minutes.with_columns(pl.col("start").dt.date().alias("day"))
        .join(schedule, on="day")
        .filter(
            (pl.col("start") >= pl.col("session_open"))
            & (pl.col("start") < pl.col("session_close"))
        )
        .drop("day", "session_open", "session_close")
    )


def matching_share(minutes: pl.DataFrame, daily: pl.DataFrame) -> float:
    """Share of days whose last minute closes near the daily close as traded."""
    last_minutes = (
        minutes.with_columns(pl.col("start").dt.date().alias("date"))
        .group_by("date")
        .agg(pl.col("close").sort_by("start").last().alias("minute_close"))
    )
    traded = daily.select(
        "date", (pl.col("close") / pl.col("adjustment")).alias("day_close")
    )
    both = last_minutes.join(traded, on="date")
    if both.is_empty():
        return 0.0
    near = (pl.col("minute_close") / pl.col("day_close") - 1).abs() <= CLOSE_TOLERANCE
    return both.select(near.mean()).item()


def minute_sources(code: str, history: IndexHistory) -> tuple[str, ...]:
    """EODHD codes that may hold `code`'s 1-minute bars, to try in order."""
    old = re.fullmatch(r"(.+)_old\d*", code)
    ticker = (old.group(1) if old else code).replace("-", ".")
    own = history.spans_of(ticker)
    left = {span.end for span in own if span.end is not None}
    joined = {span.start for span in own}
    renamed_to = sorted(
        {eodhd_code(s.ticker) for s in history.spans if s.start in left} - {code}
    )
    renamed_from = sorted(
        {eodhd_code(s.ticker) for s in history.spans if s.end in joined} - {code}
    )
    first = () if old else (code,)  # EODHD has no minutes under _old codes
    return tuple(dict.fromkeys((*first, *renamed_to, *renamed_from)))


def download_year(
    client: EodhdClient,
    code: str,
    year: int,
    last_day: date,
    daily: pl.DataFrame,
    history: IndexHistory,
) -> tuple[pl.DataFrame, YearOfMinutes]:
    """One stock's regular-session minutes for one year, checked against `daily`."""
    first, last = date(year, 1, 1), min(date(year, 12, 31), last_day)
    in_year = len(sessions(first, last))
    daily_bars = daily.filter(
        (pl.col("ticker") == code) & pl.col("date").is_between(first, last)
    )
    for source in minute_sources(code, history):
        minutes = regular_session(client.minute_bars(source, first, last))
        valid = minutes.filter(VALID_MINUTE)
        share = matching_share(valid, daily_bars)
        if share >= MATCHING_DAYS:
            stored = valid.with_columns(pl.lit(code).alias("ticker"))
            return stored, YearOfMinutes(
                code,
                year,
                source,
                stored.height,
                in_year,
                share,
                dropped=minutes.height - valid.height,
            )
    return pl.DataFrame(schema=MINUTE_SCHEMA), YearOfMinutes(
        code, year, None, 0, in_year, 0.0
    )


def backfill_minutes(
    client: EodhdClient,
    daily: pl.DataFrame,
    history: IndexHistory,
    universes: dict[int, Iterable[str]],
    root: Path,
    last_day: date,
    workers: int = 4,
    report: Callable[[YearOfMinutes], None] = lambda summary: None,
) -> list[YearOfMinutes]:
    """Store the minutes of every stock in `universes` ({year: tickers}).

    Years before `last_day`'s year that already have minutes are final and
    skipped, so an interrupted run picks up where it stopped; the current year
    is downloaded again. A year without matching minutes gets no file and is
    tried again next time, when its sources may be known. A download that fails
    is reported with its error and does not stop the others.
    """
    tasks = [
        (code, year)
        for year, codes in sorted(universes.items())
        for code in codes
        if not (year < last_day.year and _has_minutes(intraday_path(root, code, year)))
    ]

    def run(task: tuple[str, int]) -> YearOfMinutes:
        code, year = task
        try:
            minutes, summary = download_year(
                client, code, year, last_day, daily, history
            )
        except RuntimeError as error:  # EODHD still failing after its retries
            return YearOfMinutes(code, year, None, 0, 0, 0.0, error=str(error))
        if not minutes.is_empty():
            path = intraday_path(root, code, year)
            path.parent.mkdir(parents=True, exist_ok=True)
            minutes.write_parquet(path)
        return summary

    done = []
    # A few requests at a time: each is a few seconds and up to 10 MB, far
    # below EODHD's limit of 1,000 requests a minute.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for finished in as_completed(pool.submit(run, task) for task in tasks):
            summary = finished.result()
            report(summary)
            done.append(summary)
    return done


def minute_feed(
    root: Path, universes: dict[int, Iterable[str]], first: date, last: date
) -> Iterator[Bar]:
    """The stored minutes of each year's universe as Bar events, in time order.

    One year of files is read at a time, about 5 million minutes for 50
    stocks; turning all years into events at once would take gigabytes.
    Within a minute the bars come in ticker order, so the same files always
    give the same sequence. Prices are as traded: multiply by the day's
    `adjustment` from the daily bars to compare prices across a split.
    """
    for year in range(first.year, last.year + 1):
        paths = [
            intraday_path(root, ticker, year) for ticker in universes.get(year, ())
        ]
        paths = [path for path in paths if _has_minutes(path)]
        if not paths:
            continue
        minutes = (
            pl.scan_parquet(paths)
            .filter(pl.col("start").dt.date().is_between(first, last))
            .sort("start", "ticker")
            .collect()
        )
        for row in minutes.iter_rows(named=True):
            yield Bar(
                ticker=row["ticker"],
                start=row["start"],
                seconds=60,
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
            )


def _has_minutes(path: Path) -> bool:
    return path.exists() and pl.scan_parquet(path).select(pl.len()).collect().item() > 0
