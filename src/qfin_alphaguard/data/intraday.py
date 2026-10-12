"""1-minute bars for the trading universe and for ETFs, cut to the regular session.

For each year, the stocks of that year's trading universe (decision 0005) get
the year's 1-minute bars, and the stocks that left the universe at the new
year get January's, so that a backtest can sell them (minute_windows). Prices
and volume are as traded, and only the minutes that start inside the regular
session are kept (sessions.py). One Parquet file per stock and year, under
data/raw/intraday/<ticker>/<year>.parquet.

Two providers sell the consolidated tape, EODHD and Alpaca (alpaca.py), and
each lacks days, wholly or in part, that the other has (decision 0008). The
stocks' minutes come from EODHD first; an ETF's, such as QQQ's for decision
0011, from Alpaca first. The second provider is asked only for a year in
which the first lacks a minute of some trading day's session, and each day's
minutes then come whole from the one that has more of them (by_day). The
file's `source` column says which.

A provider may file a company's minutes under another of its tickers, such
as META for FB_old; each says which codes to try (minute_codes). Every year
of a provider's minutes is checked against the daily closes and kept only if
they match: a check on the data, and on the company.
"""

from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol

import polars as pl

from qfin_alphaguard.data.eodhd import MINUTE_SCHEMA
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

# A stored minute: a provider's bar, and which provider it came from.
STORED_SCHEMA = {**MINUTE_SCHEMA, "source": pl.String}


class MinuteClient(Protocol):
    """A provider of minutes: EodhdClient or AlpacaClient."""

    name: str  # stored in the source column, such as EODHD

    def minute_codes(self, code: str, history: IndexHistory) -> tuple[str, ...]: ...

    def minute_bars(self, code: str, first: date, last: date) -> pl.DataFrame: ...


@dataclass(frozen=True)
class YearOfMinutes:
    """What was stored for one stock and year."""

    ticker: str  # the daily code, such as FB_old
    year: int
    # (provider, its code, days taken from it) for each provider whose minutes
    # matched, such as ("EODHD", "META", 250); empty if none did
    parts: tuple[tuple[str, str, int], ...]
    rows: int
    trading_days: int  # days with a daily bar, from 1 January to the last day asked
    matching: float  # share of the days with minutes whose last minute matched
    error: str | None = None  # why the download failed, if it did
    dropped: int = 0  # minutes that broke the Bar rules
    days: int = 0  # trading days with minutes from either provider


def intraday_path(root: Path, ticker: str, year: int) -> Path:
    return root / "intraday" / ticker / f"{year}.parquet"


def minute_windows(universes: dict[int, Iterable[str]]) -> dict[int, dict[str, date]]:
    """For each year, the stocks whose minutes it needs, and until which day.

    A year's trading universe needs the whole year. A stock of the previous
    year's universe that is not in this one may still be held on the year's
    first day; the band sells it at the first open (decision 0006), and
    Guard's order limits let even a 25% position go within a week, so its
    minutes are needed until the end of January.
    """
    lists = {year: tuple(tickers) for year, tickers in universes.items()}
    windows = {}
    for year, tickers in lists.items():
        needed = dict.fromkeys(tickers, date(year, 12, 31))
        for ticker in lists.get(year - 1, ()):
            needed.setdefault(ticker, date(year, 1, 31))
        windows[year] = needed
    return windows


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


def by_day(*sources: pl.DataFrame) -> pl.DataFrame:
    """Each day's minutes from the source with the most of them, the first on a tie.

    On an ordinary day the providers have the same minutes, so a day one of
    them lacks, wholly or in part, comes from another. A day is never mixed:
    on most days the two give the first minute a different volume (Alpaca's
    is larger), and a day keeps to one provider's rules.
    """
    ranked = pl.concat(
        [
            frame.with_columns(
                pl.col("start").dt.date().alias("day"), pl.lit(rank).alias("rank")
            )
            for rank, frame in enumerate(sources)
        ]
    )
    best = (
        ranked.group_by("day", "rank")
        .len()
        .sort("day", "len", "rank", descending=[False, True, False])
        .unique("day", keep="first")
        .drop("len")
    )
    return ranked.join(best, on=["day", "rank"]).drop("day", "rank").sort("start")


def download_year(
    clients: Sequence[MinuteClient],
    code: str,
    year: int,
    last_day: date,
    daily: pl.DataFrame,
    history: IndexHistory,
) -> tuple[pl.DataFrame, YearOfMinutes]:
    """One stock's regular-session minutes for one year, checked against `daily`.

    The minutes run from 1 January to `last_day` or the end of the year,
    whichever comes first. The providers are asked in order, each under its
    codes until one's minutes match the daily closes, and only while some
    trading day still lacks a minute of its session. Each day's minutes come
    from the provider that has more of them (by_day).
    """
    first, last = date(year, 1, 1), min(date(year, 12, 31), last_day)
    daily_bars = daily.filter(
        (pl.col("ticker") == code) & pl.col("date").is_between(first, last)
    )
    stored = pl.DataFrame(schema=STORED_SCHEMA)
    codes = {}
    dropped = 0
    for client in clients:
        if _every_minute(stored, daily_bars):
            break
        for source in client.minute_codes(code, history):
            minutes = regular_session(client.minute_bars(source, first, last))
            valid = minutes.filter(VALID_MINUTE)
            if matching_share(valid, daily_bars) >= MATCHING_DAYS:
                codes[client.name] = source
                dropped += minutes.height - valid.height
                labelled = valid.with_columns(
                    pl.lit(code).alias("ticker"), pl.lit(client.name).alias("source")
                )
                stored = by_day(stored, labelled)
                break
    days_from = dict(
        stored.group_by("source").agg(pl.col("start").dt.date().n_unique()).iter_rows()
    )
    return stored, YearOfMinutes(
        code,
        year,
        tuple((name, source, days_from.get(name, 0)) for name, source in codes.items()),
        stored.height,
        daily_bars.height,
        matching_share(stored, daily_bars),
        dropped=dropped,
        days=stored["start"].dt.date().n_unique(),
    )


def backfill_minutes(
    clients: Sequence[MinuteClient],
    daily: pl.DataFrame,
    history: IndexHistory,
    universes: dict[int, Iterable[str]],
    root: Path,
    last_day: date,
    workers: int = 4,
    report: Callable[[YearOfMinutes], None] = lambda summary: None,
) -> list[YearOfMinutes]:
    """Store the minutes each year of `universes` ({year: tickers}) needs.

    minute_windows says which stocks and until which day, and download_year
    asks `clients` in order. Stored minutes that reach the stock's last daily
    bar in its window are complete and skipped, so an interrupted run picks up
    where it stopped, and a year stored while it was still running is
    downloaded again. Minutes missing inside a window do not count: neither
    provider has some, and a new download would not bring them. A year
    without matching minutes gets no file and is tried again next time, when
    its sources may be known. A download that fails, from either provider, is
    reported with its error, keeps the year's old file, and does not stop the
    others.
    """
    tasks = [
        (code, year, min(until, last_day))
        for year, needed in sorted(minute_windows(universes).items())
        for code, until in needed.items()
        if not _complete(root, daily, code, year, min(until, last_day))
    ]

    def run(task: tuple[str, int, date]) -> YearOfMinutes:
        code, year, until = task
        try:
            minutes, summary = download_year(clients, code, year, until, daily, history)
        except RuntimeError as error:  # a provider still failing after its retries
            return YearOfMinutes(code, year, (), 0, 0, 0.0, error=str(error))
        if not minutes.is_empty():
            path = intraday_path(root, code, year)
            path.parent.mkdir(parents=True, exist_ok=True)
            minutes.write_parquet(path)
        return summary

    done = []
    # A few requests at a time, each a second or more and up to 10 MB: below
    # EODHD's limit of 1,000 requests a minute, and about Alpaca's of 200.
    # Alpaca answers a faster pace with HTTP 429, which its client retries.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for finished in as_completed(pool.submit(run, task) for task in tasks):
            summary = finished.result()
            report(summary)
            done.append(summary)
    return done


def minute_feed(
    root: Path, universes: dict[int, Iterable[str]], first: date, last: date
) -> Iterator[Bar]:
    """The stored minutes each year needs as Bar events, in time order.

    Each year brings its universe's minutes, and those of the stocks that
    left it at the new year until the end of January (minute_windows). One
    year of files is read at a time, about 5 million minutes for 50 stocks;
    turning all years into events at once would take gigabytes. Within a
    minute the bars come in ticker order, so the same files always give the
    same sequence. Prices are as traded: multiply by the day's `adjustment`
    from the daily bars to compare prices across a split.
    """
    windows = minute_windows(universes)
    for year in range(first.year, last.year + 1):
        needed = {
            ticker: until
            for ticker, until in windows.get(year, {}).items()
            if _has_minutes(intraday_path(root, ticker, year))
        }
        if not needed:
            continue
        until = pl.LazyFrame(
            {"ticker": list(needed), "until": list(needed.values())},
            schema={"ticker": pl.String, "until": pl.Date},
        )
        # One file at a time: a file stored before the source column existed
        # has one column fewer.
        files = [
            pl.scan_parquet(intraday_path(root, ticker, year)).select(
                list(MINUTE_SCHEMA)
            )
            for ticker in needed
        ]
        minutes = (
            pl.concat(files)
            .filter(pl.col("start").dt.date().is_between(first, last))
            .join(until, on="ticker")
            .filter(pl.col("start").dt.date() <= pl.col("until"))
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


def _every_minute(minutes: pl.DataFrame, daily: pl.DataFrame) -> bool:
    """`minutes` has every minute of the session on each day of `daily`."""
    days = set(daily["date"].to_list())
    if not days:
        return True  # no trading day, so no minute is missing
    have = dict(minutes.group_by(pl.col("start").dt.date()).len().iter_rows())
    return all(
        have.get(session.day, 0) == session.minutes
        for session in sessions(min(days), max(days))
        if session.day in days
    )


def _complete(
    root: Path, daily: pl.DataFrame, code: str, year: int, last: date
) -> bool:
    """Stored minutes reach the stock's last daily bar from 1 January to `last`.

    Minutes stored before the second provider filled holes have no source
    column; they are downloaded again, once.
    """
    path = intraday_path(root, code, year)
    if not path.exists() or "source" not in pl.read_parquet_schema(path):
        return False
    stored = pl.scan_parquet(path).select(pl.col("start").max()).collect().item()
    bars = daily.filter(
        (pl.col("ticker") == code) & pl.col("date").is_between(date(year, 1, 1), last)
    )
    return (
        stored is not None
        and not bars.is_empty()
        and stored.date() >= bars["date"].max()
    )
