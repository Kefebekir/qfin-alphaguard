"""What a round trip costs in KUTBIT's instruments: the quoted spread, measured.

A marketable order buys at the ask and sells at the bid, so a round trip at market pays one
bid-ask spread, half on the way in and half on the way out. That holds before any price
improvement, and only while the order fits within the size shown at the best price. The skill
test charges 2 bps a round trip (decision 0024) and the first agent's gym 5 bps (decision
0026); neither was measured. Here the spread is measured on the consolidated tape's quotes,
the NBBO of Alpaca's SIP feed, which is free back to 2016. Only 2016-2021 are read:

- `moments`: SESSIONS_A_MONTH full sessions of every month, drawn with SEED, and in each the
  MOMENTS, every 10 minutes from 09:31 and at 15:59. At each moment, for every instrument of
  that day in data/kutbit/days.parquet, usable for the features or not: the spread of the
  quotes in the WINDOW seconds from it, each weighted by how long it stood, in bps of the mid;
  and the first quote's prices and sizes.
- `events`: the first agent's extreme minutes (data/kutbit/agent/real), up to PER_KIND of each
  kind, drawn with SEED. For each, the spread where it enters, at the next minute's open, and
  where a HOLD-minute hold leaves, or a minute before the close if that comes first.
- `page`: the costs by instrument, by part of the day and by year; at the extreme minutes,
  each kind weighed by its number of events in the gym, against the gym's 5 bps, and when only
  those whose spread at entry is at most each of NARROW are kept; and how often an order of
  each size in ORDERS fits within the size shown at the best price, on both sides, by part of
  the day and at the extreme minutes' entries.
- `trial`: the first month's sessions and TRIAL_EVENTS events, to see the speed and the page.

A quote counts when its bid is above zero and below its ask. Locked and crossed quotes are
counted and left out, and the time they stand is left out of the weights. An instrument
without a valid quote in WINDOW seconds is asked for again over WIDER seconds. Alpaca gives
these years' sizes in round lots of LOT shares; a size in shares would be a multiple of LOT,
and the page checks how many are. The size shown is that of one market at the best price, so
the depth at that price is at least that.

    uv run --env-file .env python research/kutbit/costs.py trial [DATA]
    uv run --env-file .env python research/kutbit/costs.py moments [DATA]
    uv run --env-file .env python research/kutbit/costs.py events [DATA]
    uv run python research/kutbit/costs.py page [DATA]

Each session's moments and each year's events go to their own file under DATA/kutbit/costs as
they are measured, and a file already written is not asked for again, so a stopped download
goes on where it stopped. WORKERS requests are in flight at once, at most RATE a minute, under
Alpaca's 200. The live engine shares that limit, so a download starts and goes on only while
the market is quiet, from QUIET after a session's close to QUIET before the next one's open,
and stops before the open, to go on after the close.
"""

import html
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from functools import partial
from pathlib import Path

import gym
import numpy as np
import polars as pl
from skill import COST as SKILL_BPS

from qfin_alphaguard.data.alpaca import api_keys
from qfin_alphaguard.sessions import Session, sessions

DATA = Path("data")
YEARS = gym.YEARS  # 2016-2021; 2022-2026 stay unread
SESSIONS_A_MONTH = 2
SEED = 27
MOMENTS = (*range(1, 390, 10), 389)  # minutes after the open of a full session
WINDOW = 2  # seconds of quotes at a moment
WIDER = 30  # seconds, for an instrument without a valid quote in WINDOW
PER_KIND = 400  # extreme minutes of each kind
TRIAL_EVENTS = 20
HOLD = 30  # minutes: the first agent's longest hold, the one it nearly always chose
LOT = 100  # shares in a round lot
ORDERS = (30_000, 120_000)  # USD: 1x and 4x of a 30,000 USD account
PARTS = (
    ("the first 30 minutes", 0, 30),
    ("the day between", 30, 360),
    ("the last 30 minutes", 360, 390),
)
GYM_BPS = round(gym.COST * 1e4, 6)  # the first agent's gym: a stock's round trip
OPEN_KINDS = frozenset(
    k for k, name in enumerate(gym.KINDS) if name.startswith("at the open")
)
NARROW = (2, 3, 5, 10)  # bps: the widest spread at entry an order would go out at
# the skill test's strongest 1% at 15-60 minutes, before costs (0024's result)
EDGE_BPS = 1.89
URL = "https://data.alpaca.markets/v2/stocks/quotes"
WORKERS = 4  # requests in flight at once
RATE = 180  # requests a minute at most, under Alpaca's 200
A_REQUEST = 100  # tickers asked for in one request
# the live engine's tasks run from 09:15 to 16:30 New York time, around the session
QUIET = timedelta(minutes=45)
# the tickers an instrument's quotes are under before it changed its ticker
EARLIER = {"BKNG": ("PCLN",), "AABA": ("YHOO",)}
COLUMNS = {
    "spread": pl.Float64,
    "bid": pl.Float64,
    "ask": pl.Float64,
    "bid_size": pl.Int64,
    "ask_size": pl.Int64,
    "quotes": pl.Int64,
    "invalid": pl.Int64,
    "symbol": pl.String,
    "window": pl.Int64,
}
EMPTY = {name: None for name in COLUMNS} | {"quotes": 0, "invalid": 0}
MOMENT_SCHEMA = {"day": pl.Date, "minute": pl.Int64, "ticker": pl.String, **COLUMNS}
SIDES = ("enter", "leave")
EVENT_SCHEMA = {
    "day": pl.Date,
    "minute": pl.Int64,
    "ticker": pl.String,
    "kind": pl.Int64,
    **{f"{side}_{name}": kind for side in SIDES for name, kind in COLUMNS.items()},
}


# --- one moment's quotes ----------------------------------------------------------------------


def symbols(code: str) -> tuple[str, ...]:
    """Alpaca's tickers whose quotes may be the instrument `code`'s, to try in order: the code
    without its _old suffix and with a dot for a class share, then any ticker it had before."""
    return (re.sub(r"_old\d*$", "", code).replace("-", "."), *EARLIER.get(code, ()))


def spread_bps(bid, ask):
    """The bid-ask spread in bps of the mid."""
    return (ask - bid) / ((ask + bid) / 2) * 1e4


def moment(when: datetime) -> np.datetime64:
    """A UTC time as numpy's, to the nanosecond."""
    return np.datetime64(when.astimezone(UTC).replace(tzinfo=None), "ns")


def measure(quotes: list[dict], start: datetime, seconds: float) -> dict | None:
    """One instrument's quotes from `start` for `seconds`. The valid quotes' spread, each
    weighted by how long it stood before the next quote or the window's end; the first valid
    quote's prices and sizes; and how many quotes there were, and how many were locked or
    crossed. None if no quote in the window is valid."""
    begin = moment(start)
    end = begin + np.timedelta64(int(seconds * 1e9), "ns")
    at = np.array(
        [np.datetime64(q["t"].rstrip("Z"), "ns") for q in quotes], "datetime64[ns]"
    )
    order = np.argsort(at, kind="stable")
    order = order[(at[order] >= begin) & (at[order] < end)]
    if not len(order):
        return None
    at = at[order]
    bid = np.array([quotes[i]["bp"] for i in order], dtype=float)
    ask = np.array([quotes[i]["ap"] for i in order], dtype=float)
    stood = (np.append(at[1:], end) - at).astype(np.int64).astype(float)
    valid = (bid > 0) & (ask > bid)
    if not valid.any():
        return None
    spreads, weights = spread_bps(bid[valid], ask[valid]), stood[valid]
    mean = (
        (spreads * weights).sum() / weights.sum()
        if weights.sum() > 0
        else spreads.mean()
    )
    first = quotes[order[np.flatnonzero(valid)[0]]]
    return {
        "spread": float(mean),
        "bid": float(first["bp"]),
        "ask": float(first["ap"]),
        "bid_size": int(first["bs"]),
        "ask_size": int(first["as"]),
        "quotes": len(order),
        "invalid": int((~valid).sum()),
    }


def quotes_at(
    tickers: list[str], start: datetime, seconds: float, get: Callable[[dict], dict]
) -> dict[str, list[dict]]:
    """Every quote of `tickers` from `start` for `seconds`, over all of Alpaca's pages."""
    found = defaultdict(list)
    for first in range(0, len(tickers), A_REQUEST):
        params = {
            "symbols": ",".join(tickers[first : first + A_REQUEST]),
            "start": start.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": (start + timedelta(seconds=seconds))
            .astimezone(UTC)
            .strftime("%Y-%m-%dT%H:%M:%SZ"),
            "feed": "sip",
            "limit": "10000",
            "sort": "asc",
        }
        while True:
            page = get(params)
            for ticker, quotes in (page.get("quotes") or {}).items():
                found[ticker] += quotes
            if not page.get("next_page_token"):
                break
            params = {**params, "page_token": page["next_page_token"]}
    return dict(found)


def at_moment(codes: list[str], start: datetime, get: Callable[[dict], dict]) -> dict:
    """Each instrument's measure at `start`, from WINDOW seconds of quotes, or from WIDER for
    one without a valid quote in WINDOW, under the first of its tickers that has one."""
    found, missing = {}, list(codes)
    for seconds in (WINDOW, WIDER):
        if not missing:
            break
        quotes = quotes_at(
            sorted({t for c in missing for t in symbols(c)}), start, seconds, get
        )
        for code in list(missing):
            for ticker in symbols(code):
                measured = measure(quotes.get(ticker, []), start, seconds)
                if measured is not None:
                    found[code] = measured | {"symbol": ticker, "window": seconds}
                    missing.remove(code)
                    break
    return found


def alpaca(keys: tuple[str, str], rate: float = RATE) -> Callable[[dict], dict]:
    """A page of Alpaca's quotes for a request's parameters, asked again while Alpaca is busy
    or unreachable. However many threads ask, the requests start at most `rate` a minute."""
    headers = {"APCA-API-KEY-ID": keys[0], "APCA-API-SECRET-KEY": keys[1]}
    lock, turn = threading.Lock(), [time.monotonic()]

    def wait_turn() -> None:
        with lock:
            now = time.monotonic()
            mine = max(now, turn[0])
            turn[0] = mine + 60 / rate
        time.sleep(max(0.0, mine - now))

    def get(params: dict) -> dict:
        request = urllib.request.Request(
            f"{URL}?{urllib.parse.urlencode(params)}", headers=headers
        )
        for attempt in range(6):
            wait_turn()
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    return json.load(response)
            except urllib.error.HTTPError as problem:
                if problem.code != 429 and problem.code < 500:
                    raise RuntimeError(
                        f"Alpaca refused the quotes from {params['start']}: "
                        f"HTTP {problem.code}"
                    ) from None
            except OSError:
                pass
            time.sleep(2 ** (attempt + 1))
        raise RuntimeError(f"Alpaca's quotes from {params['start']} failed six times")

    return get


class Counted:
    """A `get` that counts its requests, from any thread."""

    def __init__(self, get: Callable[[dict], dict]):
        self.get, self.requests, self.lock = get, 0, threading.Lock()

    def __call__(self, params: dict) -> dict:
        with self.lock:
            self.requests += 1
        return self.get(params)


def market_is_quiet(now: datetime | None = None) -> bool:
    """Whether the live engine can spare Alpaca's requests: no session is open now, nor
    within QUIET of its open or its close."""
    now = now or datetime.now(UTC)
    near = sessions(now.date() - timedelta(days=1), now.date() + timedelta(days=1))
    return not any(s.open - QUIET <= now <= s.close + QUIET for s in near)


# --- what is measured, and when ------------------------------------------------------------------


def drawn_sessions(
    years=YEARS, a_month: int = SESSIONS_A_MONTH, seed: int = SEED
) -> list:
    """`a_month` full sessions of every month of `years`, drawn with `seed`, oldest first."""
    if max(years) > YEARS[-1]:
        raise ValueError("the costs read 2016-2021 at most; 2022-2026 stay unread")
    months = defaultdict(list)
    for s in sessions(date(min(years), 1, 1), date(max(years), 12, 31)):
        if s.minutes == 390 and s.day.year in years:
            months[s.day.year, s.day.month].append(s)
    rng = np.random.default_rng(seed)
    chosen = []
    for key in sorted(months):
        picked = rng.choice(len(months[key]), size=a_month, replace=False)
        chosen += [months[key][i] for i in sorted(picked)]
    return chosen


def universe(data: Path) -> dict[date, list[str]]:
    """Each day's instruments in data/kutbit/days.parquet. A session that is not usable for
    the features, for want of the sessions before it, still has its quotes and its cost."""
    days = pl.read_parquet(data / "kutbit" / "days.parquet", columns=["day", "ticker"])
    days = days.filter(pl.col("day").dt.year() <= YEARS[-1])
    return {
        day: sorted(codes)
        for day, codes in days.group_by("day").agg("ticker").iter_rows()
    }


def drawn_events(
    data: Path, per_kind: int = PER_KIND, seed: int = SEED
) -> pl.DataFrame:
    """Up to `per_kind` of the first agent's extreme minutes of each kind, drawn with `seed`,
    oldest first."""
    folder = data / "kutbit" / "agent" / "real"
    events = gym.load_events(folder, YEARS)
    tickers = json.loads((folder / "tickers.json").read_text())
    rng = np.random.default_rng(seed)
    keep = []
    for kind in range(len(gym.KINDS)):
        of_kind = np.flatnonzero(events["kind"] == kind)
        if len(of_kind) > per_kind:
            of_kind = rng.choice(of_kind, size=per_kind, replace=False)
        keep.append(of_kind)
    keep = np.sort(np.concatenate(keep))
    return pl.DataFrame(
        {
            "day": [date.fromordinal(int(d)) for d in events["day"][keep]],
            "minute": events["minute"][keep].astype(np.int64),
            "ticker": [tickers[int(t)] for t in events["ticker"][keep]],
            "kind": events["kind"][keep].astype(np.int64),
        },
        schema={
            "day": pl.Date,
            "minute": pl.Int64,
            "ticker": pl.String,
            "kind": pl.Int64,
        },
    )


def kinds_in_gym(data: Path) -> dict[int, int]:
    """How many of the first agent's extreme minutes of each kind the gym holds over YEARS.
    At most PER_KIND of a kind are measured, so the kinds are weighed by these."""
    kinds = gym.load_events(data / "kutbit" / "agent" / "real", YEARS)["kind"]
    return {int(k): int(n) for k, n in zip(*np.unique(kinds, return_counts=True))}


def entry_and_exit(session: Session, minute: int) -> tuple[datetime, datetime]:
    """Where the first agent enters at an extreme minute, at the next minute's open, and where
    a HOLD-minute hold leaves: at its last minute's close, or a minute before the session's
    close if that comes first."""
    enter = session.open + timedelta(minutes=minute + 1)
    last = session.close - timedelta(minutes=1)
    return enter, max(enter, min(enter + timedelta(minutes=HOLD), last))


# --- the downloads ------------------------------------------------------------------------------


def write(rows: list[dict], schema: dict, path: Path) -> None:
    """Rows to a Parquet file, whole or not at all."""
    part = path.with_suffix(".part")
    pl.DataFrame(rows, schema=schema).write_parquet(part)
    part.rename(path)


STOPPED = "stopped before the market's open; run it again after the close to go on"


def download_moments(
    data: Path,
    chosen: list,
    get,
    folder: str = "moments",
    quiet: Callable[[], bool] | None = None,
    workers: int = WORKERS,
    log=print,
) -> Path:
    """Each chosen session's moments, one file a session, the moments asked for `workers` at
    a time. A session already written is skipped, and the download stops before a session
    once `quiet` says the market is not."""
    out = data / "kutbit" / "costs" / folder
    out.mkdir(parents=True, exist_ok=True)
    codes_of = universe(data)
    with ThreadPoolExecutor(workers) as pool:
        for s in chosen:
            path = out / f"{s.day.isoformat()}.parquet"
            if path.exists():
                continue
            if quiet is not None and not quiet():
                log(STOPPED)
                break
            started, codes = time.time(), codes_of.get(s.day, [])
            starts = [s.open + timedelta(minutes=m) for m in MOMENTS] if codes else []
            found = pool.map(partial(at_moment, codes, get=get), starts)
            rows = [
                {"day": s.day, "minute": minute, "ticker": c, **at.get(c, EMPTY)}
                for minute, at in zip(MOMENTS, found)
                for c in codes
            ]
            write(rows, MOMENT_SCHEMA, path)
            log(f"{s.day}: {len(codes)} instruments ({time.time() - started:.0f} s)")
    return out


def entry_and_exit_measured(event: dict, session: Session, get) -> dict:
    """An event with its entry's and its exit's measures."""
    row = dict(event)
    for side, at in zip(SIDES, entry_and_exit(session, event["minute"])):
        found = at_moment([event["ticker"]], at, get).get(event["ticker"], EMPTY)
        row |= {f"{side}_{name}": value for name, value in found.items()}
    return row


def download_events(
    data: Path,
    events: pl.DataFrame,
    get,
    folder: str = "events",
    quiet: Callable[[], bool] | None = None,
    workers: int = WORKERS,
    log=print,
) -> Path:
    """Each event's entry and exit, one file a year, the events asked for `workers` at a
    time. A year already written is skipped, and the download stops before a year once
    `quiet` says the market is not."""
    out = data / "kutbit" / "costs" / folder
    out.mkdir(parents=True, exist_ok=True)
    of_day = {s.day: s for s in sessions(date(YEARS[0], 1, 1), date(YEARS[-1], 12, 31))}
    with ThreadPoolExecutor(workers) as pool:
        for year in sorted({d.year for d in events["day"]}):
            path = out / f"{year}.parquet"
            if path.exists():
                continue
            if quiet is not None and not quiet():
                log(STOPPED)
                break
            started = time.time()
            of_year = events.filter(pl.col("day").dt.year() == year).iter_rows(
                named=True
            )
            rows = list(
                pool.map(
                    lambda e: entry_and_exit_measured(e, of_day[e["day"]], get), of_year
                )
            )
            write(rows, EVENT_SCHEMA, path)
            log(f"{year}: {len(rows)} events ({time.time() - started:.0f} s)")
    return out


# --- what it shows --------------------------------------------------------------------------------


def load(folder: Path, schema: dict) -> pl.DataFrame:
    """Every file a download wrote to `folder`, as one table."""
    files = sorted(folder.glob("*.parquet")) if folder.exists() else []
    frames = [pl.read_parquet(f) for f in files]
    return pl.concat(frames) if frames else pl.DataFrame(schema=schema)


def part_of_day() -> pl.Expr:
    """The part of the day of a row's minute after the open."""
    expression = pl.lit(PARTS[-1][0])
    for name, _, end in reversed(PARTS[:-1]):
        expression = (
            pl.when(pl.col("minute") < end).then(pl.lit(name)).otherwise(expression)
        )
    return expression


def valid_moments(moments: pl.DataFrame, etfs: set[str]) -> pl.DataFrame:
    """The moments with a valid quote, with the mid, the part of the day, whether the
    instrument is an ETF, and the dollars shown at the best price on the thinner side."""
    return moments.filter(pl.col("spread").is_not_null()).with_columns(
        ((pl.col("bid") + pl.col("ask")) / 2).alias("mid"),
        part_of_day().alias("part"),
        pl.col("ticker")
        .is_in(pl.Series(sorted(etfs), dtype=pl.String).implode())
        .alias("etf"),
        pl.min_horizontal(
            pl.col("bid_size") * LOT * pl.col("bid"),
            pl.col("ask_size") * LOT * pl.col("ask"),
        ).alias("shown"),
    )


def weighted_mean(frame: pl.DataFrame, column: str) -> float | None:
    """The mean of `column`, each row weighing its `weight`, nulls left out."""
    kept = frame.filter(pl.col(column).is_not_null())
    total = kept["weight"].sum()
    return float((kept[column] * kept["weight"]).sum() / total) if total else None


def weighted_median(frame: pl.DataFrame, column: str) -> float | None:
    """The smallest value of `column` with at least half the weight at or below it."""
    kept = frame.filter(pl.col(column).is_not_null()).sort(column)
    if kept.is_empty():
        return None
    reached = kept["weight"].cum_sum() >= kept["weight"].sum() / 2
    return float(kept[column].filter(reached)[0])


def weighed(trips: pl.DataFrame) -> dict:
    """The round trips of `trips`, each event weighing its kind's events in the gym over the
    events of that kind measured."""
    above = trips.with_columns(
        (pl.col("trip") > GYM_BPS).cast(pl.Float64).alias("above")
    )
    return {
        "events": trips.height,
        "in the gym": float(trips["weight"].sum()),
        "enter": weighted_mean(trips, "enter_spread"),
        "leave": weighted_mean(trips, "leave_spread"),
        "mean": weighted_mean(trips, "trip"),
        "median": weighted_median(trips, "trip"),
        "above": weighted_mean(above, "above"),
        "usual": weighted_mean(trips, "usual"),
    }


def summary(
    moments: pl.DataFrame, events: pl.DataFrame, etfs: set[str], in_gym: dict[int, int]
) -> dict:
    """The costs by instrument, by part of the day, by year and at the extreme minutes, whose
    kinds weigh as many as `in_gym` holds of each; the orders that fit; and the checks."""
    rows = valid_moments(moments, etfs)
    by_instrument = (
        rows.group_by("ticker")
        .agg(
            pl.col("etf").first(),
            pl.len().alias("moments"),
            pl.col("mid").median().alias("price"),
            pl.col("spread").mean().alias("mean"),
            pl.col("spread").median().alias("median"),
            *[
                pl.col("spread").filter(pl.col("part") == name).mean().alias(name)
                for name, _, _ in PARTS
            ],
            *[
                (pl.col("shown") >= order).mean().alias(f"fits {order}")
                for order in ORDERS
            ],
        )
        .sort("mean", "ticker")
    )
    by_part = (
        rows.group_by("etf", "part")
        .agg(
            pl.col("spread").mean().alias("mean"),
            pl.col("spread").median().alias("median"),
        )
        .sort("etf", "part")
    )
    by_year = (
        rows.with_columns(pl.col("day").dt.year().alias("year"))
        .group_by("etf", "year")
        .agg(
            pl.col("spread").mean().alias("mean"),
            pl.col("spread").median().alias("median"),
        )
        .sort("etf", "year")
    )
    fit = [(pl.col("shown") >= order).mean().alias(f"fits {order}") for order in ORDERS]
    fits = rows.group_by("etf").agg(*fit)
    fits_by_part = rows.group_by("etf", "part").agg(*fit)
    trips = events.filter(
        pl.col("enter_spread").is_not_null() & pl.col("leave_spread").is_not_null()
    ).with_columns(
        ((pl.col("enter_spread") + pl.col("leave_spread")) / 2).alias("trip"),
        pl.min_horizontal(
            pl.col("enter_bid_size") * LOT * pl.col("enter_bid"),
            pl.col("enter_ask_size") * LOT * pl.col("enter_ask"),
        ).alias("shown"),
    )
    usual = by_instrument.select("ticker", pl.col("mean").alias("usual"))
    trips = trips.join(usual, on="ticker", how="left")
    measured = dict(trips.group_by("kind").len().iter_rows())
    if missing := sorted(set(measured) - set(in_gym)):
        raise ValueError(f"the gym holds no events of kinds {missing}")
    trips = trips.with_columns(
        pl.col("kind")
        .replace_strict(
            {k: in_gym[k] / n for k, n in measured.items()}, return_dtype=pl.Float64
        )
        .alias("weight")
    )
    by_kind = (
        trips.group_by("kind")
        .agg(
            pl.len().alias("events"),
            pl.col("enter_spread").mean().alias("enter"),
            pl.col("leave_spread").mean().alias("leave"),
            pl.col("trip").mean().alias("mean"),
            pl.col("trip").median().alias("median"),
            (pl.col("trip") > GYM_BPS).mean().alias("above"),
            pl.col("usual").mean().alias("usual"),
        )
        .with_columns(
            pl.col("kind")
            .replace_strict(in_gym, return_dtype=pl.Int64)
            .alias("in the gym")
        )
        .sort("kind")
    )
    at_open = pl.col("kind").is_in(
        pl.Series(sorted(OPEN_KINDS), dtype=pl.Int64).implode()
    )
    by_time = {
        "at the open": weighed(trips.filter(at_open)),
        "later in the day": weighed(trips.filter(~at_open)),
        "every kind": weighed(trips),
    }
    narrow = []
    for most in NARROW if trips.height else ():
        kept = trips.filter(pl.col("enter_spread") <= most)
        share_kept = kept["weight"].sum() / trips["weight"].sum()
        narrow.append({"at most": most, "kept": share_kept} | weighed(kept))
    fits_at_entry = {
        order: weighted_mean(
            trips.with_columns(
                (pl.col("shown") >= order).cast(pl.Float64).alias("fits")
            ),
            "fits",
        )
        for order in ORDERS
    }
    priced = rows.filter(pl.col("mid") >= 1)
    sizes = pl.concat([rows["bid_size"], rows["ask_size"]])
    return {
        "moments": moments.height,
        "rows": rows,
        "by_instrument": by_instrument,
        "by_part": by_part,
        "by_year": by_year,
        "fits": fits,
        "fits_by_part": fits_by_part,
        "fits_at_entry": fits_at_entry,
        "events": trips,
        "by_kind": by_kind,
        "by_time": by_time,
        "narrow": narrow,
        "checks": {
            "with a valid quote": rows.height / max(moments.height, 1),
            "found only in the wider window": float(
                (rows["window"] == WIDER).mean() or 0
            ),
            "quotes locked or crossed": float(
                moments["invalid"].sum() / max(moments["quotes"].sum(), 1)
            ),
            "spreads under a cent above 1 USD": float(
                ((priced["ask"] - priced["bid"]) < 0.01 - 1e-9).mean() or 0
            ),
            "sizes that are a multiple of LOT": float(((sizes % LOT) == 0).mean() or 0),
            "event trips measured": trips.height / max(events.height, 1),
        },
    }


STYLE = """:root { color-scheme: light; --surface: #fcfcfb; --grid: #e4e3df; --text: #0b0b0b;
  --muted: #77766f; --secondary: #52514e; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark;
  --surface: #1a1a19; --grid: #33332f; --text: #f0efec; --muted: #9a998f; --secondary: #c3c2b7; } }
:root[data-theme="dark"] { color-scheme: dark; --surface: #1a1a19; --grid: #33332f;
  --text: #f0efec; --muted: #9a998f; --secondary: #c3c2b7; }
body { margin: 0; background: var(--surface); color: var(--text);
  font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1100px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 15px; margin: 28px 0 8px; }
.tables { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: left; padding: 4px 8px; border-bottom: 1px solid var(--grid); }
th { color: var(--secondary); font-weight: 600; }
.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.note { color: var(--muted); font-size: 12px; }
.verdict { font-size: 16px; margin: 12px 0; }"""


def table_html(head: list[str], rows: list[list[str]], numeric_from: int = 1) -> str:
    """A table whose columns from `numeric_from` on hold numbers."""

    def cell(tag: str, i: int, text: str) -> str:
        return f"<{tag}{' class=num' if i >= numeric_from else ''}>{html.escape(text)}</{tag}>"

    top = "".join(cell("th", i, h) for i, h in enumerate(head))
    body = "".join(
        "<tr>" + "".join(cell("td", i, c) for i, c in enumerate(row)) + "</tr>"
        for row in rows
    )
    return f"<div class=tables><table><tr>{top}</tr>{body}</table></div>"


def bps(value) -> str:
    return "-" if value is None else f"{value:.2f}"


def share(value) -> str:
    return "-" if value is None else f"{value:.0%}"


def page(found: dict, title: str) -> str:
    """The page of what a round trip costs."""
    stocks = found["by_instrument"].filter(~pl.col("etf"))
    etfs = found["by_instrument"].filter(pl.col("etf"))
    sections = [f"<h1>{html.escape(title)}</h1>"]
    sections.append(
        "<p class=note>The quoted spread of the NBBO in bps of the mid: what a round trip at "
        "market pays, half in and half out, before any price improvement and while the order "
        "fits within the size shown. Each moment's spread is weighted over the quotes of "
        f"{WINDOW} seconds. The skill test charges {SKILL_BPS:g} bps, the first agent's gym "
        f"{GYM_BPS:g} bps, and the skill test's strongest 1% earn {EDGE_BPS:g} bps a signal "
        "at 15-60 minutes before costs.</p>"
    )
    if found["by_instrument"].height:
        under = {
            limit: int((found["by_instrument"]["mean"] < limit).sum())
            for limit in (EDGE_BPS, SKILL_BPS, GYM_BPS)
        }
        sections.append(
            f"<p class=verdict>Over {found['rows'].height:,} measured moments, the median "
            f"stock's round trip costs {bps(stocks['mean'].median())} bps on average and the "
            f"median ETF's {bps(etfs['mean'].median())} bps. Of "
            f"{found['by_instrument'].height} instruments, {under[EDGE_BPS]} cost less than "
            f"{EDGE_BPS:g} bps, {under[SKILL_BPS]} less than {SKILL_BPS:g} bps and "
            f"{under[GYM_BPS]} less than {GYM_BPS:g} bps.</p>"
        )
    if found["events"].height:
        times = found["by_time"]
        sections.append(
            "<p class=verdict>At the first agent's extreme minutes, each kind weighed by its "
            f"events in the gym, a round trip costs {bps(times['every kind']['mean'])} bps "
            f"on average: {bps(times['at the open']['mean'])} at the open and "
            f"{bps(times['later in the day']['mean'])} later in the day, against the "
            f"gym's {GYM_BPS:g} bps.</p>"
        )
    head = ["instrument", "price", "mean", "median", *(name for name, _, _ in PARTS)]
    head += [*(f"{order:,} USD fits" for order in ORDERS), "moments"]
    rows = [
        [
            r["ticker"] + (" (ETF)" if r["etf"] else ""),
            f"{r['price']:,.2f}",
            bps(r["mean"]),
            bps(r["median"]),
            *(bps(r[name]) for name, _, _ in PARTS),
            *(share(r[f"fits {order}"]) for order in ORDERS),
            f"{r['moments']:,}",
        ]
        for r in found["by_instrument"].iter_rows(named=True)
    ]
    sections += [
        "<h2>A round trip by instrument, in bps, cheapest first</h2>",
        table_html(head, rows),
    ]
    parts = {(r["etf"], r["part"]): r for r in found["by_part"].iter_rows(named=True)}
    rows = [
        [
            ("ETFs" if etf else "stocks") + ", " + name,
            bps(parts[etf, name]["mean"]),
            bps(parts[etf, name]["median"]),
        ]
        for etf in (False, True)
        for name, _, _ in PARTS
        if (etf, name) in parts
    ]
    sections += [
        "<h2>By part of the day</h2>",
        table_html(["", "mean", "median"], rows),
    ]
    rows = [
        [
            ("ETFs" if r["etf"] else "stocks") + f", {r['year']}",
            bps(r["mean"]),
            bps(r["median"]),
        ]
        for r in found["by_year"].iter_rows(named=True)
    ]
    sections += ["<h2>By year</h2>", table_html(["", "mean", "median"], rows)]

    def trip_row(name: str, r: dict) -> list[str]:
        return [name, f"{round(r['in the gym']):,}", f"{r['events']:,}", bps(r["enter"]),
                bps(r["leave"]), bps(r["mean"]), bps(r["median"]), share(r["above"]),
                bps(r["usual"])]  # fmt: skip

    rows = [
        trip_row(gym.KINDS[r["kind"]], r)
        for r in found["by_kind"].iter_rows(named=True)
    ]
    if rows:
        rows += [trip_row(f"{name}, at the gym's mix", r) for name, r in
                 found["by_time"].items()]  # fmt: skip
    head = ["kind of event", "in the gym", "measured", "the spread at entry", "at exit",
            "a round trip", "its median", f"above {GYM_BPS:g} bps",
            "the same instruments at ordinary moments"]  # fmt: skip
    sections += [
        (
            "<h2>At the first agent's extreme minutes: half the spread in, half out, "
            f"{HOLD} minutes later</h2>"
        ),
        (
            f"<p class=note>At most {PER_KIND} events of a kind are measured. A row of "
            "several kinds weighs each kind by its events in the gym, "
            f"{YEARS[0]}-{YEARS[-1]}.</p>"
        ),
        table_html(head, rows),
    ]
    rows = [
        [f"at most {r['at most']:g} bps", share(r["kept"]), bps(r["mean"]),
         bps(r["leave"])]
        for r in found["narrow"]
    ]  # fmt: skip
    sections += [
        "<h2>If an order goes out only while the spread at entry is narrow</h2>",
        (
            "<p class=note>The extreme minutes whose spread at entry is at most each "
            "width, at the gym's mix. An order sees that spread as it goes out, but for the "
            f"{WINDOW} seconds it is weighed over; what the events left out would have "
            "earned is not measured here.</p>"
        ),
        table_html(
            [
                "the spread at entry",
                "events kept",
                "a round trip",
                "the spread at exit",
            ],
            rows,
        ),
    ]
    by_part = {
        (r["etf"], r["part"]): r for r in found["fits_by_part"].iter_rows(named=True)
    }
    rows = []
    for r in found["fits"].sort("etf").iter_rows(named=True):
        kind = "ETFs" if r["etf"] else "stocks"
        rows.append([kind, *(share(r[f"fits {order}"]) for order in ORDERS)])
        rows += [
            [f"{kind}, {name}", *(share(part[f"fits {order}"]) for order in ORDERS)]
            for name, _, _ in PARTS
            if (part := by_part.get((r["etf"], name)))
        ]
    if found["events"].height:
        rows.append(
            ["the first agent's extreme minutes, at entry",
             *(share(found["fits_at_entry"][order]) for order in ORDERS)]
        )  # fmt: skip
    sections += [
        "<h2>How often an order fits within the size shown at the best price</h2>",
        table_html(["on both sides", *(f"{order:,} USD" for order in ORDERS)], rows),
    ]
    checks = "".join(
        f"<li>{html.escape(name)}: {value:.2%}</li>"
        for name, value in found["checks"].items()
    )
    sections += ["<h2>Checks</h2>", f"<ul>{checks}</ul>"]
    return (
        "<!doctype html>\n<html lang=en><head><meta charset=utf-8>"
        '<meta name=viewport content="width=device-width, initial-scale=1">'
        f"<title>KUTBIT costs</title><style>{STYLE}</style></head><body><main>"
        + "".join(sections)
        + "</main></body></html>\n"
    )


def write_page(
    data: Path, folder: str = "", title: str = "What a round trip costs"
) -> Path:
    """The page from what the downloads under DATA/kutbit/costs/`folder` wrote."""
    base = data / "kutbit" / "costs" / folder
    etfs = set(
        pl.read_parquet(data / "kutbit" / "days.parquet", columns=["ticker", "etf"])
        .filter("etf")["ticker"]
        .unique()
    )
    events = load(base / "events", EVENT_SCHEMA)
    in_gym = kinds_in_gym(data) if events.height else {}
    found = summary(load(base / "moments", MOMENT_SCHEMA), events, etfs, in_gym)
    path = base / "costs.html"
    path.write_text(page(found, title))
    return path


# --- the commands ---------------------------------------------------------------------------------


def main(argv: list[str]) -> None:
    if not argv or argv[0] not in ("trial", "moments", "events", "page"):
        raise SystemExit("usage: costs.py trial|moments|events|page [DATA]")
    data = Path(argv[1]) if len(argv) > 1 else DATA
    if argv[0] == "page":
        print(f"page: {write_page(data)}")
        return
    if not market_is_quiet():
        raise SystemExit(
            "the market is open or opens within 45 minutes, and the live engine needs "
            "Alpaca's requests; run it after the close"
        )
    get = Counted(alpaca(api_keys()))
    started = time.time()
    if argv[0] == "moments":
        download_moments(data, drawn_sessions(), get, quiet=market_is_quiet)
    elif argv[0] == "events":
        download_events(data, drawn_events(data), get, quiet=market_is_quiet)
    else:
        chosen = drawn_sessions()[:SESSIONS_A_MONTH]
        download_moments(
            data, chosen, get, folder="trial/moments", quiet=market_is_quiet
        )
        asked = get.requests
        events = drawn_events(data).head(TRIAL_EVENTS)
        download_events(data, events, get, folder="trial/events", quiet=market_is_quiet)
        seconds = time.time() - started
        print(
            f"trial: {asked} requests for {len(chosen) * len(MOMENTS)} moments and "
            f"{get.requests - asked} for {events.height} events in {seconds:.0f} s; "
            f"{write_page(data, 'trial', 'What a round trip costs: the trial')}"
        )
        return
    print(f"{argv[0]}: {get.requests} requests in {time.time() - started:.0f} s")


if __name__ == "__main__":
    main(sys.argv[1:])
