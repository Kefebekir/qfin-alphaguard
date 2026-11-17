"""KUTBIT's data set: every instrument's sessions, minute by minute, with what was known before
each (decision 0023). Nothing is predicted or chosen here.

The instruments of each year from FIRST_YEAR are:
- its trading universe: the 50 most liquid S&P 500 members, chosen on the evening before the
  year from the year before (data/universe.py), so the choice sees nothing of the year;
- the 16 ETFs of G1's gym.

For each instrument and session it keeps:
- the regular session's bars on a grid of one a minute, as stored (data/raw/intraday). A
  minute without a trade carries the last close, with no volume, and is marked unseen; one
  before the first trade carries that trade's open. A bar whose open or close lies outside its
  low and high is widened to them, as decision 0016 does for daily bars.
- from the LOOKBACK sessions before it, on the session's own basis, splits taken into account:
  - the previous close;
  - the volatility of daily close-to-close log returns;
  - the median volume at each minute;
  - the usual size of a move at each minute (`swing`);
  - its beta to QQQ, from half-hour returns;
  - what it did apart from QQQ over the 30 minutes after each minute, in the last
    HABIT_SESSIONS of them (`habit`).
  A stock's splits come from the daily prices; an ETF's are found where its price jumps
  overnight by a clean ratio.
- the headlines naming it (Benzinga, data/frontier/news_all), counted by the minute they were
  first published, and those between the previous close and the open. As in the news tests, a
  headline counts only if it names one or two symbols.
- the VIX's close on the day before (data/raw/cboe).

A session is `usable` when the LOOKBACK sessions before it, and the one just before, all have
minutes, QQQ's too, and the VIX is known. A stock that joins the universe in January is therefore usable
from its 22nd session.

Output, under data/kutbit/ (git-ignored):
- days.parquet: one row for each instrument's session: day, year, row, ticker, etf, minutes,
  prev_close, vol, beta, vix, news_before, split and usable;
- <year>.npz: that year's open, high, low, close, volume, profile, swing and habit as
  [rows, 390] float32, seen as bool and news as uint16. A half day's 210 minutes are followed by its last close.
- results_dataset.txt beside this file: what the data set holds, year by year.

Usage: uv run python research/kutbit/dataset.py [DATA]   (DATA, the data folder, is data/)
"""

import bisect
import math
import sys
from collections import defaultdict
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from labels import labels

from qfin_alphaguard.data.universe import trading_universes
from qfin_alphaguard.kutbit.features import (
    CHANNELS,
    FULL_SESSION,
    HABIT_MINUTES,
    HABIT_SESSIONS,
    LOOKBACK,
    Day,
    channels,
    inputs,
)
from qfin_alphaguard.sessions import Session, sessions

DATA = Path("data")
RAW = DATA / "raw"
NEWS = DATA / "frontier" / "news_all"
RESULTS = Path(__file__).parent / "results_dataset.txt"
FIRST_YEAR, LAST_YEAR = 2016, 2026
ETFS = (
    "DIA",
    "GLD",
    "HYG",
    "IWM",
    "QQQ",
    "SPY",
    "TLT",
    "XLB",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLU",
    "XLV",
    "XLY",
)
MARKET = "QQQ"
PRICE_FIELDS = ("open", "high", "low", "close")
MINUTE_FIELDS = (*PRICE_FIELDS, "volume", "seen", "profile", "news", "swing", "habit")
HALF_HOUR = 30  # minutes; the returns a beta is measured on
SMOOTH = 7  # minutes on each side over which the usual move is averaged
SWING_FLOOR = 0.25  # the least usual move, as a share of the session's median
SPLIT_JUMP = math.log(1.3)  # an ETF's overnight move this large is checked for a split
SPLIT_RATIOS = (2, 3, 4, 5, 10, 1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 10)


def on_grid(minutes: pl.DataFrame, session: Session) -> dict[str, np.ndarray] | None:
    """A session's bars, one a minute from its open, with `seen`; None if it has none."""
    n = session.minutes
    offset = (minutes["start"] - session.open).dt.total_seconds().to_numpy() // 60
    keep = (offset >= 0) & (offset < n)
    if not keep.any():
        return None
    index = offset[keep].astype(int)
    seen = np.zeros(n, dtype=bool)
    seen[index] = True
    grid = {}
    for field in (*PRICE_FIELDS, "volume"):
        grid[field] = np.full(n, np.nan)
        grid[field][index] = minutes[field].to_numpy().astype(float)[keep]
    # the last trade's close, or before the first trade that trade's open
    last = np.maximum.accumulate(np.where(seen, np.arange(n), -1))
    carried = np.where(
        last >= 0, grid["close"][np.maximum(last, 0)], grid["open"][index.min()]
    )
    for field in PRICE_FIELDS:
        grid[field] = np.where(seen, grid[field], carried)
    grid["volume"] = np.where(seen, grid["volume"], 0.0)
    grid["seen"] = seen
    return grid


def repaired(grid: dict[str, np.ndarray]) -> int:
    """Widen the bars whose open or close lies outside their low and high; how many were."""
    body_high = np.maximum(grid["open"], grid["close"])
    body_low = np.minimum(grid["open"], grid["close"])
    broken = (grid["high"] < body_high) | (grid["low"] > body_low)
    grid["high"] = np.maximum(grid["high"], body_high)
    grid["low"] = np.minimum(grid["low"], body_low)
    return int(broken.sum())


def etf_splits(
    days: list[date], opens: list[float], closes: list[float]
) -> tuple[dict[date, float], list[tuple[date, float]]]:
    """An ETF's split factor on each day, as the daily prices define it: how many of the last
    day's shares one share of that day became. Found where the open jumps from the previous
    close by close to a clean ratio. Also the splits found, (day, ratio)."""
    factor, out, found = 1.0, {}, []
    for i in range(len(days) - 1, -1, -1):
        out[days[i]] = factor
        if i == 0:
            break
        shares = closes[i - 1] / opens[i]  # new shares for one old share, if a split
        if abs(math.log(shares)) > SPLIT_JUMP:
            ratio = min(SPLIT_RATIOS, key=lambda r: abs(math.log(shares / r)))
            if abs(math.log(shares / ratio)) < 0.05:
                factor *= ratio
                found.append((days[i], ratio))
    return out, sorted(found)


def minute_moves(grid: dict[str, np.ndarray]) -> np.ndarray:
    """Each minute's log return, as features.moves reads it."""
    previous = np.concatenate([[grid["open"][0]], grid["close"][:-1]])
    return np.log(grid["close"] / previous)


def usual_moves(history: list[dict]) -> np.ndarray | None:
    """The usual size of a move at each minute of a full session: the root mean square of
    that minute's log returns over the sessions, averaged over SMOOTH minutes on each side.
    It is never below SWING_FLOOR of its median; None if nothing moved at all."""
    squares = np.full((len(history), FULL_SESSION), np.nan)
    for i, grid in enumerate(history):
        moves = minute_moves(grid)
        squares[i, : len(moves)] = moves**2
    mean = np.nanmean(squares, axis=0)
    width = 2 * SMOOTH + 1
    smooth = np.convolve(np.pad(mean, SMOOTH, mode="edge"), np.ones(width) / width)
    swing = np.sqrt(smooth[width - 1 : -width + 1])
    if not np.median(swing) > 0:
        return None
    return np.maximum(swing, SWING_FLOOR * np.median(swing))


def half_hours(grid: dict[str, np.ndarray]) -> np.ndarray:
    """The session's log return over each HALF_HOUR minutes, the first from the open."""
    ends = grid["close"][HALF_HOUR - 1 :: HALF_HOUR]
    return np.diff(np.log(np.concatenate([[grid["open"][0]], ends])))


def beta_to(history: list[dict], market: list[dict | None]) -> float | None:
    """The slope, through zero, of the sessions' half-hour returns on the market's; None if
    the market lacks one of the sessions or did not move."""
    if any(
        m is None or len(m["close"]) != len(h["close"])
        for h, m in zip(history, market, strict=True)
    ):
        return None
    own = np.concatenate([half_hours(h) for h in history])
    of_market = np.concatenate([half_hours(m) for m in market])
    squares = float(of_market @ of_market)
    return float(own @ of_market) / squares if squares > 0 else None


def habit_of(
    history: list[dict], market: list[dict | None], beta: float
) -> np.ndarray | None:
    """At each minute of a full session, the mean over the sessions of the log return apart
    from the market from that minute's close to HABIT_MINUTES later, or to the close if that
    comes first; None if the market lacks one of the sessions."""
    if any(
        m is None or len(m["close"]) != len(h["close"])
        for h, m in zip(history, market, strict=True)
    ):
        return None
    apart = np.full((len(history), FULL_SESSION), np.nan)
    for i, (h, m) in enumerate(zip(history, market, strict=True)):
        n = len(h["close"])
        later = np.minimum(np.arange(n) + HABIT_MINUTES, n - 1)
        own = np.log(h["close"][later] / h["close"])
        apart[i, :n] = own - beta * np.log(m["close"][later] / m["close"])
    seen = (~np.isnan(apart)).any(axis=0)
    return np.where(seen, np.nanmean(np.where(seen, apart, 0.0), axis=0), 0.0)


def lookback(
    history: list[dict | None], factors: list[float], target: float
) -> tuple[float, float, np.ndarray, np.ndarray] | None:
    """From the LOOKBACK + 1 sessions before a session: its previous close, the volatility of
    daily log returns, the median volume at each minute and the usual move at each minute, on
    its basis (`target` is its split factor); None unless every one of those sessions has
    minutes and prices moved in them."""
    if len(history) < LOOKBACK + 1 or any(h is None for h in history):
        return None
    closes = np.array(
        [h["close"][-1] / f * target for h, f in zip(history, factors, strict=True)]
    )
    vol = float(np.std(np.diff(np.log(closes)), ddof=1))
    volumes = np.full((LOOKBACK, FULL_SESSION), np.nan)
    for i, (h, f) in enumerate(zip(history[1:], factors[1:], strict=True)):
        volumes[i, : len(h["volume"])] = h["volume"] * f / target
    swing = usual_moves(history[1:])
    if swing is None:
        return None
    return float(closes[-1]), vol, np.nanmedian(volumes, axis=0), swing


def news_symbol(ticker: str) -> str:
    """The symbol headlines name a stored ticker by: an earlier company's without its _old."""
    return ticker.split("_old")[0]


def headline_minutes(symbols: set[str], folder: Path = NEWS) -> dict[str, np.ndarray]:
    """For each symbol, the minutes (since 1970, UTC) its headlines were first published in,
    sorted, counting only headlines that name one or two symbols."""
    frames = []
    for path in sorted(folder.glob("*.jsonl")):
        month = pl.read_ndjson(
            path, schema={"created_at": pl.String, "symbols": pl.List(pl.String)}
        )
        frames.append(
            month.filter(pl.col("symbols").list.len().is_between(1, 2))
            .explode("symbols", empty_as_null=False)
            .select(
                pl.col("symbols")
                .str.to_uppercase()
                .str.replace_all(".", "-", literal=True)
                .alias("symbol"),
                (
                    pl.col("created_at").str.to_datetime(time_zone="UTC").dt.epoch("s")
                    // 60
                ).alias("minute"),
            )
        )
    if not frames:
        return {}
    found = pl.concat(frames).filter(pl.col("symbol").is_in(list(symbols)))
    return {
        symbol: np.sort(part["minute"].to_numpy())
        for (symbol,), part in found.partition_by("symbol", as_dict=True).items()
    }


def minute_of(moment) -> int:
    return int(moment.timestamp() // 60)


def counted(
    times: np.ndarray, session: Session, previous: Session | None
) -> tuple[np.ndarray, int]:
    """Headlines in each minute of a session, and between the previous close and its open."""
    start = minute_of(session.open)
    inside = times[
        np.searchsorted(times, start) : np.searchsorted(times, start + session.minutes)
    ]
    per_minute = np.bincount(inside - start, minlength=session.minutes).astype(
        np.uint16
    )
    if previous is None:
        return per_minute, 0
    since = minute_of(previous.close)
    return per_minute, int(
        np.searchsorted(times, start) - np.searchsorted(times, since)
    )


def vix_closes(path: Path = RAW / "cboe" / "VIX_History.csv") -> tuple[list, list]:
    """The VIX's closes by day, oldest first."""
    rows = (
        pl.read_csv(path)
        .select(
            pl.col("DATE").str.to_date("%m/%d/%Y").alias("day"),
            pl.col("CLOSE").alias("close"),
        )
        .sort("day")
    )
    return rows["day"].to_list(), rows["close"].to_list()


def last_before(days: list, values: list, day: date):
    """The value of the last day before `day`, or None."""
    i = bisect.bisect_left(days, day)
    return values[i - 1] if i > 0 else None


def padded(array: np.ndarray, fill) -> np.ndarray:
    return np.concatenate([array, np.full(FULL_SESSION - len(array), fill)])


def build(raw: Path = RAW, news: Path = NEWS, years=None):
    """The rows of days.parquet, each year's arrays and notes on what was found."""
    years = list(years or range(FIRST_YEAR, LAST_YEAR + 1))
    prices = pl.read_parquet(raw / "prices_eodhd.parquet")
    universes = trading_universes(prices, years)
    calendar = list(sessions(date(years[0] - 1, 1, 1), date(years[-1], 12, 31)))
    stock_factor = {
        (t, d): f
        for t, d, f in prices.select("ticker", "date", "split_factor").iter_rows()
    }
    included: dict[str, set[int]] = defaultdict(set)
    for year in years:
        for ticker in (*universes[year], *ETFS):
            included[ticker].add(year)
    vix_days, vix_values = vix_closes(raw / "cboe" / "VIX_History.csv")
    headlines = headline_minutes({news_symbol(t) for t in included}, news)
    rows, arrays, notes = [], defaultdict(lambda: defaultdict(list)), []
    market_grids, _ = stored_grids(raw, MARKET, calendar)
    for ticker in sorted(included):
        grids, widened = stored_grids(raw, ticker, calendar)
        if not grids:
            continue
        days = [s.day for s in calendar if s.day in grids]
        etf = ticker in ETFS
        if etf:
            factor, splits = etf_splits(
                days,
                [grids[d]["open"][0] for d in days],
                [grids[d]["close"][-1] for d in days],
            )
            notes += [f"{ticker}: a {r:g}-for-1 split on {d}" for d, r in splits]
        else:
            factor = {d: stock_factor.get((ticker, d), 1.0) for d in days}
        if widened:
            notes.append(f"{ticker}: {widened} bars widened")
        times = headlines.get(news_symbol(ticker), np.array([], dtype=np.int64))
        for i, session in enumerate(calendar):
            if session.day.year not in included[ticker] or session.day not in grids:
                continue
            grid = grids[session.day]
            back = calendar[max(0, i - LOOKBACK - 1) : i]
            before = [grids.get(b.day) for b in back]
            known = lookback(
                before, [factor.get(b.day, 1.0) for b in back], factor[session.day]
            )
            beta = habit = None
            if known:
                of_market = [market_grids.get(b.day) for b in back]
                beta = beta_to(before[1:], of_market[1:])
            if beta is not None:
                habit = habit_of(
                    before[-HABIT_SESSIONS:], of_market[-HABIT_SESSIONS:], beta
                )
            per_minute, before_open = counted(
                times, session, calendar[i - 1] if i else None
            )
            vix = last_before(vix_days, vix_values, session.day)
            store = arrays[session.day.year]
            row = len(store["open"])
            last_close = grid["close"][-1]
            for field in PRICE_FIELDS:
                store[field].append(padded(grid[field], last_close).astype(np.float32))
            store["volume"].append(padded(grid["volume"], 0.0).astype(np.float32))
            store["seen"].append(padded(grid["seen"], False))
            store["news"].append(padded(per_minute, 0).astype(np.uint16))
            profile = known[2] if known else np.zeros(FULL_SESSION)
            store["profile"].append(profile.astype(np.float32))
            swing = known[3] if known else np.zeros(FULL_SESSION)
            store["swing"].append(swing.astype(np.float32))
            store["habit"].append(
                (habit if habit is not None else np.zeros(FULL_SESSION)).astype(
                    np.float32
                )
            )
            rows.append(
                {
                    "day": session.day,
                    "year": session.day.year,
                    "row": row,
                    "ticker": ticker,
                    "etf": etf,
                    "minutes": session.minutes,
                    "prev_close": known[0] if known else None,
                    "vol": known[1] if known else None,
                    "beta": beta,
                    "vix": vix,
                    "news_before": before_open,
                    "split": factor[session.day],
                    "usable": known is not None
                    and known[1] > 0
                    and habit is not None
                    and vix is not None,
                }
            )
    found = defaultdict(set)
    for row in rows:
        found[row["year"]].add(row["ticker"])
    for year in years:
        for ticker in (*universes[year], *ETFS):
            if ticker not in found[year]:
                notes.append(
                    f"{ticker}: in {year}'s instruments, but no minutes stored then"
                )
    stacked = {
        year: {field: np.stack(values) for field, values in fields.items()}
        for year, fields in arrays.items()
    }
    return pl.DataFrame(rows).sort("day", "ticker"), stacked, notes


def stored_grids(raw: Path, ticker: str, calendar: list[Session]) -> tuple[dict, int]:
    """An instrument's stored sessions on their grids, by day, and how many bars were widened."""
    paths = sorted((raw / "intraday" / ticker).glob("*.parquet"))
    if not paths:
        return {}, 0
    minutes = pl.concat([pl.read_parquet(p) for p in paths]).sort("start")
    by_day = minutes.with_columns(
        pl.col("start").dt.convert_time_zone("America/New_York").dt.date().alias("day")
    ).partition_by("day", as_dict=True)
    grids, widened = {}, 0
    for session in calendar:
        part = by_day.get((session.day,))
        grid = on_grid(part, session) if part is not None else None
        if grid is not None:
            widened += repaired(grid)
            grids[session.day] = grid
    return grids, widened


def day_of(days: pl.DataFrame, arrays: dict, ticker: str, day: date) -> Day:
    """One stored session as features.Day."""
    row = days.filter((pl.col("ticker") == ticker) & (pl.col("day") == day)).row(
        0, named=True
    )
    return day_from(row, arrays)


def day_from(row: dict, arrays: dict) -> Day:
    """The session of a row of days.parquet, as features.Day."""
    a, n = arrays[row["year"]], row["minutes"]
    i = row["row"]
    return Day(
        **{f: a[f][i, :n].astype(float) for f in (*PRICE_FIELDS, "volume")},
        seen=a["seen"][i, :n],
        profile=a["profile"][i, :n].astype(float),
        news=a["news"][i, :n].astype(float),
        swing=a["swing"][i, :n].astype(float),
        habit=a["habit"][i, :n].astype(float),
        beta=row["beta"],
        minutes=n,
        prev_close=row["prev_close"],
        vol=row["vol"],
        vix=row["vix"],
        news_before=row["news_before"],
        etf=row["etf"],
    )


def upto(day: Day, t: int) -> Day:
    """The session as it stood at the close of bar t, as the live engine will see it."""
    return replace(day, **{f: getattr(day, f)[: t + 1] for f in MINUTE_FIELDS})


def report(
    days: pl.DataFrame, arrays: dict, notes: list[str], checks: int = 200
) -> str:
    """What the data set holds, year by year, and a check that each minute's inputs use only
    the bars up to it."""
    lines = ["dataset.py: KUTBIT's data set (decision 0023)", ""]
    by_year = (
        days.group_by("year")
        .agg(
            pl.col("ticker").filter(~pl.col("etf")).n_unique().alias("stocks"),
            pl.col("ticker").n_unique().alias("instruments"),
            pl.len().alias("sessions"),
            pl.col("usable").mean().alias("usable"),
            (pl.col("news_before") > 0).mean().alias("news_before"),
        )
        .sort("year")
    )
    lines.append(
        "year  stocks  instruments  sessions  usable  minutes seen  news before the open"
    )
    for year, stocks, instruments, count, usable, before_open in by_year.iter_rows():
        a = arrays[year]
        part = days.filter(pl.col("year") == year)
        real = np.arange(FULL_SESSION) < part.sort("row")["minutes"].to_numpy()[:, None]
        seen = a["seen"][real].mean()
        lines.append(
            f"{year}  {stocks:>6}  {instruments:>11}  {count:>8,}  {usable:>6.1%}  "
            f"{seen:>12.2%}  {before_open:>20.1%}"
        )
    # labels, on usable sessions: how far a 30-minute position moves
    sample = days.filter(pl.col("usable")).sample(
        min(checks, days["usable"].sum()), seed=1
    )
    moves, worst, same, early, midday = [], [], 0, [], []
    move = CHANNELS.index("move")
    rng = np.random.default_rng(1)
    for row in sample.iter_rows(named=True):
        day = day_of(days, arrays, row["ticker"], row["day"])
        market = day_of(days, arrays, MARKET, row["day"])
        found = labels(day.open, day.high, day.low, day.close)[30]
        moves.append(np.nanmean(np.abs(found["ret"])))
        worst.append(np.nanmean(found["worst"]))
        t = int(rng.integers(0, row["minutes"]))
        if row["minutes"] == FULL_SESSION:
            size = np.abs(channels(day, market)[:, move])
            early.append(size[:5].mean())
            midday.append(size[150:210].mean())
        full = inputs(day, market)
        part = inputs(upto(day, t), upto(market, t))
        same += int(
            np.array_equal(full[0][t], part[0][t])
            and np.array_equal(full[1][t], part[1][t])
        )
    lines += [
        "",
        (
            f"On {len(sample)} usable sessions drawn at random, a 30-minute long position "
            f"moved {np.mean(moves) * 1e4:.1f} bps on average, up or down, and its worst "
            f"moment along the way was {np.mean(worst) * 1e4:.1f} bps."
        ),
        (
            "In the inputs, a move of the first five minutes has the size "
            f"{np.mean(early):.2f} on average and one between 12:00 and 13:00 "
            f"{np.mean(midday):.2f}, each in its minute's usual moves."
        ),
        (
            "The inputs at a minute drawn at random in each, from the session cut at that "
            f"minute, equal those from the whole session on {same} of {len(sample)}."
        ),
        "",
        "Notes:",
        *(f"- {n}" for n in notes),
    ]
    return "\n".join(lines) + "\n"


def main(data: Path = DATA) -> None:
    days, arrays, notes = build(data / "raw", data / "frontier" / "news_all")
    out = data / "kutbit"
    out.mkdir(parents=True, exist_ok=True)
    days.write_parquet(out / "days.parquet")
    for year, fields in arrays.items():
        np.savez_compressed(out / f"{year}.npz", **fields)
    text = report(days, arrays, notes)
    RESULTS.write_text(text)
    print(text)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DATA)
