"""What follows an extreme minute: the event study before KUTBIT's first agent.

Efe's idea (18 November 2026): at minute resolution, when a stock makes an extreme move, an
agent learns whether to follow it, bet against it or let it be, how large, and how much of the
market part to hedge. Before any agent learns, this study measures what followed such minutes
on 2016-2018, the years an agent learns on. 2019, where its settings are chosen, and
2020-2021, where it is measured once, are not read; nor are 2022-2026.

- **An extreme minute:** a stock's move apart from QQQ in one minute, over its usual move at
  that minute (the `apart` channel, features.channels), of at least SIZES[0], in a minute that
  traded after one that traded too, with no such minute of the same stock in the COOLDOWN
  minutes before. The session's last bar is left out, since nothing is filled after it.
- **What followed,** from the next bar's open, as the labels are, for each horizon whose bars
  all come before the close: the move apart from QQQ in bps and in usual moves (z), signed by
  the extreme minute's direction, so that a positive mean is a continuation and a negative one
  a reversal; the stock's own move, signed the same way; and the worst move against a position
  that followed. The bounce, from the extreme minute's close to the next bar's open, is
  measured apart: no order placed at the close can catch it.
- **By context:** the move's size; a headline naming the stock in the NEWS_MINUTES up to it;
  the minute's volume against its usual; QQQ's own move in that minute; the part of the day;
  up or down.
- Means come with a t clustered by day, since one day's events move together.

    PYTHONPATH=src uv run python research/kutbit/events.py [DATA]

Output: research/kutbit/results_events.txt.
"""

import sys
import time
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import numpy as np
import polars as pl
from dataset import MARKET, day_from
from labels import targets
from skill import clustered_mean, load

from qfin_alphaguard.kutbit.features import CHANNELS, Day, channels

DATA = Path("data")
RESULTS = Path(__file__).parent / "results_events.txt"
YEARS = (2016, 2017, 2018)
LAST_YEAR = 2018  # 2019 chooses an agent's settings and 2020-2021 measure it once
HORIZONS = (1, 5, 15, 30, 60)  # minutes
SIZES = (4.0, 6.0, 8.0, 12.0)  # the least size of an extreme move, and its bins' edges
COOLDOWN = 30  # minutes before an extreme minute without another of the same stock
NEWS_MINUTES = 30  # minutes up to an extreme minute in which a headline counts
VOLUMES = (2.0, 5.0)  # a minute's volume over its usual: its bins' edges
MARKET_SHOCK = (
    3.0  # QQQ's own move, in its usual moves, that marks a market-wide minute
)
COSTS = (2.0, 5.0, 10.0)  # bps a round trip
PARTS = {
    "open": (0, 29),
    "morning": (30, 119),
    "midday": (120, 299),
    "late": (300, 389),
}
APART, VOLUME, OF_MARKET, SEEN = (
    CHANNELS.index(c) for c in ("apart", "volume", "market", "seen")
)
FIELDS = ("apart", "z", "own", "adverse")
KEYS = (
    "minute",
    "size",
    "sign",
    "volume",
    "news",
    "qqq",
    "bounce",
    *(f"{f}_{h}" for f in FIELDS for h in HORIZONS),
)


# --- the events -------------------------------------------------------------------------


def fresh(
    apart: np.ndarray, seen: np.ndarray, least: float, cooldown: int
) -> np.ndarray:
    """The minutes of fresh extreme moves: |apart| >= least, in a minute that traded after
    one that traded too (the first minute's move is from its own open), with no extreme
    minute in the `cooldown` minutes before; never the session's last bar."""
    traded = np.asarray(seen, dtype=bool)
    big = (np.abs(apart) >= least) & traded
    big[1:] &= traded[:-1]
    out, last = [], -cooldown - 1
    for t in np.flatnonzero(big):
        if t - last > cooldown and t < len(apart) - 1:
            out.append(t)
        last = t
    return np.array(out, dtype=int)


def session_events(day: Day, market: Day) -> dict[str, np.ndarray]:
    """The fresh extreme minutes of one session, their context and what followed each."""
    rows = channels(day, market)
    at = fresh(rows[:, APART], rows[:, SEEN], SIZES[0], COOLDOWN)
    if not len(at):
        return {k: np.array([]) for k in KEYS}
    sign = np.sign(rows[at, APART])
    headlines = np.convolve(day.news, np.ones(NEWS_MINUTES))[: len(day.news)]
    out = {
        "minute": at,
        "size": np.abs(rows[at, APART]),
        "sign": sign,
        "volume": np.exp(rows[at, VOLUME]),
        "news": headlines[at],
        "qqq": rows[at, OF_MARKET],
        "bounce": sign * np.log(day.open[at + 1] / day.close[at]) * 1e4,
    }
    found = targets(day, market, HORIZONS)
    for h in HORIZONS:
        f = found[h]
        whole = ~f["cut"][at]
        adverse = np.where(sign > 0, f["worst"][at], -f["best"][at])
        for name, values in (
            ("apart", sign * f["apart"][at] * 1e4),
            ("z", sign * f["z"][at]),
            ("own", sign * f["ret"][at] * 1e4),
            ("adverse", adverse * 1e4),
        ):
            out[f"{name}_{h}"] = np.where(whole, values, np.nan)
    return out


def part_of(minute: np.ndarray) -> np.ndarray:
    out = np.full(len(minute), "", dtype=object)
    for name, (first, last) in PARTS.items():
        out[(minute >= first) & (minute <= last)] = name
    return out


def collect(days: pl.DataFrame, arrays: dict, years=YEARS) -> pl.DataFrame:
    """Every stock's fresh extreme minutes in `years`, on days when QQQ's session is usable
    too, one row each: day, year, ticker, the context and what followed."""
    if max(years) > LAST_YEAR:
        raise ValueError(
            "the event study reads 2016-2018 only, the years an agent learns on"
        )
    usable = days.filter(pl.col("usable") & pl.col("year").is_in(list(years)))
    markets = {
        r["day"]: r
        for r in usable.filter(pl.col("ticker") == MARKET).iter_rows(named=True)
    }
    stocks = usable.filter(~pl.col("etf") & pl.col("day").is_in(list(markets)))
    parts = {k: [] for k in (*KEYS, "day", "year", "ticker")}
    for (d,), group in stocks.sort("day", "ticker").group_by(
        "day", maintain_order=True
    ):
        market = day_from(markets[d], arrays)
        for row in group.iter_rows(named=True):
            day = day_from(row, arrays)
            if len(day.open) != len(market.open):
                continue
            found = session_events(day, market)
            n = len(found["minute"])
            if not n:
                continue
            found["day"] = np.full(n, d.toordinal())
            found["year"] = np.full(n, d.year)
            found["ticker"] = np.full(n, row["ticker"]).astype(str)
            for k, v in found.items():
                parts[k].append(v)
    frame = pl.DataFrame({k: np.concatenate(v) for k, v in parts.items()})
    return frame.with_columns(
        pl.col("minute").cast(pl.Int64),
        pl.Series("part", part_of(frame["minute"].to_numpy()), dtype=pl.String),
    )


# --- the measures -----------------------------------------------------------------------


def size_bin(size: np.ndarray) -> np.ndarray:
    out = np.full(len(size), "", dtype=object)
    for name, low, high in size_bins():
        out[(size >= low) & (size < high)] = name
    return out


def size_bins() -> list[tuple[str, float, float]]:
    edges = [*SIZES, np.inf]
    return [
        (f"{low:g}+" if high == np.inf else f"{low:g}-{high:g}", low, high)
        for low, high in pairwise(edges)
    ]


def volume_names() -> list[str]:
    low, high = VOLUMES
    return [f"under {low:g}x", f"{low:g}-{high:g}x", f"{high:g}x+"]


def contexts(frame: pl.DataFrame) -> dict[str, tuple[np.ndarray, list[str]]]:
    """Each way of splitting the events: every event's group, and the groups in order."""
    volume = frame["volume"].to_numpy()
    low, high = VOLUMES
    names = volume_names()
    return {
        "all": (np.full(len(frame), "all", dtype=object), ["all"]),
        "size": (size_bin(frame["size"].to_numpy()), [b[0] for b in size_bins()]),
        "headline": (
            np.where(frame["news"].to_numpy() > 0, "a headline", "none").astype(object),
            ["a headline", "none"],
        ),
        "volume": (
            np.where(
                volume < low, names[0], np.where(volume < high, names[1], names[2])
            ).astype(object),
            names,
        ),
        "QQQ": (
            np.where(
                np.abs(frame["qqq"].to_numpy()) >= MARKET_SHOCK, "moved too", "quiet"
            ).astype(object),
            ["quiet", "moved too"],
        ),
        "part": (frame["part"].to_numpy(), list(PARTS)),
        "direction": (
            np.where(frame["sign"].to_numpy() > 0, "up", "down").astype(object),
            ["up", "down"],
        ),
    }


def mean_of(values: np.ndarray, day: np.ndarray) -> dict:
    """The mean of the values that are not NaN, with a t clustered by day."""
    keep = np.isfinite(values)
    return clustered_mean(values[keep], day[keep])


def measure(frame: pl.DataFrame, field: str = "apart") -> dict:
    """For each way of splitting the events, each group and each horizon: the signed mean of
    `field` with its t clustered by day, and the share that continued."""
    day = frame["day"].to_numpy()
    out = {}
    for split, (groups, order) in contexts(frame).items():
        out[split] = {}
        for group in order:
            keep = groups == group
            if not keep.any():
                continue
            row = {"events": int(keep.sum())}
            for h in HORIZONS:
                values = frame[f"{field}_{h}"].to_numpy()[keep]
                finite = values[np.isfinite(values)]
                row[h] = mean_of(values, day[keep])
                row[h]["continued"] = (
                    float(np.mean(finite > 0)) if len(finite) else float("nan")
                )
            out[split][group] = row
    return out


def adverse(frame: pl.DataFrame) -> dict:
    """By size, how far each horizon went against a position that followed: its median and
    its worst tenth, in bps."""
    sizes = size_bin(frame["size"].to_numpy())
    out = {}
    for group, _, _ in size_bins():
        if not (sizes == group).any():
            continue
        out[group] = {}
        for h in HORIZONS:
            values = frame[f"adverse_{h}"].to_numpy()[sizes == group]
            values = values[np.isfinite(values)]
            out[group][h] = (
                {
                    "median": float(np.median(values)),
                    "worst tenth": float(np.quantile(values, 0.1)),
                }
                if len(values)
                else {"median": float("nan"), "worst tenth": float("nan")}
            )
    return out


def follow_the_news(frame: pl.DataFrame) -> dict:
    """A simple rule to beat: follow an extreme minute with a headline, bet against one
    without; its signed mean in bps at each horizon, with its t."""
    day = frame["day"].to_numpy()
    side = np.where(frame["news"].to_numpy() > 0, 1.0, -1.0)
    return {h: mean_of(side * frame[f"apart_{h}"].to_numpy(), day) for h in HORIZONS}


def kinds(frame: pl.DataFrame) -> dict[str, np.ndarray]:
    """The kinds of event the hedge and the rules are read on."""
    moved = np.abs(frame["qqq"].to_numpy()) >= MARKET_SHOCK
    at_open = frame["part"].to_numpy() == "open"
    headline = frame["news"].to_numpy() > 0
    return {
        "all": np.ones(len(frame), dtype=bool),
        "QQQ moved too": moved,
        "QQQ quiet": ~moved,
        "at the open": at_open,
        "no headline": ~headline,
        "at the open, no headline": at_open & ~headline,
        "QQQ moved, no headline": moved & ~headline,
        "either, no headline": (at_open | moved) & ~headline,
    }


def hedge_effect(frame: pl.DataFrame) -> dict:
    """For each kind of event and horizon, the std of the stock's own move and of its move
    apart from QQQ, in bps, and the share of the variance the hedge takes out."""
    out = {}
    for kind in ("all", "QQQ moved too", "QQQ quiet", "at the open"):
        keep = kinds(frame)[kind]
        out[kind] = {}
        for h in HORIZONS:
            own = frame[f"own_{h}"].to_numpy()[keep]
            apart = frame[f"apart_{h}"].to_numpy()[keep]
            both = np.isfinite(own) & np.isfinite(apart)
            if both.sum() < 2:
                continue
            own_std, apart_std = own[both].std(), apart[both].std()
            out[kind][h] = {
                "own": float(own_std),
                "apart": float(apart_std),
                "removed": float(1 - apart_std**2 / own_std**2),
            }
    return out


def fades(frame: pl.DataFrame) -> dict:
    """Betting against the extreme minute, for the kinds of event without a headline: its
    mean in bps apart from QQQ at each horizon, with its t, before costs."""
    day = frame["day"].to_numpy()
    out = {}
    for kind in (
        "no headline",
        "at the open, no headline",
        "QQQ moved, no headline",
        "either, no headline",
    ):
        keep = kinds(frame)[kind]
        out[kind] = {
            h: mean_of(-frame[f"apart_{h}"].to_numpy()[keep], day[keep])
            for h in HORIZONS
        }
    return out


# --- the run ----------------------------------------------------------------------------


def study(data: Path = DATA, years=YEARS, log=print) -> dict:
    started = time.time()
    days, arrays = load(data, years)
    frame = collect(days, arrays, years)
    log(f"events: {len(frame):,} ({time.time() - started:.0f} s)")
    return {
        "run_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M"),
        "seconds": round(time.time() - started),
        "years": (min(years), max(years)),
        "events": len(frame),
        "by_year": dict(frame.group_by("year").len().sort("year").iter_rows()),
        "days": frame["day"].n_unique(),
        "bounce": mean_of(frame["bounce"].to_numpy(), frame["day"].to_numpy()),
        "apart": measure(frame, "apart"),
        "z": measure(frame, "z"),
        "own": measure(frame, "own"),
        "adverse": adverse(frame),
        "news rule": follow_the_news(frame),
        "hedge": hedge_effect(frame),
        "fades": fades(frame),
    }


# --- the report -------------------------------------------------------------------------


def cell(found: dict, digits: int = 1) -> str:
    if found["n"] < 2:
        return "-"
    return f"{found['mean']:+.{digits}f} ({found['t']:+.1f})"


def table(title: str, by: dict, digits: int = 1) -> list[str]:
    lines = [
        "",
        title,
        f"  {'group':<16}{'events':>9}"
        + "".join(f"{f'{h} min':>16}" for h in HORIZONS),
    ]
    for group, row in by.items():
        lines.append(
            f"  {group:<16}{row['events']:>9,}"
            + "".join(f"{cell(row[h], digits):>16}" for h in HORIZONS)
        )
    return lines


def report(found: dict) -> str:
    lines = [
        "KUTBIT's extreme minutes: the event study before the first agent",
        (
            f"Stored minutes of {found['years'][0]}-{found['years'][1]}; 2019-2026 not "
            f"read. Run {found['run_utc']} UTC in {found['seconds']} s."
        ),
        (
            f"{found['events']:,} events on {found['days']:,} days: a stock's move apart "
            f"from QQQ of at least {SIZES[0]:g} usual moves in one minute, none in the "
            f"{COOLDOWN} minutes before. By year: "
            + ", ".join(f"{y} {n:,}" for y, n in found["by_year"].items())
            + "."
        ),
        (
            "Each cell: the mean in bps, signed by the extreme minute's direction (+ it "
            "continued, - it reversed), from the next bar's open; its t clustered by day."
        ),
        (
            f"The bounce from the extreme minute's close to the next open, which no order "
            f"can catch: {cell(found['bounce'])} bps. A round trip costs "
            + ", ".join(f"{c:g}" for c in COSTS)
            + " bps by assumption."
        ),
    ]
    names = {
        "all": "All events",
        "size": "By size, in usual moves",
        "headline": f"By a headline naming the stock in the {NEWS_MINUTES} minutes up to it",
        "volume": "By the minute's volume over its usual",
        "QQQ": f"By QQQ's own move that minute (moved too: {MARKET_SHOCK:g}+ usual moves)",
        "part": "By the part of the day",
        "direction": "By direction",
    }
    for split, title in names.items():
        lines += table(f"{title}: the move apart from QQQ", found["apart"][split])
    lines += table("All events: the stock's own move, unhedged", found["own"]["all"])
    lines += table(
        "All events: the move apart from QQQ in usual moves (z, not bps)",
        found["z"]["all"],
        digits=3,
    )
    continued = found["apart"]["all"]["all"]
    lines += [
        "",
        "Share of events that continued, apart from QQQ: "
        + ", ".join(f"{h} min {continued[h]['continued']:.1%}" for h in HORIZONS),
        "",
        (
            "How far it went against a position that followed, in bps from the next "
            "open: median | worst tenth"
        ),
        f"  {'size':<16}" + "".join(f"{f'{h} min':>16}" for h in HORIZONS),
    ]
    for group, row in found["adverse"].items():
        lines.append(
            f"  {group:<16}"
            + "".join(
                f"{row[h]['median']:+.0f} | {row[h]['worst tenth']:+.0f}".rjust(16)
                for h in HORIZONS
            )
        )
    rule = found["news rule"]
    lines += [
        "",
        "A rule to beat, follow with a headline and bet against without: "
        + ", ".join(f"{h} min {cell(rule[h])}" for h in HORIZONS),
        "",
        (
            "What the hedge takes out: the std in bps of the stock's own move -> of its "
            "move apart from QQQ (share of the variance taken out)"
        ),
        f"  {'kind':<16}" + "".join(f"{f'{h} min':>16}" for h in HORIZONS),
    ]
    for kind, row in found["hedge"].items():
        lines.append(
            f"  {kind:<16}"
            + "".join(
                (
                    f"{row[h]['own']:.0f}->{row[h]['apart']:.0f} ({row[h]['removed']:+.0%})"
                    if h in row
                    else "-"
                ).rjust(16)
                for h in HORIZONS
            )
        )
    lines += [
        "",
        (
            "Betting against it, apart from QQQ, in bps before costs (t); to pay, it "
            "must beat a round trip of " + ", ".join(f"{c:g}" for c in COSTS) + " bps"
        ),
        f"  {'kind':<26}" + "".join(f"{f'{h} min':>16}" for h in HORIZONS),
    ]
    for kind, row in found["fades"].items():
        lines.append(f"  {kind:<26}" + "".join(f"{cell(row[h]):>16}" for h in HORIZONS))
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> None:
    data = Path(argv[0]) if argv else DATA
    text = report(study(data))
    RESULTS.write_text(text)
    print(text)


if __name__ == "__main__":
    main(sys.argv[1:])
