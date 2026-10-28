"""How fast member news is priced within the session (seen years, no model; before N7).

For each year's most liquid members (data/raw/intraday: 1-minute bars of the regular session),
every article N2's rule counts for one of them, known in the session from 09:31 to 15:00 New
York time, whose headline points a way (event_study.py's kinds: upgrades and downgrades,
targets raised and lowered, earnings beating and missing, guidance raised and lowered). One
event a stock, day and way: its first such article.

From the close of the minute the article became known in, the stock's return less QQQ's over
the same minutes, signed by the kind's way, in bps:
- the 30 minutes before it (to the close of the minute before), to see what came first;
- the arrival minute itself (its close over the minute before's);
- 1, 5, 30 and 60 minutes on, and to the session's close.
The mean over events and its t, for 2016-2021 and 2022-2026 apart. Every one of these years
has been seen: nothing here is a test.

Usage: uv run python research/news_reader/intraday_reaction.py
"""

import sys
from datetime import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from event_study import KINDS, PAIRS, headlines, kinds_of

NEW_YORK = ZoneInfo("America/New_York")
MINUTES = Path("data/raw/intraday")
N6 = Path("data/news_reader/n6")
INDEX = Path("data/news_reader/emb/index.parquet")
OUT = Path(__file__).parent / "results_intraday_reaction.txt"
FIRST, LAST = time(9, 31), time(15, 0)
AHEAD = (1, 5, 30, 60)
WAY = {k: (1 if k == up else -1) for up, down in PAIRS for k in (up, down)}


def bars(ticker: str, year: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """The minutes' starts (int64 UTC microseconds), closes and New York dates."""
    path = MINUTES / ticker / f"{year}.parquet"
    if not path.exists():
        return None
    f = pl.read_parquet(path, columns=["start", "close"]).sort("start")
    when = f["start"]
    return (
        when.cast(pl.Int64).to_numpy(),
        f["close"].to_numpy(),
        when.dt.convert_time_zone("America/New_York").dt.date().to_numpy(),
    )


def moves(stock, market, known: np.ndarray) -> np.ndarray:
    """Per event: the arrival minute's move, then each AHEAD and to the close, stock less
    market; nan where a horizon leaves the day or a minute is missing."""
    out = np.full((len(known), 3 + len(AHEAD)), np.nan)
    s_when, s_close, s_day = stock
    m_when, m_close, _ = market
    at = np.searchsorted(s_when, known, "right") - 1
    for e, i in enumerate(at):
        if i < 1 or s_day[i] != s_day[i - 1] or known[e] - s_when[i] >= 60_000_000:
            continue
        j = np.searchsorted(m_when, s_when[i])
        if j >= len(m_when) or m_when[j] != s_when[i] or j < 1:
            continue
        day_end = np.searchsorted(s_day, s_day[i], "right") - 1

        def leg(a, b, ma, mb):
            return (s_close[b] / s_close[a] - 1) - (m_close[mb] / m_close[ma] - 1)

        a, ma = i - 31, j - 31
        if a >= 0 and s_day[a] == s_day[i] and ma >= 0 and m_when[ma] == s_when[a]:
            out[e, 0] = leg(a, i - 1, ma, j - 1)
        out[e, 1] = leg(i - 1, i, j - 1, j)
        for k, n in enumerate(AHEAD):
            b, mb = i + n, j + n
            if b <= day_end and mb < len(m_when) and m_when[mb] == s_when[b]:
                out[e, 2 + k] = leg(i, b, j, mb)
        mb = np.searchsorted(m_when, s_when[day_end])
        if mb < len(m_when) and m_when[mb] == s_when[day_end]:
            out[e, -1] = leg(i, day_end, j, mb)
    return out


def main() -> None:
    days = pl.read_parquet(
        N6 / "days.parquet", columns=["day", "ticker", "start", "stop"]
    )
    pairs = pl.read_parquet(N6 / "pairs.parquet", columns=["id"])
    ticker = np.repeat(
        days["ticker"].to_numpy(), (days["stop"] - days["start"]).to_numpy()
    )
    day = np.repeat(days["day"].to_numpy(), (days["stop"] - days["start"]).to_numpy())
    pairs = pairs.with_columns(ticker=pl.Series(ticker), day=pl.Series(day))
    known = pl.read_parquet(INDEX, columns=["id", "known"])
    pairs = pairs.join(known, on="id").with_columns(
        clock=pl.col("known").dt.convert_time_zone("America/New_York").dt.time()
    )
    pairs = pairs.filter(pl.col("clock").is_between(FIRST, LAST))
    text = headlines(set(pairs["id"].to_list()))
    pairs = pairs.with_columns(
        kinds=pl.Series([sorted(kinds_of(text[i])) for i in pairs["id"].to_list()])
    ).explode("kinds", empty_as_null=True)
    pairs = pairs.drop_nulls("kinds").with_columns(
        way=pl.col("kinds").replace_strict(WAY, return_dtype=pl.Int8)
    )
    # a stock, day and way once: its first article; a day pointing both ways is left out
    both = pairs.group_by("ticker", "day").agg(pl.col("way").n_unique().alias("ways"))
    events = (
        pairs.join(both.filter(pl.col("ways") == 1), on=["ticker", "day"])
        .sort("known")
        .unique(["ticker", "day", "way"], keep="first")
        .with_columns(year=pl.col("day").dt.year())
    )
    results = []
    cache = {}
    for (name, year), part in events.group_by("ticker", "year"):
        stock = bars(name, year)
        if year not in cache:
            cache[year] = bars("QQQ", year)
        if stock is None or cache[year] is None:
            continue
        signed = moves(stock, cache[year], part["known"].cast(pl.Int64).to_numpy())
        signed *= part["way"].to_numpy()[:, None]
        results.append(
            part.select("kinds", "year").with_columns(
                **{
                    c: pl.Series(signed[:, k])
                    for k, c in enumerate(
                        ["before", "arrival", *[f"+{n}" for n in AHEAD], "close"]
                    )
                }
            )
        )
    table = pl.concat(results)
    columns = ["before", "arrival", *[f"+{n}" for n in AHEAD], "close"]
    lines = [
        "intraday_reaction.py: the most liquid members' news known in the session (09:31-15:00 New",
        "York), from the close of the arrival minute: the stock less QQQ, signed by the headline's",
        "way, in bps (t; events).",
        "",
    ]
    for era, (a, b) in {"2016-2021": (2016, 2021), "2022-2026": (2022, 2026)}.items():
        lines.append(era)
        lines.append(
            f"{'kind':<18}"
            + "".join(
                f"{c:>20}"
                for c in [
                    "30 min before",
                    "arrival minute",
                    *[f"{n} min on" for n in AHEAD],
                    "to the close",
                ]
            )
        )
        part = table.filter(pl.col("year").is_between(a, b))
        for kind in [*KINDS, "all"]:
            rows = part if kind == "all" else part.filter(pl.col("kinds") == kind)
            cells = ""
            for c in columns:
                v = rows[c].drop_nulls().drop_nans().to_numpy()
                if len(v) < 3:
                    cells += f"{'':>20}"
                    continue
                t = v.mean() / v.std(ddof=1) * np.sqrt(len(v))
                cells += f"{v.mean() * 1e4:>+8.1f} ({t:>+5.1f};{len(v):>4})"
            lines.append(f"{kind:<18}{cells}")
        lines.append("")
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
