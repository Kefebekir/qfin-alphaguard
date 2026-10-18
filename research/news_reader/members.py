"""Which articles count for which member and session: N2's rule, for N3's scripts.

The same rule as features.py, which built N2's features and is left as it ran: an article
tagging one or two stocks, known from 16:00 New York time on the session before to 15:30 on
the day, counts for each tagged stock that is an S&P 500 member that day. Benzinga's symbols
are read with spaces stripped and "/" and "." as "-", and go to whichever of a ticker and its
"<ticker>_old" was a member that day; a symbol fitting two members, or none, is left out.
"""

import re
from collections import defaultdict
from datetime import UTC, date, datetime, time
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from qfin_alphaguard.sessions import sessions

NEW_YORK = ZoneInfo("America/New_York")


def at(day: date, hour: int, minute: int) -> np.datetime64:
    """New York's hour:minute on `day`, as a UTC datetime64 like the index's times."""
    local = datetime.combine(day, time(hour, minute), tzinfo=NEW_YORK)
    return np.datetime64(local.astimezone(UTC).replace(tzinfo=None), "us")


def normal(symbol: str) -> str:
    return symbol.strip().replace("/", "-").replace(".", "-")


def member_articles(index: pl.DataFrame, prices: pl.DataFrame) -> pl.DataFrame:
    """(id, ticker, day) for each article counting for a member's session.

    `index` holds id, known (UTC) and symbols, in the order articles became known;
    `prices` holds date, ticker and the sp500 flag.
    """
    known = index["known"].dt.replace_time_zone(None).to_numpy()
    days = [
        s.day
        for s in sessions(index["known"].min().date(), index["known"].max().date())
    ]
    starts = np.array([at(days[i - 1], 16, 0) for i in range(1, len(days))])
    ends = np.array([at(d, 15, 30) for d in days[1:]])
    members = defaultdict(list)
    for d, ticker in (
        prices.filter(pl.col("sp500")).select("date", "ticker").iter_rows()
    ):
        members[(d, re.sub(r"_old\d*$", "", ticker))].append(ticker)
    first = np.searchsorted(known, starts, "right")
    last = np.searchsorted(known, ends, "right")
    ids, symbols = index["id"].to_list(), index["symbols"].to_list()
    rows = []
    for w, (a, b) in enumerate(zip(first, last, strict=True)):
        day = days[w + 1]
        for row in range(a, b):
            tags = symbols[row]
            if not 1 <= len(tags) <= 2:
                continue
            for tag in tags:
                fits = members.get((day, normal(tag)), [])
                if len(fits) == 1:
                    rows.append((ids[row], fits[0], day))
    return pl.DataFrame(
        rows, schema={"id": pl.Int64, "ticker": pl.String, "day": pl.Date}, orient="row"
    )


def checkpoint(known_utc: datetime) -> str:
    """The PIT checkpoint that reads an article known at `known_utc` (N3): December of the
    year before, by New York's calendar; November 2015 for 2016; December 2024 for 2025 on."""
    year = known_utc.astimezone(NEW_YORK).year
    if year <= 2016:
        return "201511"
    return f"{min(year - 1, 2024)}12"
