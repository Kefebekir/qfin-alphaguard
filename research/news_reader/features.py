"""N1's and N2's features from the headline embeddings, for each ChronoBERT vintage.

For each vintage written by embed.py, over the sessions it serves (to its last test year),
with the index of articles in the order they became known and each article's embedding found
by its id:

- N1, one row a session: the articles known from 16:00 New York time on the session before
  to 09:55 on the day (every article, whatever it tags). Features: their mean embedding and
  log(1 + their number). A session with none gets zeros (none so far).
- N2, one row a member and session: the articles tagging at most two stocks, known from
  16:00 New York time on the session before to 15:30 on the day, that tag a stock in the
  S&P 500 that day. Features: their mean embedding and log(1 + their number).

Benzinga's symbols are matched to our tickers on the day: spaces stripped, "/" and "." read as
"-" (BRK/B, BRK.B and BRK-B are one stock), and a stock whose ticker was later reused is our
"<ticker>_old" (or "_old1"): the symbol goes to whichever of them was a member that day. A
symbol that fits two members, or none, is left out.

Output, under EMB_DIR: n1_v<vintage>.parquet (day, count) with n1_v<vintage>.npy, and
n2_v<vintage>.parquet (ticker, day, count) with n2_v<vintage>.npy: float16 [rows, 768], the
mean embeddings in row order.

Usage: features.py EMB_DIR PRICES_PARQUET
"""

import re
import sys
from collections import defaultdict
from datetime import UTC, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from qfin_alphaguard.sessions import sessions

EMB, PRICES = Path(sys.argv[1]), sys.argv[2]
NEW_YORK = ZoneInfo("America/New_York")

index = pl.read_parquet(EMB / "index.parquet")
known = index["known"].to_numpy()  # sorted, UTC
days = [
    s.day for s in sessions(index["known"].min().date(), index["known"].max().date())
]


def at(day, hour, minute):
    """New York's hour:minute on `day`, as a UTC datetime64 like the index's times."""
    local = datetime.combine(day, time(hour, minute), tzinfo=NEW_YORK)
    return np.datetime64(local.astimezone(UTC).replace(tzinfo=None), "us")


# Windows: (16:00 on the session before, end on the day], end 09:55 for N1 and 15:30 for N2.
starts = np.array([at(days[i - 1], 16, 0) for i in range(1, len(days))])
n1_ends = np.array([at(d, 9, 55) for d in days[1:]])
n2_ends = np.array([at(d, 15, 30) for d in days[1:]])
window_days = days[1:]

# Members by day: (day, base symbol) -> our ticker, or None if two fit.
members = defaultdict(list)
for d, ticker in (
    pl.read_parquet(PRICES, columns=["date", "ticker", "sp500"])
    .filter(pl.col("sp500"))
    .select("date", "ticker")
    .iter_rows()
):
    members[(d, re.sub(r"_old\d*$", "", ticker))].append(ticker)


def normal(symbol: str) -> str:
    return symbol.strip().replace("/", "-").replace(".", "-")


# N2's (article row, day, ticker) pairs, from the articles tagging one or two stocks.
# the first article after each window's start
first = np.searchsorted(known, starts, "right")
last_n2 = np.searchsorted(known, n2_ends, "right")
pairs = []
symbols = index["symbols"].to_list()
for w, (a, b) in enumerate(zip(first, last_n2, strict=True)):
    day = window_days[w]
    for row in range(a, b):
        tags = symbols[row]
        if not 1 <= len(tags) <= 2:
            continue
        for tag in tags:
            fits = members.get((day, normal(tag)), [])
            if len(fits) == 1:
                pairs.append((row, w, fits[0]))
print(f"N2: {len(pairs):,} member articles in windows")

ids = index["id"].to_numpy()


def last_test_year(vintage: int) -> int:
    return 2026 if vintage == 2024 else vintage + 1


for vintage_file in sorted(EMB.glob("v*.npy")):
    if not re.fullmatch(r"v\d{4}", vintage_file.stem):
        continue  # the ids, or a vintage not finished
    vintage = vintage_file.stem
    emb = np.load(vintage_file, mmap_mode="r")
    emb_ids = np.load(EMB / f"{vintage}.ids.npy")
    # Each index row's row in this vintage's embeddings, by article id (-1 if absent).
    order = np.argsort(emb_ids)
    found = np.minimum(np.searchsorted(emb_ids[order], ids), len(order) - 1)
    row_of = np.where(emb_ids[order][found] == ids, order[found], -1)
    cover = last_test_year(int(vintage[1:]))
    keep = [w for w in range(len(window_days)) if window_days[w].year <= cover]
    missing = 0

    # N1: the sessions this vintage serves; every article in a window must be embedded.
    last_n1 = np.searchsorted(known, n1_ends, "right")
    n1 = np.zeros((len(keep), emb.shape[1]), dtype=np.float32)
    counts = np.zeros(len(keep), dtype=np.int64)
    for k, w in enumerate(keep):
        rows = row_of[first[w] : last_n1[w]]
        missing += int((rows < 0).sum())
        counts[k] = len(rows)
        if len(rows):
            n1[k] = emb[np.sort(rows)].astype(np.float32).mean(0)
    pl.DataFrame(
        {"day": [window_days[w] for w in keep], "count": counts}
    ).write_parquet(EMB / f"n1_{vintage}.parquet")
    np.save(EMB / f"n1_{vintage}.npy", n1.astype(np.float16))

    # N2: group the pairs by (ticker, day) on the sessions this vintage serves.
    groups, sums, counts2 = {}, [], []
    for row, w, ticker in pairs:
        if window_days[w].year > cover:
            continue
        r = row_of[row]
        if r < 0:
            missing += 1
            continue
        key = (ticker, w)
        g = groups.get(key)
        if g is None:
            g = groups[key] = len(sums)
            sums.append(np.zeros(emb.shape[1], dtype=np.float32))
            counts2.append(0)
        sums[g] += emb[r].astype(np.float32)
        counts2[g] += 1
    if missing:
        sys.exit(f"{vintage}: {missing} articles in its windows were not embedded")
    keys = list(groups)
    n2 = np.stack(sums) / np.array(counts2, dtype=np.float32)[:, None]
    pl.DataFrame(
        {
            "ticker": [k[0] for k in keys],
            "day": [window_days[k[1]] for k in keys],
            "count": counts2,
        }
    ).write_parquet(EMB / f"n2_{vintage}.parquet")
    np.save(EMB / f"n2_{vintage}.npy", n2.astype(np.float16))
    print(
        f"{vintage}: N1 {len(keep):,} sessions ({counts.mean():.0f} articles a morning), "
        f"N2 {len(keys):,} member-days",
        flush=True,
    )
