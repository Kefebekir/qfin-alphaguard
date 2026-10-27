"""Does N5's record compute what N4 tested? Ten past sessions through forward_record.py.

The research's articles (data/frontier/news_all, each in its last version, those created
within the capture's margin of 30 days before the window) stand in for the 15:31 captures of
the last ten sessions to 8 October 2026, the latest whose features N3 and N4 built with
PIT-4B-202412. They go through the record's own path (counted, embed, member_days, the frozen
arrays, score_day) and are compared, step by step, with what the research computed: the
articles and members (members.py), the member-days and their counts (n3_rows.parquet), each
headline's embedding (PIT-4B/202412.last.npy), the features (n3_base-last.npy), the frozen
model's forecasts on each, the days' positions and scores, and the scores against the
research's own scoring (harness.n4_days, with its next-day returns). No headline is printed.

Usage: PYTHONPATH=src data/llmenv/bin/python research/news_reader/forward_check.py
"""

import json
import sys
import types
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
import forward
import forward_record as fr
import harness
from members import member_articles

from qfin_alphaguard.data.news_capture import MARGIN
from qfin_alphaguard.sessions import sessions

DAYS = [s.day for s in sessions(date(2026, 9, 24), date(2026, 10, 8))][-10:]
PIT_EMB = Path("data/news_reader/pit_emb")

prices = pl.read_parquet(
    "data/raw/prices.parquet", columns=["date", "ticker", "close", "sp500"]
)
index = pl.read_parquet("data/news_reader/emb/index.parquet")
articles = {}
for path in sorted(Path("data/frontier/news_all").glob("2026-*.jsonl")):
    for line in path.read_text().splitlines():
        a = json.loads(line)
        articles[a["id"]] = a  # the last version on disk, as the research read it
print(f"{len(DAYS)} sessions, {DAYS[0]} to {DAYS[-1]}")

# 1. The articles and members.
ours = {}
for day in DAYS:
    start, end = fr.window(day)
    near = [
        a
        for a in articles.values()
        if start - MARGIN < fr.stamp(a["created_at"]) <= end + timedelta(days=1)
    ]
    ours[day] = fr.counted(near, day, fr.members_on(prices, day))
theirs = member_articles(index, prices.select("date", "ticker", "sp500"))
theirs = theirs.filter(pl.col("day").is_in(DAYS))
mine = {(i, t, d) for d, pairs in ours.items() for i, t, _ in pairs}
research = set(theirs.select("id", "ticker", "day").iter_rows())
print(
    f"1. article-member pairs: record {len(mine):,}, research {len(research):,}, same: {mine == research}"
)

# 2. The member-days and their counts.
rows = pl.read_parquet(PIT_EMB / "n3_rows.parquet").with_row_index("row")
rows = rows.filter(pl.col("day").is_in(DAYS))
ids = sorted({i for pairs in ours.values() for i, _, _ in pairs})
texts = {i: articles[i]["headline"] for i in ids}
vectors = dict(zip(ids, fr.embed([texts[i] for i in ids]), strict=True))
same_counts = True
x_ours, x_theirs, keys = [], [], []
stored = np.load(PIT_EMB / "n3_base-last.npy", mmap_mode="r")
for day in DAYS:
    tickers, counts, x = fr.member_days(ours[day], vectors)
    research_day = rows.filter(pl.col("day") == day).sort("ticker")
    same_counts &= research_day["ticker"].to_list() == tickers
    same_counts &= research_day["count"].to_list() == counts
    x_ours.append(x)
    x_theirs.append(
        np.hstack(
            [
                stored[research_day["row"].to_numpy()].astype(np.float32),
                np.log1p(research_day["count"].to_numpy())[:, None],
            ]
        )
    )
    keys += [(day, t) for t in tickers]
print(
    f"2. member-days: {len(keys):,}; tickers and counts the same each day: {same_counts}"
)

# 3. Each headline's embedding against the one the research stored.
stored_ids = np.load(PIT_EMB / "PIT-4B" / "202412.ids.npy")
stored_last = np.load(PIT_EMB / "PIT-4B" / "202412.last.npy", mmap_mode="r")
where = {int(a): i for i, a in enumerate(stored_ids)}
a = np.array([vectors[i] for i in ids], dtype=np.float32)
b = np.array([stored_last[where[i]] for i in ids], dtype=np.float32)
cos = (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
print(
    f"3. headline embeddings: {len(ids):,}; cosine min {cos.min():.6f}, largest gap {np.abs(a - b).max():.4f}"
)

# 4. Features, forecasts, positions and scores.
xo, xt = np.vstack(x_ours), np.vstack(x_theirs)
print(
    f"4. features: largest gap {np.abs(xo - xt).max():.4f} (float16 storage: about 0.03)"
)
arrays = forward.load(fr.WEIGHTS)
chosen = json.loads((harness.VARIANTS / "N4_chosen.json").read_text())
levers = {k: chosen[k] for k in harness.N4_LEVERS}
start = 0
same_sides, gaps, record_days, forecasts = 0, [], {}, []
for day, x_day in zip(DAYS, x_ours, strict=True):
    n = len(x_day)
    fo = forward.forecast(arrays, x_day)
    ft = forward.forecast(arrays, xt[start : start + n])
    start += n
    tickers = [t for d, t in keys if d == day]
    returns = fr.next_returns(prices, day)
    ro = fr.score_day(tickers, fo, returns, levers, harness)
    rt = fr.score_day(tickers, ft, returns, levers, harness)
    wo = fr.positions(fo, levers, harness)
    wt = fr.positions(ft, levers, harness)
    same_sides += bool(np.array_equal(np.sign(wo), np.sign(wt)))
    gaps.append(abs(ro["score"] - rt["score"]) * 1e4)
    record_days[day] = ro
    forecasts += [(day, t, float(f)) for t, f in zip(tickers, fo, strict=True)]
print(
    f"5. forecasts' sides the same on {same_sides} of {len(DAYS)} days; largest score gap {max(gaps):.4f} bps"
)

# 6. The record's scores against the research's own scoring on the same forecasts.
test = (
    pl.DataFrame(forecasts, schema=["day", "ticker", "forecast"], orient="row")
    .join(harness.next_day_returns(), on=["day", "ticker"], how="inner")
    .with_columns(vol=pl.lit(1.0), sector=pl.lit(0))
)
research_days = {
    d["day"]: d
    for d in harness.n4_days(
        types.SimpleNamespace(test=test), test["forecast"].to_numpy(), levers
    )
}
same_days = set(research_days) == {
    d for d, r in record_days.items() if r["score"] is not None
}
largest = max(
    abs(record_days[d][k] - research_days[d][k]) * 1e4
    for d in research_days
    for k in ("score", "long", "short", "everyone")
)
print(
    f"6. against harness.n4_days: the same {len(research_days)} days scored: {same_days}; "
    f"largest gap in score, sides and mean {largest:.6f} bps"
)
