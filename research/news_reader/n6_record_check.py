"""Does N6's record compute what N6's development and second look did? Ten past sessions.

As forward_check.py does for N5: the research's articles (data/frontier/news_all, each in its
last version, those created within the capture's margin) stand in for the 15:31 captures of
the last ten sessions to 8 October 2026. They go through the record's own path (counted,
embed, N6's scale, member_inputs, the frozen networks, score_day) and are compared, step by
step, with what N6's research built and read (n6_harness's arrays from n6_data.py and the
store of N6's eyes): the member-days and their counts, each member-day's headlines in order,
their age, edit and two-tag flags, each headline's vector, the frozen networks' forecasts on
both, and the days' scores. No headline is printed.

Usage: PYTHONPATH=src data/llmenv/bin/python research/news_reader/n6_record_check.py
"""

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import torch

sys.path.insert(0, str(Path(__file__).parent))
import forward_record as fr
import harness
import n6_freeze
import n6_harness as nh
import n6_record as n6r

from qfin_alphaguard.data.news_capture import MARGIN
from qfin_alphaguard.sessions import sessions

DAYS = [s.day for s in sessions(date(2026, 9, 24), date(2026, 10, 8))][-10:]


def main() -> None:
    prices = pl.read_parquet(
        "data/raw/prices.parquet", columns=["date", "ticker", "close", "sp500"]
    )
    articles = {}
    for path in sorted(Path("data/frontier/news_all").glob("2026-*.jsonl")):
        for line in path.read_text().splitlines():
            a = json.loads(line)
            articles[a["id"]] = a
    print(f"{len(DAYS)} sessions, {DAYS[0]} to {DAYS[-1]}")

    near, pairs = {}, {}
    for day in DAYS:
        start, end = fr.window(day)
        near[day] = [
            a
            for a in articles.values()
            if start - MARGIN < fr.stamp(a["created_at"]) <= end + timedelta(days=1)
        ]
        pairs[day] = fr.counted(near[day], day, fr.members_on(prices, day))
    ids = sorted({i for p in pairs.values() for i, _, _ in p})
    with np.load(n6r.EYES) as kept:
        eyes = {k: kept[k] for k in (f"mean_{n6r.READER}", f"spread_{n6r.READER}")}
    read = fr.embed([articles[i]["headline"] for i in ids])
    vectors = dict(zip(ids, n6r.scaled(read, eyes), strict=True))

    a = nh.load()
    when = a.days["day"].to_numpy()
    store = np.load(nh.N6 / "headlines.npy", mmap_mode="r")
    store_ids = np.load(nh.N6 / "ids.npy")
    with np.load(n6r.WEIGHTS, allow_pickle=False) as kept:
        models = n6_freeze.models_from(dict(kept))

    same_days = same_heads = True
    info_gap = forecast_gap = score_gap = 0.0
    vector_cos = 1.0
    spread = []
    for day in DAYS:
        tickers, x, mine = n6r.member_inputs(near[day], pairs[day], day, vectors)
        rows = np.flatnonzero(when == np.datetime64(day))
        theirs = a.days[rows]["ticker"].to_list()
        same_days &= tickers == theirs and np.allclose(x["count"], a.count[rows])
        for n, r in enumerate(rows):
            record_ids = [i for i, t, _ in pairs[day] if t == tickers[n]]
            record_ids = record_ids[-nh.MAX_HEADLINES :]
            research_rows = a.heads[r][a.heads[r] >= 0]
            same_heads &= store_ids[research_rows].tolist() == record_ids
            if len(research_rows) != len(record_ids):
                continue
            ours = mine[x["heads"][n][: len(record_ids)]].astype(np.float32)
            stored = np.asarray(store[research_rows], dtype=np.float32)
            cos = (ours * stored).sum(1) / (
                np.linalg.norm(ours, axis=1) * np.linalg.norm(stored, axis=1)
            )
            vector_cos = min(vector_cos, float(cos.min()))
            gap = np.abs(
                x["info"][n][: len(record_ids)] - a.head_info[r][: len(record_ids)]
            )
            info_gap = max(info_gap, float(gap.max()))
        f_record = n6r.forecast(models, mine, x)
        needed = np.unique(a.heads[rows][a.heads[rows] >= 0])
        heads = np.where(a.heads[rows] >= 0, np.searchsorted(needed, a.heads[rows]), -1)
        research_x = {
            "heads": torch.as_tensor(heads.astype(np.int32)),
            "info": torch.as_tensor(a.head_info[rows]),
            "count": torch.as_tensor(a.count[rows]),
        }
        mini = torch.as_tensor(np.asarray(store[needed]))
        f_research = n6_freeze.forecast(models, mini, research_x)
        spread.append(np.std(f_research))
        forecast_gap = max(forecast_gap, float(np.abs(f_record - f_research).max()))
        returns = fr.next_returns(prices, day)
        s_record = fr.score_day(tickers, f_record, returns, nh.LEVERS, harness)
        s_research = fr.score_day(theirs, f_research, returns, nh.LEVERS, harness)
        if s_record["score"] is not None and s_research["score"] is not None:
            score_gap = max(
                score_gap, abs(s_record["score"] - s_research["score"]) * 1e4
            )
    print(f"1. member-days, their tickers and counts the same each day: {same_days}")
    print(f"2. each member-day's headlines, in order, the same articles: {same_heads}")
    print(f"3. age, edit and two-tag flags: largest gap {info_gap:.2e}")
    print(f"4. each headline's vector against N6's store: cosine min {vector_cos:.6f}")
    print(
        f"5. the frozen networks' forecasts: largest gap {forecast_gap:.2e} "
        f"(their spread about {np.mean(spread):.2e})"
    )
    print(f"6. the days' scores: largest gap {score_gap:.4f} bps")


if __name__ == "__main__":
    main()
