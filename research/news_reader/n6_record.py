"""N6's record, day by day: the frozen network forward, from the headlines captured at 15:31.

As N5's record (forward_record.py), whose capture, rules and scoring it uses unchanged: each
session captured by `qfin news-capture` is scored once the next session's close is known, by N4's
variant 2's portfolio (a tenth a side, ties in, equal weights, at least 20 members, 2 bps a
day). The model differs. Each counted headline is read by PIT-4B-202412, as N5 reads it, and
put in 202412's scaled reading by N6's eyes (n6-eyes.npz). Each member-day's last 32
headlines, by the time they became known, go in one by one, with log(1 + their age in hours
at 15:30), whether they count from an edit and whether they tag two stocks, and log(1 + the
member-day's count). The five frozen networks (news-n6-forward.npz) read them and their
forecasts are averaged.

A session with no capture has no score and is listed; a day scored is not scored again. The
test is the card's (pre-registration N6): decision 0020's betting test against no trade, clip
0.092, alpha 0.00625, promoted when the wealth reaches 160. Reported beside it: N6 less N5's
frozen N4, day by day, on the sessions both scored.

    PYTHONPATH=src data/llmenv/bin/python research/news_reader/n6_record.py

It copies the captures and prices from S3 first, as N5's record does, then writes
data/news_reader/forward_n6/days.parquet and record.html. No headline is printed or kept.
"""

import hashlib
import html
import sys
from collections import defaultdict
from datetime import UTC, date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
import forward_record as fr
from dev_report import PAGE, _cell, signed
from n6_harness import LEVERS, MAX_HEADLINES

from qfin_alphaguard.challengers import load_card, weights_path
from qfin_alphaguard.evidence import BettingTest, Evidence
from qfin_alphaguard.sessions import sessions

NEW_YORK = ZoneInfo("America/New_York")
OUT = Path("data/news_reader/forward_n6")
WEIGHTS = Path("data/models/news-n6-forward.npz")
EYES = Path("data/models/n6-eyes.npz")
CARD = Path("challengers/news-n6-forward.yaml")
N5_DAYS = fr.OUT / "days.parquet"
READER = "202412"
TEST = BettingTest(scale=1.0, clip=0.092, alpha=0.00625)  # pre-registration N6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def scaled(vectors: np.ndarray, eyes: dict) -> np.ndarray:
    """202412's readings in N6's one scale (n6_eyes.aligned for 202412), as stored: float16."""
    z = (np.asarray(vectors, dtype=np.float32) - eyes[f"mean_{READER}"]) / eyes[
        f"spread_{READER}"
    ]
    return z.astype(np.float16)


def member_inputs(
    articles: list[dict],
    pairs: list[tuple[int, str, str]],
    day: date,
    vectors: dict[int, np.ndarray],
    most: int = MAX_HEADLINES,
) -> tuple[list[str], dict[str, np.ndarray], np.ndarray]:
    """The day's member-days: their tickers, the network's inputs (heads into the day's own
    store, their info, log(1 + count)) and that store, in N6's scale."""
    by_id = {a["id"]: a for a in articles}
    end = datetime.combine(day, time(15, 30), tzinfo=NEW_YORK).astimezone(UTC)
    per = defaultdict(list)
    for (
        article,
        ticker,
        _,
    ) in pairs:  # counted() lists them in the order they became known
        per[ticker].append(article)
    tickers = sorted(per)
    ids = sorted({i for t in tickers for i in per[t][-most:]})
    row = {i: k for k, i in enumerate(ids)}
    store = np.array([vectors[i] for i in ids], dtype=np.float16).reshape(len(ids), -1)
    heads = np.full((len(tickers), most), -1, dtype=np.int32)
    info = np.zeros((len(tickers), most, 3), dtype=np.float32)
    for n, t in enumerate(tickers):
        for k, i in enumerate(per[t][-most:]):
            a = by_id[i]
            seen = fr.known(a)
            heads[n, k] = row[i]
            info[n, k] = (
                np.log1p((end - seen).total_seconds() / 3600),
                float(seen != fr.stamp(a["created_at"])),
                float(len(a["symbols"]) == 2),
            )
    count = np.log1p([len(per[t]) for t in tickers]).astype(np.float32)
    return tickers, {"heads": heads, "info": info, "count": count}, store


def forecast(models, store: np.ndarray, x: dict) -> np.ndarray:
    """The frozen networks' mean forecast for the day's member-days (on the CPU)."""
    import n6_freeze
    import torch

    tensors = {
        "heads": torch.as_tensor(x["heads"]),
        "info": torch.as_tensor(x["info"]),
        "count": torch.as_tensor(x["count"]),
    }
    return n6_freeze.forecast(models, torch.as_tensor(store), tensors)


def frozen(card) -> list:
    """The card's models, after checking their bytes and the eyes they read with."""
    import n6_freeze

    with np.load(weights_path(card), allow_pickle=False) as kept:
        arrays = dict(kept)
    if str(arrays["eyes_sha256"]) != sha256(EYES):
        raise SystemExit(f"{EYES} is not the eyes N6 was frozen with")
    return n6_freeze.models_from(arrays)


def run(
    card_path: Path = CARD, now: datetime | None = None, embedder=fr.embed
) -> pl.DataFrame:
    """Score each session from the card's first day not scored yet, in order, and keep it.
    Returns every day so far. The captures must have been copied from S3 first."""
    now = now or datetime.now(UTC)
    card = load_card(card_path)
    if card.test() != TEST:
        raise SystemExit(f"{card_path}'s test is not pre-registration N6's: {TEST}")
    if Path(card.weights) != WEIGHTS:
        raise SystemExit(f"{card_path} names {card.weights}, not {WEIGHTS}")
    models = frozen(card)
    with np.load(EYES) as kept:
        eyes = {k: kept[k] for k in (f"mean_{READER}", f"spread_{READER}")}
    import harness

    prices = pl.read_parquet(fr.PRICES, columns=["date", "ticker", "close", "sp500"])
    store_path = OUT / "days.parquet"
    done = (
        pl.read_parquet(store_path)
        if store_path.exists()
        else pl.DataFrame(schema=fr.DAYS)
    )
    scored = set(done["day"].to_list())
    days = [
        s.day
        for s in sessions(card.record_from, max(card.record_from, now.date()))
        if s.day not in scored and fr.capture_ended(s.day, now)
    ]
    pending, rows = [], []
    for day in days:
        path = fr.CAPTURES / f"{day.isoformat()}.jsonl"
        if not path.exists():
            rows.append({"day": day, "why": "no capture"})
            continue
        returns = fr.next_returns(prices, day)
        if returns is None:
            break  # the next close is not in the prices yet: later days wait too
        articles, moment = fr.read_capture(path, day)
        pairs = fr.counted(articles, day, fr.members_on(prices, day))
        pending.append((day, moment, articles, pairs, returns))
    texts = {a: h for *_, pairs, _ in pending for a, _, h in pairs}
    ids = sorted(texts)
    read = embedder([texts[i] for i in ids]) if ids else np.zeros((0, 1))
    vectors = dict(zip(ids, scaled(read, eyes), strict=True)) if ids else {}
    for day, moment, articles, pairs, returns in pending:
        row = {"day": day, "captured_at": moment, "articles": len(pairs)}
        if pairs:
            tickers, x, store = member_inputs(articles, pairs, day, vectors)
            row |= fr.score_day(
                tickers, forecast(models, store, x), returns, LEVERS, harness
            )
        else:
            row |= {"members": 0, "why": "no member news"}
        rows.append(row)
    if not rows:
        return done
    for row in rows:
        row |= {"season": fr.earnings_season(row["day"]), "scored_at": now}
    new = pl.DataFrame(
        [{key: row.get(key) for key in fr.DAYS} for row in rows], schema=fr.DAYS
    )
    all_days = pl.concat([done, new]).sort("day")
    OUT.mkdir(parents=True, exist_ok=True)
    all_days.write_parquet(store_path)
    return all_days


def evidence(days: pl.DataFrame) -> Evidence:
    """The betting test on the record's scored days, in order, against no trade."""
    scores = days.filter(pl.col("score").is_not_null()).sort("day")["score"].to_numpy()
    return TEST.run(scores, np.zeros(len(scores)))


def against_n5(
    days: pl.DataFrame, n5: pl.DataFrame | None
) -> tuple[float, float | None, int]:
    """N6 less N5's frozen N4, bps a day and its t, on the sessions both scored."""
    if n5 is None or n5.is_empty():
        return 0.0, None, 0
    both = (
        days.filter(pl.col("score").is_not_null())
        .select("day", "score")
        .join(
            n5.filter(pl.col("score").is_not_null()).select(
                "day", pl.col("score").alias("n5")
            ),
            on="day",
        )
    )
    if both.is_empty():
        return 0.0, None, 0
    bps, t = fr.mean_t((both["score"] - both["n5"]).to_numpy())
    return bps, t, both.height


def page(card, days: pl.DataFrame, result: Evidence, now: datetime, n5=None) -> str:
    """The evening page: the test, N6 against N5, the sides, the months and every session."""
    scored = days.filter(pl.col("score").is_not_null()).sort("day")
    wealth = list(result.wealth)
    clipped = {scored["day"][i] for i in result.clipped}
    missing = days.filter(pl.col("why") == "no capture")["day"].to_list()
    verdict = (
        f"promoted on {scored['day'][result.promoted_on]}"
        if result.promoted
        else "not promoted"
    )
    gap, gap_t, both = against_n5(days, n5)
    summary = [
        (
            f"The record from {card.record_from}: {days.height} sessions, {scored.height} "
            f"scored, {days.height - scored.height - len(missing)} without a trade, "
            f"{len(missing)} without a capture."
        ),
        (
            f"Wealth {wealth[-1] if wealth else 1.0:.3f} against {result.threshold:g}: "
            f"{verdict}. Days beyond the clip ({TEST.clip:.1%}): {len(clipped)}."
        ),
        (
            f"N6 less N5's frozen N4 on the {both} sessions both scored: {signed(gap)} bps a "
            f"day{'' if gap_t is None else f' (t {signed(gap_t)})'}."
        ),
        (
            f"Frozen {card.frozen_at:%Y-%m-%d %H:%M} UTC, weights {card.weights_sha256[:12]}"
            f"…, trained {card.trained_from} to {card.trained_to}; reviewed on 31 "
            "December 2028."
        ),
    ]
    sides = []
    if scored.height:
        long, short, everyone = (
            scored[c].to_numpy() for c in ("long", "short", "everyone")
        )
        score = scored["score"].to_numpy()
        season = scored["season"].to_numpy()
        for label, values in (
            ("After costs", score),
            ("Before costs", long - short),
            ("Long side, against the mean", long - everyone),
            ("Short side, against the mean", everyone - short),
            ("Earnings season", score[season]),
            ("Other days", score[~season]),
        ):
            sides.append(fr.side_row(label, values))
    months = (
        scored.with_columns(pl.col("day").dt.strftime("%Y-%m").alias("month"))
        .group_by("month")
        .agg(pl.len().alias("days"), (pl.col("score").mean() * 1e4).alias("bps"))
        .sort("month")
    )
    scale = max([abs(b) for b in months["bps"]] + [5.0])
    month_rows = "".join(
        f'<tr><td>{m}</td>{_cell(b, scale)}<td class="num">{n}</td></tr>'
        for m, n, b in months.iter_rows()
    )
    by_day = dict(zip(scored["day"].to_list(), wealth, strict=True))
    day_rows = "".join(
        fr.day_row(r, by_day.get(r["day"]), clipped)
        for r in days.sort("day", descending=True).iter_rows(named=True)
    )
    body = f"""
<h2>Wealth (log scale)</h2>
{fr.wealth_chart(scored["day"].to_list(), wealth, result.threshold)}
<p class="note">The betting test against no trade, from 1: each day's score after costs,
clipped at {TEST.clip:.1%}, is bet on with a stake set from the days before. Without an edge
the wealth reaches {result.threshold:g} with a chance of at most {TEST.alpha:.3%}, however
often it is read.</p>
<h2>Beside the test (bps a day; t)</h2>
<div class="tables"><table><thead><tr><th></th><th class="num">Bps a day</th>
<th class="num">t</th><th class="num">Days</th></tr></thead><tbody>{"".join(sides)}</tbody>
</table></div>
<h2>By month (bps a day after costs)</h2>
<div class="tables"><table><thead><tr><th>Month</th><th class="num">Bps a day</th>
<th class="num">Days</th></tr></thead><tbody>{month_rows}</tbody></table></div>
<h2>Every session</h2>
<div class="tables"><table><thead><tr><th>Day</th><th class="num">Members</th>
<th class="num">Long</th><th class="num">Short</th><th class="num">Bps</th>
<th class="num">Wealth</th><th>Note</th></tr></thead><tbody>{day_rows}</tbody></table></div>
"""
    out = PAGE.format(
        title="N6 record",
        heading="N6: our own network, frozen, forward",
        updated=now.strftime("%Y-%m-%d %H:%M UTC"),
        summary="".join(f"<li>{html.escape(s)}</li>" for s in summary),
        chart_title="",
        chart="",
        table_title="",
        head="",
        rows="",
        note="",
    )
    start, end = out.index("<h2></h2>"), out.index("</main>")
    return out[:start] + body + out[end:]


def main() -> None:
    fr.sync()
    now = datetime.now(UTC)
    days = run(now=now)
    if days.is_empty():
        print("no session of N6's record has ended yet")
        return
    result = evidence(days)
    n5 = pl.read_parquet(N5_DAYS) if N5_DAYS.exists() else None
    (OUT / "record.html").write_text(page(load_card(CARD), days, result, now, n5))
    last = days.sort("day").row(-1, named=True)
    if last["score"] is None:
        print(f"{last['day']}: no score ({last['why']})")
    else:
        print(f"{last['day']}: {signed(last['score'] * 1e4)} bps after costs")
    wealth = result.wealth[-1] if result.wealth else 1.0
    print(
        f"{days.height} sessions, {len(result.wealth)} scored; wealth {wealth:.3f} of "
        f"{result.threshold:g}; the page: {OUT / 'record.html'}"
    )


if __name__ == "__main__":
    main()
