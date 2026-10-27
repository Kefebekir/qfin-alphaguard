"""N5's record, day by day: the frozen N4 forward, from the headlines captured at 15:31.

Each session captured by `qfin news-capture` is scored once the next session's close is known:
- the articles that count, by N2's rule: known from 16:00 on the session before to 15:30 (an
  article edited more than five minutes after it was created is known from its edit),
  tagging one or two stocks, each for the S&P 500 member it names that day;
- each headline read by PIT-4B-202412 (its first 64 tokens, the last token's state, as N3);
- each member-day's mean embedding and log(1 + count), the frozen model's forecast
  (forward.py's arrays), N4's variant 2's positions (a tenth a side, ties in, equal weights,
  no trade under 20 members), and the return from the day's close to the next, less 2 bps.

A session with no capture has no score and is listed as such; it is never filled in. A day
already scored is not scored again. The evidence is decision 0020's betting test against no
trade, as the card challengers/news-n4-forward.yaml and pre-registration N5 set it: clip
0.085, alpha 0.0125, promoted when the wealth reaches 80.

    PYTHONPATH=src data/llmenv/bin/python research/news_reader/forward_record.py

It first copies the captures and the nightly job's prices from S3 (so that a capture not yet
copied is never taken for a missing one), then scores the days due, and writes
data/news_reader/forward/days.parquet and record.html. The prices are kept apart from the
research's, which stay as the holdouts read them. No headline is printed or kept.
"""

import html
import json
import re
import subprocess
import sys
from collections import defaultdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from dev_report import PAGE, _cell, signed
from members import normal

from qfin_alphaguard.challengers import load_card, weights_path
from qfin_alphaguard.data.news_capture import EARLIEST, LATEST, NEW_YORK, window
from qfin_alphaguard.evidence import BettingTest, Evidence
from qfin_alphaguard.sessions import sessions

BUCKET = "s3://qfin-alphaguard-kefebekir"
CAPTURES = Path("data/news_capture")
OUT = Path("data/news_reader/forward")
PRICES = OUT / "prices.parquet"
READER = Path("data/news_reader/pit/PIT-4B-202412")
WEIGHTS = Path("data/models/news-n4-forward.npz")
CARD = Path("challengers/news-n4-forward.yaml")
EDIT = timedelta(minutes=5)
MAX_TOKENS = 64
COST = 0.0001  # a round trip, each side
MIN_MEMBERS = 20
TEST = BettingTest(scale=1.0, clip=0.085, alpha=0.0125)  # pre-registration N5
DAYS = {
    "day": pl.Date,
    "season": pl.Boolean,
    "captured_at": pl.Datetime("us", "UTC"),
    "articles": pl.Int64,
    "members": pl.Int64,
    "n_long": pl.Int64,
    "n_short": pl.Int64,
    "long": pl.Float64,
    "short": pl.Float64,
    "everyone": pl.Float64,
    "score": pl.Float64,
    "why": pl.String,
    "scored_at": pl.Datetime("us", "UTC"),
}


def stamp(text: str) -> datetime:
    return datetime.fromisoformat(text)


def known(article: dict) -> datetime:
    """When an article counts from: its edit if made more than five minutes after it was
    created, else its creation (N2's rule)."""
    created, updated = stamp(article["created_at"]), stamp(article["updated_at"])
    return updated if updated - created > EDIT else created


def members_on(prices: pl.DataFrame, day: date) -> dict[str, list[str]]:
    """The day's S&P 500 members by the symbol they answer to: a ticker and its _old."""
    found = defaultdict(list)
    tickers = prices.filter((pl.col("date") == day) & pl.col("sp500"))["ticker"]
    for ticker in tickers.to_list():
        found[re.sub(r"_old\d*$", "", ticker)].append(ticker)
    return found


def counted(
    articles: list[dict], day: date, members: dict
) -> list[tuple[int, str, str]]:
    """(id, ticker, headline) for each article counting for a member's session `day`."""
    start, end = window(day)
    out = []
    for article in sorted(articles, key=lambda a: (known(a), a["id"])):
        if not start < known(article) <= end:
            continue
        tags = article["symbols"]
        if not 1 <= len(tags) <= 2:
            continue
        for tag in tags:
            fits = members.get(normal(tag), [])
            if len(fits) == 1:
                out.append((article["id"], fits[0], article["headline"]))
    return out


def member_days(pairs: list[tuple[int, str, str]], vectors: dict[int, np.ndarray]):
    """The day's member-days: tickers, counts and X (mean embedding, then log(1 + count)),
    each pair counted, as N2 counts an article naming one member twice."""
    sums, counts = {}, defaultdict(int)
    for article, ticker, _ in pairs:
        sums[ticker] = sums.get(ticker, 0) + vectors[article].astype(np.float32)
        counts[ticker] += 1
    tickers = sorted(sums)
    x = np.array([np.append(sums[t] / counts[t], np.log1p(counts[t])) for t in tickers])
    return tickers, [counts[t] for t in tickers], x


def next_returns(prices: pl.DataFrame, day: date) -> dict[str, float] | None:
    """Each ticker's return from `day`'s close to the next session's; None until that close
    is in the prices."""
    following = [
        s.day for s in sessions(day + timedelta(days=1), day + timedelta(days=10))
    ]
    nxt = following[0]
    two = prices.filter(pl.col("date").is_in([day, nxt])).pivot(
        on="date", index="ticker", values="close"
    )
    if str(nxt) not in two.columns or str(day) not in two.columns:
        return None
    two = two.drop_nulls()
    return dict(
        zip(two["ticker"], (two[str(nxt)] / two[str(day)] - 1).to_list(), strict=True)
    )


def positions(forecasts: np.ndarray, levers: dict, harness) -> np.ndarray | None:
    """N4's variant 2's positions for the day (harness.n4_weights). Its levers need no
    volatility or sector, so none is given; other levers are refused."""
    if levers["weights"] != "equal" or levers["sectors"] != "none":
        raise ValueError(f"the record has no volatilities or sectors for {levers}")
    ones = np.ones(len(forecasts))
    return harness.n4_weights(forecasts, ones, np.zeros(len(forecasts), int), levers)


def score_day(
    tickers: list[str],
    forecasts: np.ndarray,
    returns: dict[str, float],
    levers,
    harness,
) -> dict:
    """The day's long-short: positions over the members with a next close."""
    have = [i for i, t in enumerate(tickers) if t in returns]
    row = {"members": len(have), "long": None, "short": None, "score": None}
    if len(have) < MIN_MEMBERS:
        return row | {"why": f"{len(have)} members with news and a next close"}
    f = forecasts[have]
    r = np.array([returns[tickers[i]] for i in have])
    w = positions(f, levers, harness)
    if w is None:
        return row | {"why": "no forecast separates the sides"}
    long, short = w > 0, w < 0
    return row | {
        "long": float(w[long] @ r[long]),
        "short": float(-w[short] @ r[short]),
        "everyone": float(r.mean()),
        "score": float(w @ r) - 2 * COST,
        "n_long": int(long.sum()),
        "n_short": int(short.sum()),
        "why": "",
    }


def earnings_season(day: date) -> bool:
    """From the 15th of January, April, July and October to the last day of the month after
    (pre-registration N5's definition)."""
    return (day.month in (1, 4, 7, 10) and day.day >= 15) or day.month in (2, 5, 8, 11)


def embed(headlines: list[str]) -> np.ndarray:
    """PIT-4B-202412's last-token states, as embed_pit.py reads a headline (GPU)."""
    import torch
    from pit import SHAPE, HiddenStates, batches, check, embeddings, load
    from tokenizers import Tokenizer

    check(json.loads((READER / "config.json").read_text()), SHAPE)
    tokenizer = Tokenizer.from_file(str(READER / "tokenizer.json"))
    encoded = [e.ids[:MAX_TOKENS] for e in tokenizer.encode_batch(headlines)]
    lengths = np.array([max(len(e), 1) for e in encoded])
    wte, blocks = load(READER / "model.safetensors")
    states = HiddenStates(wte, blocks)
    out = np.empty((len(headlines), SHAPE.width), dtype=np.float16)
    for rows in batches(np.argsort(lengths, kind="stable"), lengths, 16_384):
        batch = torch.zeros((len(rows), int(lengths[rows].max())), dtype=torch.long)
        for i, row in enumerate(rows):
            batch[i, : len(encoded[row] or [50256])] = torch.tensor(
                encoded[row] or [50256]
            )
        hidden = states(batch)
        last, _ = embeddings(
            hidden, torch.as_tensor(lengths[rows], device=hidden.device)
        )
        out[rows] = last.cpu().numpy().astype(np.float16)
    return out


def read_capture(path: Path, day: date) -> tuple[list[dict], datetime | None]:
    """A capture's articles and the moment it was taken, which must be on `day` between
    15:30 and 16:30 New York time, as `qfin news-capture` takes it."""
    articles = [json.loads(line) for line in path.read_text().splitlines() if line]
    moments = {a["captured_at"] for a in articles}
    if len(moments) > 1:
        raise SystemExit(f"{path}: more than one capture time")
    if not moments:
        return articles, None
    moment = stamp(moments.pop())
    local = moment.astimezone(NEW_YORK)
    if local.date() != day or not EARLIEST <= local.time() <= LATEST:
        raise SystemExit(f"{path} was captured at {local:%Y-%m-%d %H:%M} New York time")
    return articles, moment


def capture_ended(day: date, now: datetime) -> bool:
    """Whether the day's capture could still run: until 16:30 New York time."""
    return now.astimezone(NEW_YORK) > datetime.combine(day, LATEST, tzinfo=NEW_YORK)


def run(
    harness,
    forward,
    card_path: Path = CARD,
    now: datetime | None = None,
    embedder=embed,
) -> pl.DataFrame:
    """Score each session from the card's first day not scored yet, in order, and keep it.
    Returns every day so far. The captures must have been copied from S3 first."""
    now = now or datetime.now(UTC)
    card = load_card(card_path)
    if card.test() != TEST:
        raise SystemExit(f"{card_path}'s test is not pre-registration N5's: {TEST}")
    if Path(card.weights) != WEIGHTS:
        raise SystemExit(f"{card_path} names {card.weights}, not {WEIGHTS}")
    arrays = forward.load(weights_path(card))
    chosen = json.loads((harness.VARIANTS / "N4_chosen.json").read_text())
    levers = {lever: chosen[lever] for lever in harness.N4_LEVERS}
    prices = pl.read_parquet(PRICES, columns=["date", "ticker", "close", "sp500"])
    store = OUT / "days.parquet"
    done = pl.read_parquet(store) if store.exists() else pl.DataFrame(schema=DAYS)
    scored = set(done["day"].to_list())
    days = [
        s.day
        for s in sessions(card.record_from, max(card.record_from, now.date()))
        if s.day not in scored and capture_ended(s.day, now)
    ]
    pending, rows = [], []
    for day in days:
        path = CAPTURES / f"{day.isoformat()}.jsonl"
        if not path.exists():
            rows.append({"day": day, "why": "no capture"})
            continue
        returns = next_returns(prices, day)
        if returns is None:
            break  # the next close is not in the prices yet: later days wait too
        articles, moment = read_capture(path, day)
        pairs = counted(articles, day, members_on(prices, day))
        pending.append((day, moment, pairs, returns))
    texts = {a: h for _, _, pairs, _ in pending for a, _, h in pairs}
    ids = sorted(texts)
    vectors = (
        dict(zip(ids, embedder([texts[i] for i in ids]), strict=True)) if ids else {}
    )
    for day, moment, pairs, returns in pending:
        row = {"day": day, "captured_at": moment, "articles": len(pairs)}
        if pairs:
            tickers, _, x = member_days(pairs, vectors)
            row |= score_day(
                tickers, forward.forecast(arrays, x), returns, levers, harness
            )
        else:
            row |= {"members": 0, "why": "no member news"}
        rows.append(row)
    if not rows:
        return done
    for row in rows:
        row |= {"season": earnings_season(row["day"]), "scored_at": now}
    new = pl.DataFrame(
        [{key: row.get(key) for key in DAYS} for row in rows], schema=DAYS
    )
    all_days = pl.concat([done, new]).sort("day")
    OUT.mkdir(parents=True, exist_ok=True)
    all_days.write_parquet(store)
    return all_days


def evidence(days: pl.DataFrame) -> Evidence:
    """The betting test on the record's scored days, in order, against no trade."""
    scores = days.filter(pl.col("score").is_not_null()).sort("day")["score"].to_numpy()
    return TEST.run(scores, np.zeros(len(scores)))


def mean_t(values: np.ndarray) -> tuple[float, float | None]:
    """Bps a day and its t; no t under two days."""
    v = np.asarray(values, dtype=float)
    if len(v) < 2 or v.std(ddof=1) == 0:
        return float(v.mean() * 1e4) if len(v) else 0.0, None
    return float(v.mean() * 1e4), float(v.mean() / v.std(ddof=1) * np.sqrt(len(v)))


def season_difference(scored: pl.DataFrame) -> tuple[float, float | None]:
    """Earnings-season days less the others, bps a day, with Welch's t."""
    a = scored.filter(pl.col("season"))["score"].to_numpy()
    b = scored.filter(~pl.col("season"))["score"].to_numpy()
    if len(a) < 2 or len(b) < 2:
        return 0.0, None
    gap = a.mean() - b.mean()
    spread = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float(gap * 1e4), (float(gap / spread) if spread > 0 else None)


def _t(value: float | None) -> str:
    return "—" if value is None else signed(value)


def wealth_chart(days: list[date], wealth: list[float], threshold: float) -> str:
    """The wealth after each scored day on a log scale, from 1 to the threshold."""
    width, height, left, right, top, bottom = 900, 260, 48, 16, 12, 28
    low = min([0.5, *wealth])
    high = max([threshold * 1.25, *wealth])
    span = np.log(high) - np.log(low)

    def y(w: float) -> float:
        return top + (np.log(high) - np.log(w)) / span * (height - top - bottom)

    def x(i: int) -> float:
        n = max(len(wealth), 2) - 1
        return left + i / n * (width - left - right)

    ticks = [t for t in (0.25, 0.5, 1, 2, 5, 10, 20, 40, 80, 160) if low <= t <= high]
    parts = [
        f'<line x1="{left}" x2="{width - right}" y1="{y(t):.1f}" y2="{y(t):.1f}" '
        f'stroke="var(--grid)" stroke-width="1"/>'
        f'<text class="axis" x="{left - 6}" y="{y(t) + 4:.1f}" text-anchor="end">{t:g}</text>'
        for t in ticks
    ]
    parts.append(
        f'<line x1="{left}" x2="{width - right}" y1="{y(threshold):.1f}" '
        f'y2="{y(threshold):.1f}" stroke="var(--text-secondary)" stroke-width="1.5" '
        f'stroke-dasharray="6 4"/><text class="label" x="{width - right}" '
        f'y="{y(threshold) - 6:.1f}" text-anchor="end">promoted at {threshold:g}</text>'
    )
    parts.append(
        f'<line x1="{left}" x2="{width - right}" y1="{y(1):.1f}" y2="{y(1):.1f}" '
        f'stroke="var(--baseline)" stroke-width="1"/>'
    )
    if wealth:
        points = " ".join(f"{x(i):.1f},{y(w):.1f}" for i, w in enumerate(wealth))
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="var(--series-1)" '
            f'stroke-width="2" stroke-linejoin="round"/>'
        )
        parts += [
            f'<g class="bar"><circle cx="{x(i):.1f}" cy="{y(w):.1f}" r="8" '
            f'fill="transparent"/><circle cx="{x(i):.1f}" cy="{y(w):.1f}" r="3" '
            f'fill="var(--series-1)"/><title>{d}: wealth {w:.3f}</title></g>'
            for i, (d, w) in enumerate(zip(days, wealth, strict=True))
        ]
        parts.append(
            f'<text class="axis" x="{left}" y="{height - 8}">{days[0]}</text>'
            f'<text class="axis" x="{width - right}" y="{height - 8}" '
            f'text-anchor="end">{days[-1]}</text>'
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Wealth after each '
        f'scored day against the threshold">{"".join(parts)}</svg>'
    )


def record_page(card, days: pl.DataFrame, result: Evidence, now: datetime) -> str:
    """The evening page: the test, the sides, the seasons, the months and every day."""
    scored = days.filter(pl.col("score").is_not_null()).sort("day")
    wealth = list(result.wealth)
    clipped = {scored["day"][i] for i in result.clipped}
    missing = days.filter(pl.col("why") == "no capture")["day"].to_list()
    if result.promoted:
        verdict = f"promoted on {scored['day'][result.promoted_on]}"
    else:
        verdict = "not promoted"
    no_trade = days.height - scored.height - len(missing)
    summary = [
        (
            f"The record from {card.record_from}: {days.height} sessions, "
            f"{scored.height} scored, {no_trade} without a trade, {len(missing)} "
            "without a capture."
        ),
        (
            f"Wealth {wealth[-1] if wealth else 1.0:.3f} against "
            f"{result.threshold:g}: {verdict}. Days beyond the clip "
            f"({TEST.clip:.1%}): {len(clipped)}."
        ),
        (
            f"Frozen {card.frozen_at:%Y-%m-%d %H:%M} UTC, weights "
            f"{card.weights_sha256[:12]}\u2026, trained {card.trained_from} to "
            f"{card.trained_to}; reviewed on 31 December 2028."
        ),
    ]
    sides = []
    if scored.height:
        long, short, everyone = (
            scored[c].to_numpy() for c in ("long", "short", "everyone")
        )
        season = scored["season"].to_numpy()
        score = scored["score"].to_numpy()
        for label, values in (
            ("After costs", score),
            ("Before costs", long - short),
            ("Long side, against the mean", long - everyone),
            ("Short side, against the mean", everyone - short),
            ("Earnings season", score[season]),
            ("Other days", score[~season]),
        ):
            sides.append(side_row(label, values))
        gap, t = season_difference(scored)
        sides.append(
            f'<tr><td>Season less the others (Welch)</td><td class="num">{signed(gap)}</td>'
            f'<td class="num">{_t(t)}</td><td class="num"></td></tr>'
        )
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
        day_row(r, by_day.get(r["day"]), clipped)
        for r in days.sort("day", descending=True).iter_rows(named=True)
    )
    body = f"""
<h2>Wealth (log scale)</h2>
{wealth_chart(scored["day"].to_list(), wealth, result.threshold)}
<p class="note">The betting test against no trade, from 1: each day's score after costs,
clipped at {TEST.clip:.1%}, is bet on with a stake set from the days before. Without an edge
the wealth reaches {result.threshold:g} with a chance of at most {TEST.alpha:.2%}, however
often it is read.</p>
<h2>Beside the test (bps a day; t)</h2>
<div class="tables"><table><thead><tr><th></th><th class="num">Bps a day</th>
<th class="num">t</th><th class="num">Days</th></tr></thead><tbody>{"".join(sides)}</tbody>
</table></div>
<p class="note">Earnings season: from the 15th of January, April, July and October to the
last day of the month after (N4's 2026 observation, reported, not tested).</p>
<h2>By month (bps a day after costs)</h2>
<div class="tables"><table><thead><tr><th>Month</th><th class="num">Bps a day</th>
<th class="num">Days</th></tr></thead><tbody>{month_rows}</tbody></table></div>
<h2>Every session</h2>
<div class="tables"><table><thead><tr><th>Day</th><th class="num">Members</th>
<th class="num">Long</th><th class="num">Short</th><th class="num">Bps</th>
<th class="num">Wealth</th><th>Note</th></tr></thead><tbody>{day_rows}</tbody></table></div>
"""
    page = PAGE.format(
        title="N5 record",
        heading="N5: N4's variant 2, frozen, forward",
        updated=now.strftime("%Y-%m-%d %H:%M UTC"),
        summary="".join(f"<li>{html.escape(s)}</li>" for s in summary),
        chart_title="",
        chart="",
        table_title="",
        head="",
        rows="",
        note="",
    )
    start = page.index("<h2></h2>")
    end = page.index("</main>")
    return page[:start] + body + page[end:]


def side_row(label: str, values: np.ndarray) -> str:
    if not len(values):
        dash = '<td class="num">\u2014</td>'
        return f'<tr><td>{label}</td>{dash}{dash}<td class="num">0</td></tr>'
    m, t = mean_t(values)
    return (
        f'<tr><td>{label}</td><td class="num">{signed(m)}</td>'
        f'<td class="num">{_t(t)}</td><td class="num">{len(values)}</td></tr>'
    )


def day_row(row: dict, wealth: float | None, clipped: set) -> str:
    """One session: its members, sides, score and wealth, and why it has none."""

    def number(value) -> str:
        return f'<td class="num">{"" if value is None else value}</td>'

    score = (
        '<td class="num"></td>'
        if row["score"] is None
        else _cell(row["score"] * 1e4, 100.0)
    )
    return (
        f"<tr><td>{row['day']}</td>"
        + number(row["members"])
        + number(row["n_long"])
        + number(row["n_short"])
        + score
        + number(None if wealth is None else f"{wealth:.3f}")
        + f"<td>{html.escape(note(row, clipped))}</td></tr>"
    )


def note(row: dict, clipped: set) -> str:
    parts = [row["why"]] if row["why"] else []
    if row["day"] in clipped:
        parts.append("beyond the clip")
    if row["season"]:
        parts.append("earnings season")
    return "; ".join(parts)


def sync() -> None:
    """The captures and the nightly job's prices, from S3."""
    for command in (
        ["sync", f"{BUCKET}/data/news_capture", str(CAPTURES)],
        ["cp", f"{BUCKET}/data/raw/prices.parquet", str(PRICES)],
    ):
        subprocess.run(
            ["aws", "s3", *command, "--profile", "qfin", "--region", "eu-north-1"]
            + ["--only-show-errors"],
            check=True,
        )


def main() -> None:
    import forward
    import harness

    sync()
    now = datetime.now(UTC)
    days = run(harness, forward, now=now)
    if days.is_empty():
        print("no session of the record has ended yet")
        return
    result = evidence(days)
    (OUT / "record.html").write_text(record_page(load_card(CARD), days, result, now))
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
