"""After a news headline, does a large stock's move carry on? An upper bound for a news reader.

Alpaca's news API serves Benzinga's articles since 2015 with their publication times (to the
second) and the stocks they tag. For every stock with stored minutes (the trading universes'
large stocks), this takes the articles published in regular hours between 09:45 and 15:00 New
York time that tag at most two stocks, the first for a stock in any 60 minutes, and measures
on the stock's minute closes:

- the reaction: from the last minute closed before the article to 1 and 5 minutes after it;
- what follows: from there to 30 and 60 minutes after the article, and to the close.

If the move after the reaction carries on in the reaction's direction, a reader who acts within
a minute or five could earn it; if it reverses or stops, reading the news fast earns nothing
at our speed. The same numbers at times with no article for the stock (a random minute of each
day without one, between 09:45 and 15:00) are the comparison: intraday moves carry on or
reverse a little anyway.

Only the times are used here, not the text: what a perfect reader of direction could have
earned after the price had told it. The articles are cached under OUT_DIR (Benzinga's text
stays in the git-ignored data/).

Usage: news_reaction.py ROOT OUT_DIR   (with ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY set)
"""

import json
import math
import os
import random
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
NEW_YORK = ZoneInfo("America/New_York")
ETFS = {
    "QQQ",
    "SPY",
    "IWM",
    "DIA",
    "TLT",
    "GLD",
    "HYG",
    "XLB",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
}
ETFS |= {"XLP", "XLU", "XLV", "XLY"}
_lock, _last_request = threading.Lock(), [0.0]
HEADERS = {
    "APCA-API-KEY-ID": os.environ["ALPACA_API_KEY_ID"],
    "APCA-API-SECRET-KEY": os.environ["ALPACA_API_SECRET_KEY"],
}


def news(ticker: str, start: str, end: str) -> list[dict]:
    """Every article tagging `ticker` from `start` to `end`, cached as JSON lines."""
    path = OUT / "news" / f"{ticker}.jsonl"
    if path.exists():
        return [json.loads(line) for line in path.read_text().splitlines()]
    articles, token = [], None
    while True:
        query = {
            "symbols": ticker,
            "start": start,
            "end": end,
            "limit": 50,
            "sort": "asc",
            "include_content": "false",
        }
        if token:
            query["page_token"] = token
        url = "https://data.alpaca.markets/v1beta1/news?" + urllib.parse.urlencode(
            query
        )
        for attempt in range(6):
            with (
                _lock
            ):  # 0.34 s apart across threads: Alpaca's free plan allows 200 a minute
                wait = 0.34 - (time.monotonic() - _last_request[0])
                if wait > 0:
                    time.sleep(wait)
                _last_request[0] = time.monotonic()
            try:
                request = urllib.request.Request(url, headers=HEADERS)
                with urllib.request.urlopen(request, timeout=30) as response:
                    page = json.loads(response.read())
                break
            except (OSError, ValueError) as error:  # network errors, rate limits
                if attempt == 5:
                    raise RuntimeError(f"news for {ticker}: {error}") from None
                time.sleep(5 * 2**attempt)
        articles += [
            {
                "id": a["id"],
                "created_at": a["created_at"],
                "updated_at": a["updated_at"],
                "symbols": a["symbols"],
                "headline": a["headline"],
            }
            for a in page["news"]
        ]
        token = page.get("next_page_token")
        if not token:
            break
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(a) + "\n" for a in articles))
    return articles


def minutes(ticker: str) -> pl.DataFrame:
    files = sorted((ROOT / "intraday" / ticker).glob("*.parquet"))
    return pl.concat(
        [pl.read_parquet(f, columns=["start", "close"]) for f in files]
    ).sort("start")


def measure(frame: pl.DataFrame, moments: list[datetime]) -> list[dict]:
    """The reaction and what follows for each moment, from the stock's minute closes."""
    starts = frame["start"].to_numpy()
    closes = frame["close"].to_numpy()
    out = []
    for moment in moments:
        minute = moment.replace(second=0, microsecond=0)
        local = moment.astimezone(NEW_YORK)
        close_time = local.replace(
            hour=15, minute=59, second=0, microsecond=0
        ).astimezone(UTC)

        def price_at(t):  # the close of the last minute bar starting at or before t
            i = np.searchsorted(
                starts, np.datetime64(t.replace(tzinfo=None), "us"), "right"
            )
            return closes[i - 1] if i > 0 else np.nan

        before = price_at(minute - timedelta(minutes=1))
        if not before > 0:
            continue
        p = {k: price_at(minute + timedelta(minutes=k)) for k in (1, 5, 30, 60)}
        p["close"] = price_at(close_time)
        out.append(
            {
                "react_1": p[1] / before - 1,
                "react_5": p[5] / before - 1,
                "after_1_30": p[30] / p[1] - 1,
                "after_1_60": p[60] / p[1] - 1,
                "after_5_60": p[60] / p[5] - 1,
                "after_5_close": p["close"] / p[5] - 1,
            }
        )
    return out


rng = random.Random(0)
news_rows, quiet_rows = [], []
stocks = sorted(p.name for p in (ROOT / "intraday").iterdir() if p.name not in ETFS)


def span(ticker: str) -> tuple[str, str]:
    files = sorted((ROOT / "intraday" / ticker).glob("*.parquet"))
    first = pl.read_parquet(files[0], columns=["start"])["start"].min()
    last = pl.read_parquet(files[-1], columns=["start"])["start"].max()
    return first.strftime("%Y-%m-%dT00:00:00Z"), last.strftime("%Y-%m-%dT23:59:59Z")


# Download first, three stocks at a time; the measuring below reads the cache.
with ThreadPoolExecutor(max_workers=3) as pool:
    for done, _ in enumerate(pool.map(lambda t: news(t, *span(t)), stocks), 1):
        if done % 10 == 0:
            print(f"   news for {done}/{len(stocks)} stocks", flush=True)
for number, ticker in enumerate(stocks, 1):
    frame = minutes(ticker)
    first, last = frame["start"].min(), frame["start"].max()
    articles = news(
        ticker,
        first.strftime("%Y-%m-%dT00:00:00Z"),
        last.strftime("%Y-%m-%dT23:59:59Z"),
    )
    trading_days = set(
        frame["start"].dt.convert_time_zone("America/New_York").dt.date().to_list()
    )
    moments, busy, last_kept = [], set(), None
    for a in articles:
        t = datetime.fromisoformat(a["created_at"])
        local = t.astimezone(NEW_YORK)
        if local.date() in trading_days:
            busy.add(local.date())
        in_hours = (9, 45) <= (local.hour, local.minute) and (
            local.hour,
            local.minute,
        ) < (15, 0)
        if not in_hours or len(a["symbols"]) > 2 or local.date() not in trading_days:
            continue
        if last_kept is not None and t - last_kept < timedelta(minutes=60):
            continue
        moments.append(t)
        last_kept = t
    quiet = []
    for d in sorted(trading_days - busy):
        minute = rng.randrange(9 * 60 + 45, 15 * 60)
        quiet.append(
            datetime(
                d.year, d.month, d.day, minute // 60, minute % 60, tzinfo=NEW_YORK
            ).astimezone(UTC)
        )
    news_rows += [r | {"ticker": ticker} for r in measure(frame, moments)]
    quiet_rows += [r | {"ticker": ticker} for r in measure(frame, quiet)]
    if number % 10 == 0:
        print(
            f"   {number}/{len(stocks)} stocks, {len(news_rows):,} articles measured",
            flush=True,
        )

events = {
    "news": pl.DataFrame(news_rows).drop_nans(),
    "no news": pl.DataFrame(quiet_rows).drop_nans(),
}
print(
    f"{len(stocks)} stocks; {events['news'].height:,} articles in hours, "
    f"{events['no news'].height:,} moments without news"
)


def slope(x, y):
    x, y = np.asarray(x), np.asarray(y)
    b = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2)
    resid = y - y.mean() - b * (x - x.mean())
    se = math.sqrt(np.sum(resid**2) / (len(x) - 2) / np.sum((x - x.mean()) ** 2))
    return b, b / se


print(
    "Does the move carry on? Slope of what follows on the reaction (t), and for the biggest"
)
print("tenth of reactions, the move after it in the reaction's direction, in bps (t)")
for react, after in (
    ("react_1", "after_1_30"),
    ("react_1", "after_1_60"),
    ("react_5", "after_5_60"),
    ("react_5", "after_5_close"),
):
    cells = []
    for name, frame in events.items():
        b, t = slope(frame[react], frame[after])
        big = frame.filter(frame[react].abs() >= frame[react].abs().quantile(0.9))
        signed = (np.sign(big[react].to_numpy()) * big[after].to_numpy()) * 1e4
        tt = signed.mean() / signed.std(ddof=1) * math.sqrt(len(signed))
        cells.append(
            f"{name}: slope {b:+.3f} (t {t:+.1f}), biggest tenth {signed.mean():+.1f} bps (t {tt:+.1f}),"
            f" reaction {big[react].abs().mean() * 1e4:.0f} bps"
        )
    print(f"   {react} -> {after}:")
    for c in cells:
        print(f"      {c}")
