"""Every Benzinga article on Alpaca's news API, 2016 to October 2026, a month per file.

The news reader (the plan after research/frontier) needs all the news, not only the articles
that tag our stocks: the market's morning news for the QQQ rule, and each member's news for
the stocks. This pages through the API month by month with no symbol filter and keeps, for each
article, its id, publication and last-edit times (UTC, to the second), the stocks it tags, its
headline and its summary. The text is Benzinga's: it stays in the git-ignored data/.

A month already written is skipped, so a stopped run resumes. Requests are spaced 0.34 s apart
(Alpaca's free plan allows 200 a minute), three months at a time.

Usage: news_all.py OUT_DIR   (with ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY set)
"""

import json
import os
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

OUT = Path(sys.argv[1])
FIRST, LAST = date(2016, 1, 1), date(2026, 10, 16)
HEADERS = {
    "APCA-API-KEY-ID": os.environ["ALPACA_API_KEY_ID"],
    "APCA-API-SECRET-KEY": os.environ["ALPACA_API_SECRET_KEY"],
}
_lock, _last_request = threading.Lock(), [0.0]


def page(query: dict) -> dict:
    url = "https://data.alpaca.markets/v1beta1/news?" + urllib.parse.urlencode(query)
    for attempt in range(6):
        with _lock:
            wait = 0.34 - (time.monotonic() - _last_request[0])
            if wait > 0:
                time.sleep(wait)
            _last_request[0] = time.monotonic()
        try:
            request = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read())
        except (OSError, ValueError) as error:  # network errors, rate limits
            if attempt == 5:
                raise RuntimeError(f"news page {query.get('start')}: {error}") from None
            print(
                f"   retry {attempt + 1} for {query.get('start')}: {error}", flush=True
            )
            time.sleep(2 * 2**attempt)
    raise AssertionError


def month(first: date) -> int:
    path = OUT / f"{first:%Y-%m}.jsonl"
    if path.exists():
        return 0
    last = date(first.year + first.month // 12, first.month % 12 + 1, 1)
    query = {
        "start": f"{first}T00:00:00Z",
        "end": f"{min(last, LAST)}T00:00:00Z",
        "limit": 50,
        "sort": "asc",
        "include_content": "false",
    }
    articles = []
    while True:
        got = page(query)
        articles += [
            {
                "id": a["id"],
                "created_at": a["created_at"],
                "updated_at": a["updated_at"],
                "symbols": a["symbols"],
                "headline": a["headline"],
                "summary": a.get("summary", ""),
            }
            for a in got["news"]
        ]
        if not got.get("next_page_token"):
            break
        query["page_token"] = got["next_page_token"]
    part = path.with_suffix(".part")
    part.write_text("".join(json.dumps(a) + "\n" for a in articles))
    part.rename(path)
    return len(articles)


OUT.mkdir(parents=True, exist_ok=True)
months, d = [], FIRST
while d <= LAST:
    months.append(d)
    d = date(d.year + d.month // 12, d.month % 12 + 1, 1)
with ThreadPoolExecutor(max_workers=3) as pool:
    for first, count in zip(months, pool.map(month, months), strict=True):
        print(f"{first:%Y-%m}: {count} articles", flush=True)
