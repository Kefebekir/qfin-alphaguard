"""The news as it stands at 15:31 New York time, captured once a session (pre-registration N5).

N5's record reads each session's headlines as they stood when its decision would be made, so
that an edit after 15:30 cannot reach it. Each session, between 15:30 and 16:30 New York time,
this asks Alpaca's news API for every article from 30 days before the window to the moment of
asking (the API selects by creation; an older article edited inside the window counts from its
edit, and 30 days keeps all but 0.033% of them), and keeps each one's id, its creation and
last-edit times, the stocks it tags, its headline and its summary, with the time it was
captured. Which articles count, and from when, is decided later by N2's rule (the window from
16:00 on the session before to 15:30, an article edited more than five minutes after it was
created known from its edit).

A session is captured once: a capture already on disk or in S3 is never written again, since
a second one would see later edits. A day that is not a session has nothing to capture.

Output: ROOT/<day>.jsonl, one article a line; with a bucket, also s3://BUCKET/ROOT/<day>.jsonl.
"""

import json
import time
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from datetime import time as clock
from pathlib import Path
from zoneinfo import ZoneInfo

from qfin_alphaguard.sessions import sessions

NEW_YORK = ZoneInfo("America/New_York")
NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
EARLIEST, LATEST = clock(15, 30), clock(16, 30)  # New York time, when a capture may run
# Asked for before the window: the API selects by creation, and an article created earlier but
# edited inside the window counts from its edit. Of the 503,998 member article-pairs of
# 2016-2026, 0.11% were created more than 2 days before they counted, 0.033% more than 30.
MARGIN = timedelta(days=30)
SPACING = 0.31  # seconds between pages: the free plan allows 200 requests a minute


class CaptureError(RuntimeError):
    pass


def at(day: date, hour: int, minute: int) -> datetime:
    return datetime.combine(day, clock(hour, minute), tzinfo=NEW_YORK).astimezone(UTC)


def window(day: date) -> tuple[datetime, datetime] | None:
    """N2's window for `day`, in UTC: 16:00 on the session before to 15:30; None if `day`
    is not a session."""
    days = [s.day for s in sessions(day - timedelta(days=10), day)]
    if not days or days[-1] != day:
        return None
    return at(days[-2], 16, 0), at(day, 15, 30)


def get_json(url: str, keys: tuple[str, str]) -> dict:
    """One page of the news API; retried on network errors and rate limits."""
    headers = {"APCA-API-KEY-ID": keys[0], "APCA-API-SECRET-KEY": keys[1]}
    for attempt in range(6):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read())
        except (OSError, ValueError) as error:
            if attempt == 5:
                raise CaptureError(f"the news API: {error}") from None
            time.sleep(2 * 2**attempt)
    raise AssertionError


def fetch(start: datetime, end: datetime, get: Callable[[str], dict]) -> list[dict]:
    """Every article from `start` to `end`, page by page, as the API answers them."""
    articles, token = [], None
    while True:
        query = {
            "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "limit": 50,
            "sort": "asc",
            "include_content": "false",
        }
        if token:
            query["page_token"] = token
            time.sleep(SPACING)
        answer = get(f"{NEWS_URL}?{urllib.parse.urlencode(query)}")
        articles.extend(answer.get("news", []))
        token = answer.get("next_page_token")
        if not token:
            return articles


def record(article: dict, captured: datetime) -> dict:
    return {
        "id": article["id"],
        "created_at": article["created_at"],
        "updated_at": article["updated_at"],
        "symbols": article.get("symbols", []),
        "headline": article.get("headline", ""),
        "summary": article.get("summary", ""),
        "captured_at": captured.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def capture(
    day: date,
    root: Path,
    get: Callable[[str], dict],
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    exists_elsewhere: Callable[[Path], bool] = lambda path: False,
) -> Path | None:
    """The day's capture, written once; None if `day` is not a session."""
    span = window(day)
    if span is None:
        return None
    moment = now()
    local = moment.astimezone(NEW_YORK)
    if local.date() != day or not EARLIEST <= local.time() <= LATEST:
        raise CaptureError(
            f"a capture of {day} runs on that day between 15:30 and 16:30 New York time, "
            f"not at {local:%Y-%m-%d %H:%M}"
        )
    path = root / f"{day.isoformat()}.jsonl"
    if path.exists() or exists_elsewhere(path):
        raise CaptureError(f"{day} has been captured: {path} is never written again")
    articles = fetch(span[0] - MARGIN, moment, get)
    root.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")
    partial.write_text("".join(json.dumps(record(a, moment)) + "\n" for a in articles))
    partial.rename(path)
    return path
