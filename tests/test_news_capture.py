"""The 15:31 news capture (qfin_alphaguard.data.news_capture)."""

import json
import urllib.parse
from datetime import UTC, date, datetime

import pytest

from qfin_alphaguard.data import news_capture as nc


def utc(*args):
    return datetime(*args, tzinfo=UTC)


def test_the_window_runs_from_16_00_on_the_session_before_to_15_30():
    # Monday 19 October 2026 (New York on summer time, UTC-4): from Friday 16:00.
    assert nc.window(date(2026, 10, 19)) == (
        utc(2026, 10, 16, 20, 0),
        utc(2026, 10, 19, 19, 30),
    )
    # Monday 9 November 2026, after the clocks changed (UTC-5).
    assert nc.window(date(2026, 11, 9)) == (
        utc(2026, 11, 6, 21, 0),
        utc(2026, 11, 9, 20, 30),
    )
    # The day after Thanksgiving: the session before is Wednesday.
    assert nc.window(date(2026, 11, 27))[0] == utc(2026, 11, 25, 21, 0)
    assert nc.window(date(2026, 10, 24)) is None  # a Saturday
    assert nc.window(date(2026, 11, 26)) is None  # Thanksgiving


class FakeNews:
    """The news API: two pages, recording what was asked."""

    def __init__(self):
        self.asked = []

    def __call__(self, url):
        query = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(url).query))
        self.asked.append(query)
        article = {
            "id": 1,
            "created_at": "2026-10-19T14:00:00Z",
            "updated_at": "2026-10-19T14:00:00Z",
            "symbols": ["AAPL"],
            "headline": "a headline",
            "summary": "",
            "content": "not kept",
        }
        if "page_token" not in query:
            return {"news": [article], "next_page_token": "next"}
        return {"news": [article | {"id": 2, "symbols": []}], "next_page_token": None}


def test_a_capture_keeps_every_page_with_its_capture_time(tmp_path, monkeypatch):
    slept = []
    monkeypatch.setattr(nc.time, "sleep", slept.append)
    news = FakeNews()
    moment = utc(2026, 10, 19, 19, 31, 5)  # 15:31:05 in New York
    path = nc.capture(date(2026, 10, 19), tmp_path, news, now=lambda: moment)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["id"] for r in rows] == [1, 2]
    assert rows[0] == {
        "id": 1,
        "created_at": "2026-10-19T14:00:00Z",
        "updated_at": "2026-10-19T14:00:00Z",
        "symbols": ["AAPL"],
        "headline": "a headline",
        "summary": "",
        "captured_at": "2026-10-19T19:31:05Z",
    }
    # Asked from 30 days before the window (Friday 16:00, so 16 September) to the moment.
    assert news.asked[0]["start"] == "2026-09-16T20:00:00Z"
    assert news.asked[0]["end"] == "2026-10-19T19:31:05Z"
    assert news.asked[1]["page_token"] == "next"
    assert slept == [nc.SPACING]  # between the two pages, not before the first
    assert not path.with_suffix(".part").exists()


def test_a_capture_runs_only_on_its_day_between_15_30_and_16_30(tmp_path):
    for moment in (
        utc(2026, 10, 19, 19, 29),  # 15:29
        utc(2026, 10, 19, 20, 31),  # 16:31
        utc(2026, 10, 20, 19, 31),  # the next day
    ):
        with pytest.raises(nc.CaptureError, match="between 15:30 and 16:30"):
            nc.capture(date(2026, 10, 19), tmp_path, FakeNews(), now=lambda m=moment: m)
    assert list(tmp_path.iterdir()) == []


def test_a_session_is_captured_once_and_a_holiday_not_at_all(tmp_path):
    moment = lambda: utc(2026, 10, 19, 19, 31)
    nc.capture(date(2026, 10, 19), tmp_path, FakeNews(), now=moment)
    with pytest.raises(nc.CaptureError, match="never written again"):
        nc.capture(date(2026, 10, 19), tmp_path, FakeNews(), now=moment)
    other = tmp_path / "other"
    with pytest.raises(nc.CaptureError, match="never written again"):
        nc.capture(
            date(2026, 10, 19),
            other,
            FakeNews(),
            now=moment,
            exists_elsewhere=lambda p: True,
        )
    news = FakeNews()
    assert nc.capture(date(2026, 11, 26), tmp_path, news, now=moment) is None
    assert news.asked == []
