import io
import json
import urllib.error
from datetime import UTC, date, datetime, timedelta

import pytest

from qfin_alphaguard.data.eodhd import (
    BARS_SCHEMA,
    MINUTE_SCHEMA,
    EodhdClient,
    api_key,
    eodhd_code,
)

KEY = "secret-key-123"


class FakeUrlopen:
    """Stands in for urllib.request.urlopen: hands out canned replies in order.

    A reply is a JSON-ready object, or an int for an HTTP error with that code.
    """

    def __init__(self, *replies):
        self.replies = list(replies)
        self.urls = []

    def __call__(self, url, timeout):
        self.urls.append(url)
        reply = self.replies.pop(0)
        if isinstance(reply, int):
            raise urllib.error.HTTPError(url, reply, "error", {}, None)
        return io.BytesIO(json.dumps(reply).encode())


def client(*replies):
    fake = FakeUrlopen(*replies)
    return EodhdClient(KEY, urlopen=fake, sleep=lambda seconds: None), fake


# Netflix around its 10-for-1 split on 17 November 2025, as EODHD reports it:
# prices as traded, the adjusted close, and volume already split-adjusted.
NFLX_ROWS = [
    {
        "date": "2025-11-14",
        "open": 1142.73,
        "high": 1150.0,
        "low": 1100.0,
        "close": 1112.17,
        "adjusted_close": 111.217,
        "volume": 47607130,
    },
    {
        "date": "2025-11-17",
        "open": 110.75,
        "high": 112,
        "low": 109,
        "close": 110.29,
        "adjusted_close": 110.29,
        "volume": 26082730,
    },
]


def test_bars_are_adjusted_and_keep_the_factor():
    eod, _ = client(NFLX_ROWS)

    bars = eod.daily_bars("NFLX", date(2025, 11, 14), date(2025, 11, 17))

    assert dict(bars.schema) == BARS_SCHEMA
    before, after = bars.rows(named=True)
    assert before["close"] == 111.217
    assert before["open"] == pytest.approx(114.273)
    assert before["adjustment"] == pytest.approx(0.1)
    # The price as traded comes back by dividing by the adjustment.
    assert before["open"] / before["adjustment"] == pytest.approx(1142.73)
    assert after["adjustment"] == 1.0
    assert after["volume"] == 26082730
    assert bars["ticker"].to_list() == ["NFLX", "NFLX"]


def test_days_without_trades_are_dropped():
    # After a delisting EODHD repeats the last price with zero volume.
    stale = {**NFLX_ROWS[1], "date": "2025-11-18", "volume": 0}
    eod, _ = client([*NFLX_ROWS, stale])
    bars = eod.daily_bars("NFLX", date(2025, 11, 14), date(2025, 11, 18))
    assert bars["date"].max() == date(2025, 11, 17)


def test_an_unknown_code_gives_no_bars():
    eod, _ = client(404)
    bars = eod.daily_bars("NOPE", date(2025, 1, 2), date(2025, 1, 31))
    assert bars.is_empty()
    assert dict(bars.schema) == BARS_SCHEMA


def test_a_rate_limit_is_retried():
    eod, fake = client(429, 503, NFLX_ROWS)
    assert eod.daily_bars("NFLX", date(2025, 11, 14), date(2025, 11, 17)).height == 2
    assert len(fake.urls) == 3


def test_a_bad_key_fails_at_once_and_the_error_hides_the_key():
    eod, fake = client(401)

    with pytest.raises(RuntimeError, match="HTTP 401") as failure:
        eod.daily_bars("NFLX", date(2025, 11, 14), date(2025, 11, 17))

    assert len(fake.urls) == 1
    assert KEY not in str(failure.value)
    assert failure.value.__context__ is None  # no chained error carrying the URL


def test_gives_up_after_repeated_server_errors():
    eod, fake = client(500, 500, 500, 500)
    with pytest.raises(RuntimeError, match="HTTP 500"):
        eod.daily_bars("NFLX", date(2025, 11, 14), date(2025, 11, 17))
    assert len(fake.urls) == 4


def minute(start, close=100.0):
    return {
        "timestamp": int(start.timestamp()),
        "gmtoffset": 0,
        "datetime": start.strftime("%Y-%m-%d %H:%M:%S"),
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": 10,
    }


def test_minute_bars_are_asked_for_in_windows_and_come_back_in_utc():
    bar = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)
    later = bar + timedelta(minutes=1)
    # 2 January to 30 June is 180 days: two windows of at most 100 days.
    eod, fake = client([minute(bar)], [minute(bar), minute(later)])

    bars = eod.minute_bars("AAPL", date(2026, 1, 2), date(2026, 6, 30))

    assert len(fake.urls) == 2
    assert dict(bars.schema) == MINUTE_SCHEMA
    assert bars["start"].to_list() == [bar, later]  # the repeated minute once


def test_splits_are_new_shares_per_old_share():
    eod, fake = client(
        [
            {"date": "2025-11-17", "split": "10.000000/1.000000"},
            {"date": "2020-04-15", "split": "1.000000/200.000000"},  # reverse split
        ]
    )
    assert eod.splits("NFLX") == [
        (date(2020, 4, 15), 0.005),
        (date(2025, 11, 17), 10.0),
    ]
    assert "splits/NFLX.US" in fake.urls[0]


def test_old_codes_group_earlier_companies_by_ticker():
    listing = [
        {"Code": "DOW_old", "Name": "The Dow Chemical Company"},
        {"Code": "AAC_old2", "Name": "Second AAC"},
        {"Code": "AAC_old1", "Name": "First AAC"},
        {"Code": "CELG", "Name": "Celgene Corporation"},
    ]
    eod, fake = client(listing)

    assert eod.old_codes() == {"DOW": ("DOW_old",), "AAC": ("AAC_old1", "AAC_old2")}
    assert "delisted=1" in fake.urls[0]


def test_class_shares_use_a_dash():
    assert eodhd_code("BRK.B") == "BRK-B"
    assert eodhd_code("AAPL") == "AAPL"


def test_a_missing_key_says_how_to_set_it(monkeypatch):
    monkeypatch.delenv("EODHD_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="--env-file .env"):
        api_key()
