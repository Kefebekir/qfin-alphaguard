import http.client
import io
import json
import urllib.error
import urllib.parse
from datetime import UTC, date, datetime

import pytest

from qfin_alphaguard.data.alpaca import PAGE_SIZE, AlpacaClient, api_keys
from qfin_alphaguard.data.eodhd import MINUTE_SCHEMA
from qfin_alphaguard.data.universe import IndexHistory, Membership

KEY_ID = "key-id-123"
SECRET = "secret-key-456"


class FakeUrlopen:
    """Stands in for urllib.request.urlopen: hands out canned replies in order.

    A reply is a JSON-ready page, an int for an HTTP error with that code, or
    (code, answer) for an error whose answer says why.
    """

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []
        self.sleeps = []  # seconds waited before each retry

    def __call__(self, request, timeout):
        self.requests.append(request)
        reply = self.replies.pop(0)
        if isinstance(reply, int):
            reply = (reply, "")
        if isinstance(reply, tuple):
            code, answer = reply
            raise urllib.error.HTTPError(
                request.full_url, code, "error", {}, io.BytesIO(answer.encode())
            )
        if reply == "cut off":
            raise http.client.IncompleteRead(b'{"bars": {"QQQ": [{"t": "2026-1')
        return io.BytesIO(json.dumps(reply).encode())

    def query(self, n):
        """The parameters of the n-th request."""
        url = self.requests[n].full_url
        return dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))


def client(*replies):
    fake = FakeUrlopen(*replies)
    return AlpacaClient(KEY_ID, SECRET, urlopen=fake, sleep=fake.sleeps.append), fake


def bar(start, close=600.0):
    """One bar as Alpaca sends it: trade count and VWAP come too, and are not kept."""
    return {
        "t": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "o": close - 1,
        "h": close + 2,
        "l": close - 3,
        "c": close,
        "v": 1200,
        "n": 35,
        "vw": close - 0.5,
    }


def page(*bars, token=None):
    return {"bars": {"QQQ": list(bars)} if bars else {}, "next_page_token": token}


OPEN = datetime(2026, 10, 9, 13, 30, tzinfo=UTC)


def test_minute_bars_follow_the_pages_and_come_back_in_utc():
    later = datetime(2026, 10, 9, 13, 31, tzinfo=UTC)
    alpaca, fake = client(page(bar(OPEN), token="page-2"), page(bar(later, 601.0)))

    bars = alpaca.minute_bars("QQQ", date(2026, 10, 9), date(2026, 10, 9))

    assert dict(bars.schema) == MINUTE_SCHEMA
    assert bars.rows() == [
        (OPEN, "QQQ", 599.0, 602.0, 597.0, 600.0, 1200),
        (later, "QQQ", 600.0, 603.0, 598.0, 601.0, 1200),
    ]
    assert "page_token" not in fake.query(0)
    assert fake.query(1)["page_token"] == "page-2"


def test_minute_bars_ask_for_every_exchange_and_prices_as_traded():
    alpaca, fake = client(page())

    alpaca.minute_bars("QQQ", date(2016, 1, 1), date(2016, 12, 31))

    assert fake.requests[0].full_url.startswith(
        "https://data.alpaca.markets/v2/stocks/bars?"
    )
    assert fake.query(0) == {
        "symbols": "QQQ",
        "timeframe": "1Min",
        "start": "2016-01-01T00:00:00Z",
        "end": "2016-12-31T23:59:59Z",  # all of the last day, none of the next
        "feed": "sip",  # the consolidated tape, not IEX's share alone
        "adjustment": "raw",
        "sort": "asc",  # oldest first, so the pages follow on from each other
        "limit": str(PAGE_SIZE),
    }


def test_minute_bars_can_ask_for_iex_s_trades_alone():
    alpaca, fake = client(page())
    alpaca.minute_bars("QQQ", date(2026, 10, 9), date(2026, 10, 9), feed="iex")
    assert fake.query(0)["feed"] == "iex"


def test_an_unknown_feed_is_refused_before_any_request():
    alpaca, fake = client(page())
    with pytest.raises(ValueError, match="feed"):
        alpaca.minute_bars("QQQ", date(2026, 10, 9), date(2026, 10, 9), feed="otc")
    assert fake.requests == []


def test_the_keys_travel_in_headers_never_in_the_url():
    alpaca, fake = client(page())

    alpaca.minute_bars("QQQ", date(2026, 10, 9), date(2026, 10, 9))

    request = fake.requests[0]
    # urllib stores header names capitalized; HTTP ignores the case.
    assert request.get_header("Apca-api-key-id") == KEY_ID
    assert request.get_header("Apca-api-secret-key") == SECRET
    assert KEY_ID not in request.full_url
    assert SECRET not in request.full_url


def test_no_bars_give_an_empty_frame_with_the_minute_schema():
    alpaca, _ = client(page())

    bars = alpaca.minute_bars(
        "QQQ", date(2026, 10, 10), date(2026, 10, 11)
    )  # a weekend

    assert bars.is_empty()
    assert dict(bars.schema) == MINUTE_SCHEMA


def test_a_rate_limit_and_a_server_error_are_retried_after_a_growing_wait():
    alpaca, fake = client(429, 503, page(bar(OPEN)))
    assert alpaca.minute_bars("QQQ", date(2026, 10, 9), date(2026, 10, 9)).height == 1
    assert len(fake.requests) == 3
    assert fake.sleeps == [2, 4]


def test_an_answer_cut_off_on_the_way_is_retried():
    alpaca, fake = client("cut off", page(bar(OPEN)))
    assert alpaca.minute_bars("QQQ", date(2026, 10, 9), date(2026, 10, 9)).height == 1
    assert len(fake.requests) == 2


def test_a_refusal_fails_at_once_says_why_and_hides_the_keys():
    answer = '{"message":"subscription does not permit querying recent SIP data"}'
    alpaca, fake = client((403, answer))

    with pytest.raises(RuntimeError, match="HTTP 403.*recent SIP data") as failure:
        alpaca.minute_bars("QQQ", date(2026, 10, 9), date(2026, 10, 9))

    assert len(fake.requests) == 1
    assert KEY_ID not in str(failure.value)
    assert SECRET not in str(failure.value)
    assert failure.value.__context__ is None  # nothing chained carries the request


def test_gives_up_after_repeated_server_errors():
    alpaca, fake = client(500, 500, 500, 500)
    with pytest.raises(RuntimeError, match="HTTP 500"):
        alpaca.minute_bars("QQQ", date(2026, 10, 9), date(2026, 10, 9))
    assert len(fake.requests) == 4


HISTORY = IndexHistory((Membership("AAPL", date(2000, 1, 3), None),))


@pytest.mark.parametrize(
    ("code", "tickers"),
    [
        ("QQQ", ("QQQ",)),
        ("BRK-B", ("BRK.B",)),  # Alpaca writes class shares with a dot
        ("CHK_old", ("CHK",)),  # the old Chesapeake, under its own ticker
        ("FB_old", ("FB",)),
        ("AABA", ("AABA", "YHOO")),  # Altaba before June 2017 is only under YHOO
    ],
)
def test_a_stock_is_looked_for_under_its_tickers_at_alpaca(code, tickers):
    alpaca, _ = client()
    assert alpaca.minute_codes(code, HISTORY) == tickers


def test_the_keys_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY_ID", f" {KEY_ID}\n")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", SECRET)
    assert api_keys() == (KEY_ID, SECRET)


def test_a_missing_key_says_how_to_set_it(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY_ID", KEY_ID)
    monkeypatch.delenv("ALPACA_API_SECRET_KEY", raising=False)
    with pytest.raises(RuntimeError, match="--env-file .env"):
        api_keys()
