import asyncio
import json
from datetime import UTC, datetime

import pytest

from qfin_alphaguard.events import Bar
from qfin_alphaguard.live.alpaca_broker import BrokerError
from qfin_alphaguard.live.orders import Accepted, Cancelled, FillReport
from qfin_alphaguard.live.streams import (
    DATA_STREAM,
    HANDSHAKE_SECONDS,
    TRADING_STREAM,
    StreamError,
    minute_bars,
    trade_updates,
)

KEY_ID = "key-id-123"
SECRET = "secret-key-456"
ID = "qqq-2026-10-19-1"


class FakeSocket:
    """Stands in for a websockets connection: hands out canned messages in order.

    A message is JSON-ready data, sent as text; a str or bytes is sent as it is, as a
    text or a binary frame. "wait" never answers, as a stream that hangs.
    """

    def __init__(self, *incoming):
        self.incoming = list(incoming)
        self.sent = []
        self.url = None

    def __call__(self, url):  # connect(url)
        self.url = url
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def send(self, text):
        self.sent.append(json.loads(text))

    async def recv(self):
        message = self.incoming.pop(0)
        if message == "wait":
            await asyncio.sleep(3600)
        return message if isinstance(message, str | bytes) else json.dumps(message)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.incoming:
            raise StopAsyncIteration  # the connection closed
        return await self.recv()


def collect(stream):
    async def run():
        return [item async for item in stream]

    return asyncio.run(run())


# The trade updates, as Alpaca sent its handshake on 17 October 2026: binary frames.

AUTHORIZED = json.dumps(
    {
        "stream": "authorization",
        "data": {"action": "authenticate", "status": "authorized"},
    }
).encode()
LISTENING = json.dumps(
    {"stream": "listening", "data": {"streams": ["trade_updates"]}}
).encode()


def order(status, filled="0", average=None):
    return {
        "id": "61e69015-8549-4bfd-b9c3-01e75843f47d",
        "client_order_id": ID,
        "updated_at": "2026-10-19T14:00:01Z",
        "symbol": "QQQ",
        "qty": "100",
        "filled_qty": filled,
        "filled_avg_price": average,
        "order_class": "",
        "type": "limit",
        "side": "buy",
        "status": status,
    }


def trade_update(event, status, filled="0", average=None):
    message = {
        "stream": "trade_updates",
        "data": {
            "event": event,
            "timestamp": "2026-10-19T14:00:02Z",
            "order": order(status, filled, average),
        },
    }
    return json.dumps(message).encode()


AT = datetime(2026, 10, 19, 14, 0, 2, tzinfo=UTC)


def test_trade_updates_authenticate_listen_and_yield_events():
    socket = FakeSocket(
        AUTHORIZED,
        LISTENING,
        trade_update("new", "new"),
        trade_update("partial_fill", "partially_filled", "40", "600.01"),
        trade_update("canceled", "canceled", "40", "600.01"),
    )
    events = collect(trade_updates(KEY_ID, SECRET, connect=socket))
    assert socket.url == TRADING_STREAM == "wss://paper-api.alpaca.markets/stream"
    assert socket.sent == [
        {"action": "auth", "key": KEY_ID, "secret": SECRET},
        {"action": "listen", "data": {"streams": ["trade_updates"]}},
    ]
    assert events == [
        [Accepted(ID, AT)],
        [Accepted(ID, AT), FillReport(ID, AT, 40.0, 600.01)],
        [FillReport(ID, AT, 40.0, 600.01), Cancelled(ID, AT)],
    ]


def test_trade_updates_read_text_frames_too():
    socket = FakeSocket(
        AUTHORIZED.decode(), LISTENING.decode(), trade_update("new", "new").decode()
    )
    assert collect(trade_updates(KEY_ID, SECRET, connect=socket)) == [
        [Accepted(ID, AT)]
    ]


@pytest.mark.parametrize(
    "replies",
    [
        [{"stream": "authorization", "data": {"status": "unauthorized"}}],
        [{"stream": "listening", "data": {"streams": ["trade_updates"]}}],
        [AUTHORIZED, {"stream": "listening", "data": {"streams": []}}],
        [AUTHORIZED, {"stream": "authorization", "data": {"status": "authorized"}}],
        [[{"T": "success", "msg": "connected"}]],
        [b"not json"],
    ],
)
def test_trade_updates_that_refuse_the_engine_stop_it(replies):
    with pytest.raises(StreamError):
        collect(trade_updates(KEY_ID, SECRET, connect=FakeSocket(*replies)))


def test_a_message_from_another_stream_stops_the_engine():
    socket = FakeSocket(AUTHORIZED, LISTENING, {"stream": "account_updates"})
    with pytest.raises(StreamError, match="account_updates"):
        collect(trade_updates(KEY_ID, SECRET, connect=socket))


def test_an_update_the_engine_does_not_expect_stops_it():
    socket = FakeSocket(AUTHORIZED, LISTENING, trade_update("replaced", "replaced"))
    with pytest.raises(BrokerError):
        collect(trade_updates(KEY_ID, SECRET, connect=socket))


@pytest.mark.parametrize(
    "stream",
    [
        lambda socket: trade_updates(KEY_ID, SECRET, connect=socket),
        lambda socket: minute_bars(KEY_ID, SECRET, ["QQQ"], connect=socket),
    ],
    ids=["trade updates", "minute bars"],
)
def test_a_handshake_that_hangs_stops_the_engine(monkeypatch, stream):
    monkeypatch.setattr("qfin_alphaguard.live.streams.HANDSHAKE_SECONDS", 0.05)
    with pytest.raises(TimeoutError):
        collect(stream(FakeSocket("wait")))
    assert HANDSHAKE_SECONDS == 10


def test_the_keys_never_appear_in_a_refusal():
    reply = {"stream": "authorization", "data": {"status": "unauthorized"}}
    with pytest.raises(StreamError) as error:
        collect(trade_updates(KEY_ID, SECRET, connect=FakeSocket(reply)))
    assert KEY_ID not in str(error.value) and SECRET not in str(error.value)


# The minute bars, as Alpaca sent them on 17 October 2026: text frames, lists.

CONNECTED = [{"T": "success", "msg": "connected"}]
AUTHENTICATED = [{"T": "success", "msg": "authenticated"}]
SUBSCRIBED = [{"T": "subscription", "bars": ["QQQ"]}]


def bar_message(minute, symbol="QQQ", **changes):
    message = {
        "T": "b",
        "S": symbol,
        "o": 709.5,
        "h": 710,  # Alpaca writes a whole price without decimals
        "l": 709.25,
        "c": 709.82,
        "v": 1532,
        "t": f"2026-10-16T19:{minute:02d}:00Z",
        "n": 16,
        "vw": 709.7,
    }
    return message | changes


def bar(minute, symbol="QQQ"):
    return Bar(
        ticker=symbol,
        start=datetime(2026, 10, 16, 19, minute, tzinfo=UTC),
        seconds=60,
        open=709.5,
        high=710.0,
        low=709.25,
        close=709.82,
        volume=1532,
    )


def test_bars_come_as_the_backtest_s_bar_events():
    socket = FakeSocket(
        CONNECTED,
        AUTHENTICATED,
        SUBSCRIBED,
        [bar_message(58)],
        [bar_message(59)],
    )
    bars = collect(minute_bars(KEY_ID, SECRET, ["QQQ"], connect=socket))
    assert socket.url == DATA_STREAM.format(feed="iex")
    assert socket.url == "wss://stream.data.alpaca.markets/v2/iex"
    assert socket.sent == [
        {"action": "auth", "key": KEY_ID, "secret": SECRET},
        {"action": "subscribe", "bars": ["QQQ"]},
    ]
    assert bars == [bar(58), bar(59)]
    assert bars[0].end == datetime(2026, 10, 16, 19, 59, tzinfo=UTC)


def test_several_symbols_in_one_message_come_one_by_one():
    socket = FakeSocket(
        CONNECTED,
        AUTHENTICATED,
        [{"T": "subscription", "trades": [], "quotes": [], "bars": ["SPY", "QQQ"]}],
        [bar_message(58), bar_message(58, "SPY")],
    )
    bars = collect(minute_bars(KEY_ID, SECRET, ["QQQ", "SPY", "QQQ"], connect=socket))
    assert socket.sent[1] == {"action": "subscribe", "bars": ["QQQ", "SPY"]}
    assert bars == [bar(58), bar(58, "SPY")]


def test_other_feeds_have_their_own_address():
    socket = FakeSocket(
        CONNECTED, AUTHENTICATED, [{"T": "subscription", "bars": ["FAKEPACA"]}]
    )
    collect(minute_bars(KEY_ID, SECRET, ["FAKEPACA"], feed="test", connect=socket))
    assert socket.url == "wss://stream.data.alpaca.markets/v2/test"


@pytest.mark.parametrize(("symbols", "feed"), [(["QQQ"], "delayed_sip"), ([], "iex")])
def test_a_feed_or_symbols_that_cannot_be_asked_for_are_refused(symbols, feed):
    socket = FakeSocket()
    with pytest.raises(ValueError):
        collect(minute_bars(KEY_ID, SECRET, symbols, feed=feed, connect=socket))
    assert socket.url is None


@pytest.mark.parametrize(
    ("replies", "match"),
    [
        ([[{"T": "error", "code": 406, "msg": "connection limit exceeded"}]], "406"),
        ([CONNECTED, [{"T": "error", "code": 402, "msg": "auth failed"}]], "402"),
        (
            [
                CONNECTED,
                AUTHENTICATED,
                [{"T": "error", "code": 409, "msg": "insufficient subscription"}],
            ],
            "subscription",
        ),
        ([CONNECTED, AUTHENTICATED, [{"T": "subscription", "bars": []}]], "not"),
        ([CONNECTED, AUTHENTICATED, [{"T": "subscription", "bars": ["SPY"]}]], "SPY"),
        (
            [
                CONNECTED,
                AUTHENTICATED,
                SUBSCRIBED,
                [{"T": "error", "code": 407, "msg": "slow client"}],
            ],
            "407 slow client",
        ),
        (
            [CONNECTED, AUTHENTICATED, SUBSCRIBED, [{"T": "u", "S": "QQQ"}]],
            "unexpected",
        ),
        ([CONNECTED, AUTHENTICATED, SUBSCRIBED, {"T": "b"}], "list"),
        ([CONNECTED, AUTHENTICATED, SUBSCRIBED, b"not json"], "JSON"),
    ],
)
def test_market_data_that_refuses_or_confuses_the_engine_stops_it(replies, match):
    socket = FakeSocket(*replies)
    with pytest.raises(StreamError, match=match):
        collect(minute_bars(KEY_ID, SECRET, ["QQQ"], connect=socket))


@pytest.mark.parametrize(
    "changes",
    [
        {"h": 709.0},  # a high below the close
        {"t": "2026-10-16T19:58:00"},  # no zone
        {"c": None},
        {"v": -1},
        {"o": 0},
    ],
)
def test_a_bar_that_cannot_be_read_stops_the_engine(changes):
    socket = FakeSocket(
        CONNECTED, AUTHENTICATED, SUBSCRIBED, [bar_message(58, **changes)]
    )
    with pytest.raises(StreamError, match="bar"):
        collect(minute_bars(KEY_ID, SECRET, ["QQQ"], connect=socket))


def test_a_bar_s_time_becomes_utc():
    socket = FakeSocket(
        CONNECTED,
        AUTHENTICATED,
        SUBSCRIBED,
        [bar_message(58, t="2026-10-16T15:58:00-04:00")],
    )
    (got,) = collect(minute_bars(KEY_ID, SECRET, ["QQQ"], connect=socket))
    assert got == bar(58) and got.start.tzinfo is UTC
