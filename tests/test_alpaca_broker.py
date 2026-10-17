import io
import json
import urllib.error
import urllib.parse
from datetime import UTC, datetime

import pytest

from qfin_alphaguard.events import Order, OrderIntent, Side
from qfin_alphaguard.live.alpaca_broker import (
    ATTEMPTS,
    MAX_OPEN_ORDERS,
    PAPER_URL,
    AlpacaBroker,
    AlpacaOrder,
    BrokerError,
    OutcomeUnknown,
    trade_update_events,
)
from qfin_alphaguard.live.orders import (
    Accepted,
    Cancelled,
    CancelRefused,
    Expired,
    FillReport,
    InvalidTransition,
    Rejected,
    Sent,
    Status,
    apply,
    replay,
)

KEY_ID = "key-id-123"
SECRET = "secret-key-456"
NOW = datetime(2026, 10, 19, 14, 0, 5, tzinfo=UTC)
T = datetime(2026, 10, 19, 14, 0, tzinfo=UTC)
ID = "qqq-2026-10-19-1"
BROKER_ID = "61e69015-8549-4bfd-b9c3-01e75843f47d"


class Response(io.BytesIO):
    def __init__(self, status: int, body: bytes) -> None:
        super().__init__(body)
        self.status = status


class FakeUrlopen:
    """Stands in for urllib.request.urlopen: hands out canned replies in order.

    A reply is (code, JSON-ready answer or None), with an HTTP error for a code from 400
    up, or an exception to raise, as a lost connection does.
    """

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []
        self.sleeps = []

    def __call__(self, request, timeout):
        self.requests.append(request)
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        code, answer = reply
        body = b"" if answer is None else json.dumps(answer).encode()
        if code >= 400:
            raise urllib.error.HTTPError(
                request.full_url, code, "error", {}, io.BytesIO(body)
            )
        return Response(code, body)

    def call(self, n):
        """The method, the path with its query, and the JSON body of the n-th request."""
        request = self.requests[n]
        url = urllib.parse.urlsplit(request.full_url)
        body = json.loads(request.data) if request.data else None
        path = url.path + (f"?{url.query}" if url.query else "")
        return request.get_method(), path, body


def broker(*replies):
    fake = FakeUrlopen(*replies)
    return (
        AlpacaBroker(
            KEY_ID, SECRET, urlopen=fake, sleep=fake.sleeps.append, now=lambda: NOW
        ),
        fake,
    )


def make_order(quantity=100.0, side=Side.BUY, limit=600.12, client_order_id=ID):
    intent = OrderIntent("QQQ", side, quantity, limit, T)
    return Order(client_order_id=client_order_id, intent=intent, sent_at=T)


def alpaca_order(status="accepted", filled="0", average=None, **changes):
    """An order as Alpaca's API answers with it, nanoseconds and all."""
    answer = {
        "id": BROKER_ID,
        "client_order_id": ID,
        "created_at": "2026-10-19T14:00:00.123456789Z",
        "updated_at": "2026-10-19T14:00:01.987654321Z",
        "submitted_at": "2026-10-19T14:00:00.120000Z",
        "filled_at": None,
        "asset_id": "b0b6dd9d-8b9b-48a9-ba46-b9d54906e415",
        "symbol": "QQQ",
        "asset_class": "us_equity",
        "notional": None,
        "qty": "100",
        "filled_qty": filled,
        "filled_avg_price": average,
        "order_class": "",
        "order_type": "limit",
        "type": "limit",
        "side": "buy",
        "time_in_force": "day",
        "limit_price": "600.12",
        "stop_price": None,
        "status": status,
        "extended_hours": False,
        "legs": None,
    }
    return answer | changes


UPDATED = datetime(2026, 10, 19, 14, 0, 1, 987654, tzinfo=UTC)


# Sending an order.


def test_an_order_goes_out_as_a_day_limit_order_with_its_client_id():
    b, fake = broker((200, alpaca_order("accepted")))
    events = b.submit(make_order())
    assert fake.call(0) == (
        "POST",
        "/v2/orders",
        {
            "symbol": "QQQ",
            "qty": "100",
            "side": "buy",
            "type": "limit",
            "time_in_force": "day",
            "limit_price": "600.12",
            "client_order_id": ID,
        },
    )
    assert events == [Accepted(ID, UPDATED)]
    request = fake.requests[0]
    assert request.full_url.startswith(PAPER_URL + "/")
    assert request.get_header("Apca-api-key-id") == KEY_ID
    assert request.get_header("Apca-api-secret-key") == SECRET
    assert KEY_ID not in request.full_url and SECRET not in request.full_url


def test_every_request_goes_to_the_paper_account():
    assert PAPER_URL == "https://paper-api.alpaca.markets"


@pytest.mark.parametrize(
    ("quantity", "side", "text"),
    [
        (100.0, Side.BUY, "100"),
        (0.123456789, Side.BUY, "0.123456789"),
        (12.5, Side.SELL, "12.5"),
    ],
)
def test_shares_are_sent_as_alpaca_writes_them(quantity, side, text):
    b, fake = broker((200, alpaca_order(qty=text, side=side.value)))
    b.submit(make_order(quantity=quantity, side=side))
    assert fake.call(0)[2]["qty"] == text
    assert fake.call(0)[2]["side"] == side.value


def test_a_limit_price_below_a_dollar_takes_hundredths_of_a_cent():
    b, fake = broker((200, alpaca_order(limit_price="0.5012")))
    b.submit(make_order(limit=0.5012))
    assert fake.call(0)[2]["limit_price"] == "0.5012"


@pytest.mark.parametrize("limit", [600.125, 0.50125])
def test_a_limit_price_between_ticks_is_refused_before_any_request(limit):
    b, fake = broker()
    with pytest.raises(ValueError, match="tick"):
        b.submit(make_order(limit=limit))
    assert fake.requests == []


def test_a_client_order_id_alpaca_cannot_take_is_refused_before_any_request():
    b, fake = broker()
    with pytest.raises(ValueError, match="128"):
        b.submit(make_order(client_order_id="x" * 129))
    assert fake.requests == []


def test_an_order_alpaca_refuses_is_rejected_and_never_sent_again():
    answer = {"code": 40310000, "message": "insufficient buying power"}
    b, fake = broker((403, answer))
    assert b.submit(make_order()) == [Rejected(ID, NOW, "insufficient buying power")]
    assert len(fake.requests) == 1


def test_a_rate_limited_send_placed_nothing():
    b, fake = broker((429, {"message": "too many requests"}))
    assert b.submit(make_order()) == [Rejected(ID, NOW, "too many requests")]
    assert len(fake.requests) == 1


def test_a_client_order_id_alpaca_already_has_stops_the_engine():
    # The engine sent the same order twice: the first is out there.
    answer = {"code": 40010001, "message": "client_order_id must be unique"}
    b, _ = broker((422, answer))
    with pytest.raises(BrokerError, match="unique"):
        b.submit(make_order())


@pytest.mark.parametrize(
    "lost",
    [
        urllib.error.URLError(ConnectionResetError("connection reset by peer")),
        TimeoutError("timed out"),
        ConnectionResetError("connection reset by peer"),
        (500, {"message": "internal server error"}),
        (503, None),
    ],
)
def test_a_send_whose_answer_is_lost_is_never_repeated(lost):
    # The order may be at Alpaca or not; only asking for it tells.
    b, fake = broker(lost)
    with pytest.raises(OutcomeUnknown, match=ID) as error:
        b.submit(make_order())
    assert len(fake.requests) == 1 and fake.sleeps == []
    assert KEY_ID not in str(error.value) and SECRET not in str(error.value)


@pytest.mark.parametrize(
    "changes",
    [
        {"qty": "10"},
        {"side": "sell"},
        {"symbol": "SPY"},
        {"client_order_id": "other"},
    ],
)
def test_an_answer_about_another_order_stops_the_engine(changes):
    b, _ = broker((200, alpaca_order() | changes))
    with pytest.raises(BrokerError, match="answered the send"):
        b.submit(make_order())


def test_an_order_filled_on_arrival_reports_its_fills():
    b, _ = broker((200, alpaca_order("filled", filled="100", average="600.05")))
    assert b.submit(make_order()) == [FillReport(ID, UPDATED, 100.0, 600.05)]


# What a status means.


@pytest.mark.parametrize(
    ("status", "filled", "average", "events"),
    [
        ("new", "0", None, [Accepted(ID, UPDATED)]),
        ("accepted", "0", None, [Accepted(ID, UPDATED)]),
        ("pending_new", "0", None, [Accepted(ID, UPDATED)]),
        (
            "partially_filled",
            "40",
            "600.01",
            [Accepted(ID, UPDATED), FillReport(ID, UPDATED, 40.0, 600.01)],
        ),
        (
            "pending_cancel",
            "40",
            "600.01",
            [Accepted(ID, UPDATED), FillReport(ID, UPDATED, 40.0, 600.01)],
        ),
        ("filled", "100", "600.02", [FillReport(ID, UPDATED, 100.0, 600.02)]),
        ("canceled", "0", None, [Cancelled(ID, UPDATED)]),
        (
            "canceled",
            "40",
            "600.01",
            [FillReport(ID, UPDATED, 40.0, 600.01), Cancelled(ID, UPDATED)],
        ),
        ("expired", "0", None, [Expired(ID, UPDATED)]),
        ("done_for_day", "0", None, [Expired(ID, UPDATED)]),
        ("rejected", "0", None, [Rejected(ID, UPDATED)]),
    ],
)
def test_each_status_becomes_the_events_it_implies(status, filled, average, events):
    assert AlpacaOrder.parse(alpaca_order(status, filled, average)).events() == events


@pytest.mark.parametrize(
    "status",
    ["replaced", "pending_replace", "stopped", "suspended", "calculated", "held"],
)
def test_a_status_the_engine_does_not_use_stops_it(status):
    with pytest.raises(BrokerError, match=status):
        AlpacaOrder.parse(alpaca_order(status)).events()


def test_a_filled_order_with_shares_missing_stops_the_engine():
    with pytest.raises(BrokerError, match="filled with 99.0 of 100.0"):
        AlpacaOrder.parse(alpaca_order("filled", "99", "600.0")).events()


def test_fills_without_a_price_stop_the_engine():
    with pytest.raises(BrokerError, match="no price"):
        AlpacaOrder.parse(alpaca_order("partially_filled", "40", None)).events()


@pytest.mark.parametrize(
    "changes",
    [
        {"type": "market", "order_type": "market"},
        {"order_class": "bracket"},
        {"qty": None, "notional": "500"},  # an order for dollars, not shares
        {"side": "sell_short"},
        {"updated_at": "yesterday"},
        {"updated_at": "2026-10-19T14:00:01"},  # no zone
        {"filled_qty": "lots"},
    ],
)
def test_an_order_the_engine_never_sends_cannot_be_read(changes):
    with pytest.raises(BrokerError):
        AlpacaOrder.parse(alpaca_order() | changes)


def test_alpaca_s_times_become_utc():
    order = AlpacaOrder.parse(
        alpaca_order(updated_at="2026-10-19T10:00:01.987654321-04:00")
    )
    assert order.updated_at == UPDATED and order.updated_at.tzinfo is UTC


def test_the_same_status_reported_twice_changes_nothing():
    reported = AlpacaOrder.parse(alpaca_order("partially_filled", "40", "600.01"))
    once = replay(make_order(), [Sent(ID, T), *reported.events()])
    assert once.status is Status.PARTLY_FILLED
    twice = once
    for event in reported.events():
        twice = apply(twice, event)
    assert twice == once


def test_a_status_that_contradicts_the_engine_s_order_stops_it():
    # Alpaca says rejected, after the engine saw fills.
    order = replay(
        make_order(),
        [
            Sent(ID, T),
            *AlpacaOrder.parse(
                alpaca_order("partially_filled", "40", "600.01")
            ).events(),
        ],
    )
    with pytest.raises(InvalidTransition):
        for event in AlpacaOrder.parse(alpaca_order("rejected")).events():
            order = apply(order, event)


# The trade updates stream.


def update(event, order, timestamp="2026-10-19T14:00:02.5Z"):
    return {"event": event, "timestamp": timestamp, "order": order}


AT = datetime(2026, 10, 19, 14, 0, 2, 500000, tzinfo=UTC)


def test_a_fill_update_reports_the_order_s_totals():
    message = update("partial_fill", alpaca_order("partially_filled", "40", "600.01"))
    message |= {"price": "600.00", "qty": "15", "execution_id": "e1"}
    assert trade_update_events(message) == [
        Accepted(ID, AT),
        FillReport(ID, AT, 40.0, 600.01),
    ]


def test_a_refused_cancel_comes_after_what_the_order_shows():
    message = update("order_cancel_rejected", alpaca_order("filled", "100", "600.02"))
    assert trade_update_events(message) == [
        FillReport(ID, AT, 100.0, 600.02),
        CancelRefused(ID, AT, "cancel rejected"),
    ]


def test_an_update_without_a_time_takes_the_order_s():
    message = update("canceled", alpaca_order("canceled"), timestamp=None)
    assert trade_update_events(message) == [Cancelled(ID, UPDATED)]


@pytest.mark.parametrize("event", ["replaced", "trade_bust", "stopped", None])
def test_an_update_the_engine_does_not_expect_stops_it(event):
    with pytest.raises(BrokerError):
        trade_update_events(update(event, alpaca_order()))


@pytest.mark.parametrize("timestamp", ["2026-10-19T14:00:02", "soon", 5])
def test_an_update_whose_time_cannot_be_read_stops_the_engine(timestamp):
    with pytest.raises(BrokerError, match="time"):
        trade_update_events(update("new", alpaca_order("new"), timestamp=timestamp))


def test_an_update_without_its_order_stops_the_engine():
    with pytest.raises(BrokerError, match="without its order"):
        trade_update_events({"event": "fill", "timestamp": "2026-10-19T14:00:02Z"})


# Asking for orders.


def test_an_order_is_asked_for_by_its_client_id():
    b, fake = broker((200, alpaca_order("new")))
    assert b.order(ID).status == "new"
    assert fake.call(0) == (
        "GET",
        f"/v2/orders:by_client_order_id?client_order_id={ID}",
        None,
    )


def test_an_order_alpaca_never_got_is_none():
    b, _ = broker((404, {"message": "order not found"}))
    assert b.order(ID) is None


def test_an_answer_about_another_client_id_stops_the_engine():
    b, _ = broker((200, alpaca_order(client_order_id="other")))
    with pytest.raises(BrokerError, match="other"):
        b.order(ID)


def test_a_read_is_retried_after_a_server_error_or_a_lost_connection():
    b, fake = broker(
        (503, None),
        urllib.error.URLError("connection refused"),
        (429, None),
        (200, alpaca_order()),
    )
    assert b.order(ID).client_order_id == ID
    assert fake.sleeps == [2, 4, 8]


def test_a_read_gives_up_after_its_attempts_without_the_keys():
    b, fake = broker(*[(503, {"message": "unavailable"})] * ATTEMPTS)
    with pytest.raises(BrokerError, match="unavailable") as error:
        b.order(ID)
    assert len(fake.requests) == ATTEMPTS
    assert KEY_ID not in str(error.value) and SECRET not in str(error.value)
    assert error.value.__cause__ is None and error.value.__context__ is None


def test_open_orders_are_listed_oldest_first():
    b, fake = broker(
        (
            200,
            [
                alpaca_order("new"),
                alpaca_order("partially_filled", "5", "600.0", client_order_id="b"),
            ],
        )
    )
    orders = b.open_orders()
    assert [o.client_order_id for o in orders] == [ID, "b"]
    method, path, _ = fake.call(0)
    assert method == "GET" and path.startswith("/v2/orders?")
    assert dict(urllib.parse.parse_qsl(path.split("?")[1])) == {
        "status": "open",
        "limit": str(MAX_OPEN_ORDERS),
        "direction": "asc",
    }


def test_a_full_page_of_open_orders_stops_the_engine():
    many = [alpaca_order(client_order_id=f"o{i}") for i in range(MAX_OPEN_ORDERS)]
    b, _ = broker((200, many))
    with pytest.raises(BrokerError, match="open orders"):
        b.open_orders()


# Cancelling.


def test_a_cancel_uses_alpaca_s_id_from_the_send():
    b, fake = broker((200, alpaca_order()), (204, None))
    b.submit(make_order())
    assert b.cancel(ID) == []  # the trade updates confirm it
    assert fake.call(1) == ("DELETE", f"/v2/orders/{BROKER_ID}", None)


def test_a_cancel_after_a_restart_asks_for_alpaca_s_id_first():
    b, fake = broker((200, alpaca_order()), (204, None))
    assert b.cancel(ID) == []
    assert fake.call(0)[0] == "GET" and fake.call(1)[0] == "DELETE"


def test_a_cancel_alpaca_refuses_is_reported():
    b, _ = broker((200, alpaca_order()), (422, {"message": "order is not cancelable"}))
    b.submit(make_order())
    assert b.cancel(ID) == [CancelRefused(ID, NOW, "order is not cancelable")]


def test_a_cancel_of_an_order_alpaca_does_not_have_stops_the_engine():
    b, _ = broker((404, {"message": "order not found"}))
    with pytest.raises(BrokerError, match="no such order"):
        b.cancel(ID)
    b, _ = broker((200, alpaca_order()), (404, {"message": "order not found"}))
    b.submit(make_order())
    with pytest.raises(BrokerError, match="not found"):
        b.cancel(ID)


@pytest.mark.parametrize("lost", [TimeoutError("timed out"), (502, None)])
def test_a_cancel_whose_answer_is_lost_is_never_repeated(lost):
    b, fake = broker((200, alpaca_order()), lost)
    b.submit(make_order())
    with pytest.raises(OutcomeUnknown, match=ID):
        b.cancel(ID)
    assert len(fake.requests) == 2


# The account.


def test_positions_are_signed_by_side():
    b, _ = broker(
        (
            200,
            [
                {"symbol": "QQQ", "qty": "-50", "side": "short"},
                {"symbol": "SPY", "qty": "0.5", "side": "long"},
            ],
        )
    )
    assert b.positions() == {"QQQ": -50.0, "SPY": 0.5}


@pytest.mark.parametrize(
    "positions",
    [
        [{"symbol": "QQQ", "qty": "5", "side": "long"}] * 2,
        [{"symbol": "QQQ", "qty": "5", "side": "both"}],
        [{"symbol": "QQQ", "side": "long"}],
    ],
)
def test_positions_that_cannot_be_read_stop_the_engine(positions):
    b, _ = broker((200, positions))
    with pytest.raises(BrokerError):
        b.positions()


# The fields as the paper account answered on 17 October 2026.
ACCOUNT = {
    "status": "ACTIVE",
    "currency": "USD",
    "cash": "30000",
    "equity": "30000",
    "last_equity": "30000",
    "buying_power": "120000",
    "multiplier": "4",
    "trading_blocked": False,
    "account_blocked": False,
    "trade_suspended_by_user": False,
    "shorting_enabled": True,
}


def test_the_account_s_money_and_whether_it_can_trade():
    b, fake = broker((200, ACCOUNT))
    account = b.account()
    assert (account.cash, account.equity, account.buying_power) == (
        30000.0,
        30000.0,
        120000.0,
    )
    assert account.can_trade and account.shorting_enabled
    assert fake.call(0) == ("GET", "/v2/account", None)


@pytest.mark.parametrize(
    "changes",
    [
        {"trading_blocked": True},
        {"account_blocked": True},
        {"trade_suspended_by_user": True},
        {"status": "ACCOUNT_UPDATED"},
    ],
)
def test_a_blocked_account_cannot_trade(changes):
    b, _ = broker((200, ACCOUNT | changes))
    assert not b.account().can_trade


def test_an_account_that_cannot_be_read_stops_the_engine():
    b, _ = broker((200, {"status": "ACTIVE"}))
    with pytest.raises(BrokerError):
        b.account()
    b, _ = broker((401, {"message": "unauthorized"}))
    with pytest.raises(BrokerError, match="401"):
        b.account()
