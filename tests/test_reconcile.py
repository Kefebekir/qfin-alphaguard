from datetime import UTC, datetime, timedelta

import pytest

from qfin_alphaguard.events import Order, OrderIntent, Side
from qfin_alphaguard.live.alpaca_broker import AlpacaOrder
from qfin_alphaguard.live.orders import (
    Accepted,
    Cancelled,
    FillReport,
    InvalidTransition,
    LiveOrder,
    Rejected,
    Status,
)
from qfin_alphaguard.live.reconcile import ATTEMPTS, PAUSE, reconcile

T = datetime(2026, 10, 19, 14, 0, tzinfo=UTC)
UPDATED = T + timedelta(seconds=5)
NOW = T + timedelta(minutes=1)
ID = "qqq-2026-10-19-1"


def make_order(cid=ID, side=Side.BUY, quantity=100.0):
    return Order(cid, OrderIntent("QQQ", side, quantity, 600.12, T), T)


def engine(status, filled=0.0, cid=ID, side=Side.BUY):
    return LiveOrder(make_order(cid, side), status, filled, 600.0 if filled else None)


def alpaca(status, filled=0.0, cid=ID, side=Side.BUY, quantity=100.0, symbol="QQQ"):
    return AlpacaOrder(
        broker_id="b-" + cid,
        client_order_id=cid,
        symbol=symbol,
        side=side,
        quantity=quantity,
        status=status,
        filled_quantity=filled,
        average_price=600.0 if filled else None,
        updated_at=UPDATED,
    )


class FakeBroker:
    """Answers as Alpaca would, call by call. Each answer is a list of what the first,
    second... call returns; the last one repeats, as Alpaca once it has caught up."""

    def __init__(self, orders=None, open_orders=([],), positions=({},)):
        self._orders = {cid: list(answers) for cid, answers in (orders or {}).items()}
        self._open = list(open_orders)
        self._positions = list(positions)
        self.asked = []

    @staticmethod
    def _next(answers):
        return answers.pop(0) if len(answers) > 1 else answers[0]

    def order(self, cid):
        self.asked.append(cid)
        return self._next(self._orders[cid]) if cid in self._orders else None

    def open_orders(self):
        return self._next(self._open)

    def positions(self):
        return self._next(self._positions)


def run(orders, broker):
    sleeps = []
    result = reconcile(
        {o.client_order_id: o for o in orders},
        broker,
        sleep=sleeps.append,
        now=lambda: NOW,
    )
    return result, sleeps


def test_an_engine_that_agrees_with_alpaca_is_left_as_it_is():
    order = engine(Status.ACCEPTED)
    broker = FakeBroker({ID: [alpaca("new")]}, open_orders=[[alpaca("new")]])
    result, sleeps = run([order], broker)
    assert result.agrees and result.differences == ()
    assert result.events == ()  # the acceptance is old news
    assert result.orders == {ID: order}
    assert sleeps == [] and broker.asked == [ID]


def test_a_fill_missed_while_disconnected_is_caught_up():
    broker = FakeBroker(
        {ID: [alpaca("partially_filled", 40.0)]},
        open_orders=[[alpaca("partially_filled", 40.0)]],
        positions=[{"QQQ": 40.0}],
    )
    result, _ = run([engine(Status.ACCEPTED)], broker)
    assert result.agrees
    assert result.events == (FillReport(ID, UPDATED, 40.0, 600.0),)
    assert result.orders[ID].status is Status.PARTLY_FILLED


def test_a_whole_fill_and_a_cancel_missed_are_caught_up():
    other = "qqq-2026-10-19-2"
    broker = FakeBroker(
        {ID: [alpaca("filled", 100.0)], other: [alpaca("canceled", cid=other)]},
        positions=[{"QQQ": 100.0}],
    )
    result, _ = run(
        [engine(Status.ACCEPTED), engine(Status.PENDING_CANCEL, cid=other)], broker
    )
    assert result.agrees
    assert result.events == (
        FillReport(ID, UPDATED, 100.0, 600.0),
        Cancelled(other, UPDATED),
    )
    assert result.orders[ID].status is Status.FILLED
    assert result.orders[other].status is Status.CANCELLED


def test_alpaca_s_lag_is_asked_about_again_not_reported():
    # 17 October 2026: the stream had said cancelled; REST still listed the order.
    stale = alpaca("accepted")
    broker = FakeBroker(open_orders=[[stale], []])
    result, sleeps = run([engine(Status.CANCELLED)], broker)
    assert result.agrees and result.events == ()
    assert sleeps == [PAUSE]


def test_positions_that_lag_the_fills_are_asked_about_again():
    broker = FakeBroker(positions=[{}, {"QQQ": 100.0}])
    result, sleeps = run([engine(Status.FILLED, 100.0)], broker)
    assert result.agrees and sleeps == [PAUSE]


def test_a_difference_that_lasts_is_reported_after_every_attempt():
    broker = FakeBroker(open_orders=[[alpaca("accepted")]])
    result, sleeps = run([engine(Status.CANCELLED)], broker)
    assert not result.agrees
    assert result.differences == (f"{ID} is open at Alpaca and cancelled here",)
    assert sleeps == [PAUSE] * (ATTEMPTS - 1)


def test_an_order_the_engine_did_not_make_is_a_difference():
    broker = FakeBroker(open_orders=[[alpaca("new", cid="placed-by-hand")]])
    result, _ = run([], broker)
    assert result.differences == (
        "placed-by-hand is open at Alpaca, an order the engine did not make",
    )


@pytest.mark.parametrize(
    ("mine", "theirs", "difference"),
    [
        ([], {"QQQ": 10.0}, "QQQ: 0.0 shares here, 10.0 at Alpaca"),
        ([engine(Status.FILLED, 100.0)], {}, "QQQ: 100.0 shares here, 0.0 at Alpaca"),
        (
            [engine(Status.FILLED, 100.0)],
            {"QQQ": -100.0},
            "QQQ: 100.0 shares here, -100.0 at Alpaca",
        ),
        ([], {"SPY": 0.5}, "SPY: 0.0 shares here, 0.5 at Alpaca"),
    ],
)
def test_positions_that_differ_are_a_difference(mine, theirs, difference):
    result, _ = run(mine, FakeBroker(positions=[theirs]))
    assert result.differences == (difference,)


def test_a_position_held_from_an_earlier_session_is_counted():
    # Yesterday's close did not fill: 50 shares started the day; 20 were sold since.
    sold = LiveOrder(make_order(ID, Side.SELL, 20.0), Status.FILLED, 20.0, 600.0)
    broker = FakeBroker(positions=[{"QQQ": 30.0}])
    result = reconcile({ID: sold}, broker, {"QQQ": 50.0}, sleep=lambda s: None)
    assert result.agrees
    flat = reconcile(
        {ID: sold}, FakeBroker(positions=[{"QQQ": 30.0}]), sleep=lambda s: None
    )
    assert flat.differences == ("QQQ: -20.0 shares here, 30.0 at Alpaca",)


def test_a_start_and_fills_that_cancel_out_are_flat():
    sold = LiveOrder(make_order(ID, Side.SELL, 50.0), Status.FILLED, 50.0, 600.0)
    result = reconcile({ID: sold}, FakeBroker(), {"QQQ": 50.0}, sleep=lambda s: None)
    assert result.agrees


def test_a_short_position_agrees_with_alpaca_s():
    short = engine(Status.FILLED, 100.0, side=Side.SELL)
    result, _ = run([short], FakeBroker(positions=[{"QQQ": -100.0}]))
    assert result.agrees


def test_an_order_alpaca_never_got_was_not_placed():
    # Its send was lost on the way; asked again and again, Alpaca still has none.
    broker = FakeBroker()
    result, sleeps = run([engine(Status.SUBMITTED)], broker)
    assert result.agrees
    assert result.events == (Rejected(ID, NOW, "never reached Alpaca"),)
    assert result.orders[ID].status is Status.REJECTED
    assert broker.asked == [ID] * ATTEMPTS and sleeps == [PAUSE] * (ATTEMPTS - 1)


def test_an_order_still_on_its_way_is_found_when_asked_again():
    broker = FakeBroker({ID: [None, alpaca("new")]}, open_orders=[[], [alpaca("new")]])
    result, sleeps = run([engine(Status.SUBMITTED)], broker)
    assert result.agrees
    assert result.events == (Accepted(ID, UPDATED),)
    assert sleeps == [PAUSE]


def test_an_order_alpaca_accepted_and_then_lost_is_a_difference():
    result, _ = run([engine(Status.ACCEPTED)], FakeBroker())
    assert result.differences == (f"{ID} is accepted here and unknown to Alpaca",)


@pytest.mark.parametrize(
    "changes",
    [{"quantity": 10.0}, {"side": Side.SELL}, {"symbol": "SPY"}],
)
def test_an_order_alpaca_has_otherwise_is_a_difference(changes):
    broker = FakeBroker({ID: [alpaca("new", **changes)]})
    result, _ = run([engine(Status.ACCEPTED)], broker)
    assert len(result.differences) == 1
    assert "not the order the engine sent" in result.differences[0]


def test_what_the_order_rules_refuse_stops_the_engine():
    broker = FakeBroker({ID: [alpaca("rejected")]})
    with pytest.raises(InvalidTransition):
        run([engine(Status.PARTLY_FILLED, 40.0)], broker)


def test_only_open_orders_are_asked_about():
    finished = [
        engine(Status.FILLED, 100.0, cid="f"),
        engine(Status.CANCELLED, cid="c"),
        engine(Status.REJECTED, cid="r"),
        engine(Status.EXPIRED, cid="e"),
    ]
    broker = FakeBroker(positions=[{"QQQ": 100.0}])
    result, _ = run(finished, broker)
    assert result.agrees and broker.asked == []


def test_an_event_is_kept_once_however_often_it_is_asked_about():
    broker = FakeBroker(
        {ID: [alpaca("partially_filled", 40.0)]},
        positions=[{}, {}, {"QQQ": 40.0}],
    )
    result, sleeps = run([engine(Status.ACCEPTED)], broker)
    assert result.agrees and sleeps == [PAUSE, PAUSE]
    assert result.events == (FillReport(ID, UPDATED, 40.0, 600.0),)


def test_the_engine_s_orders_are_not_changed_in_place():
    orders = {ID: engine(Status.ACCEPTED)}
    broker = FakeBroker({ID: [alpaca("filled", 100.0)]}, positions=[{"QQQ": 100.0}])
    result = reconcile(orders, broker, sleep=lambda s: None)
    assert orders[ID].status is Status.ACCEPTED
    assert result.orders[ID].status is Status.FILLED


def test_at_least_one_attempt_is_made():
    with pytest.raises(ValueError):
        reconcile({}, FakeBroker(), attempts=0)
