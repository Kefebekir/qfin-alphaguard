import math
import random
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta, timezone

import pytest

from qfin_alphaguard.events import Order, OrderIntent, Side
from qfin_alphaguard.live.orders import (
    FINISHED,
    OPEN,
    Accepted,
    Cancelled,
    CancelRefused,
    CancelRequested,
    Expired,
    FillReport,
    InvalidTransition,
    LiveOrder,
    Rejected,
    Sent,
    Status,
    apply,
    replay,
)

T = datetime(2026, 10, 19, 14, 0, tzinfo=UTC)
ID = "qqq-2026-10-19-1"


def make_order(quantity: float = 100.0, client_order_id: str = ID) -> Order:
    intent = OrderIntent(
        ticker="QQQ", side=Side.BUY, quantity=quantity, limit_price=600.0, time=T
    )
    return Order(client_order_id=client_order_id, intent=intent, sent_at=T)


def live(status: Status, filled: float = 0.0) -> LiveOrder:
    return LiveOrder(make_order(), status, filled, 600.0 if filled else None)


def fill(total: float, price: float = 601.0, client_order_id: str = ID) -> FillReport:
    return FillReport(client_order_id, T, filled_quantity=total, average_price=price)


# The rules: what each event does to an order in each status. A cell is the order's
# next status, "=" for no change, or "x" for InvalidTransition.
STATUSES = {
    "new": live(Status.NEW),
    "submitted": live(Status.SUBMITTED),
    "accepted": live(Status.ACCEPTED),
    "partly 40": live(Status.PARTLY_FILLED, 40.0),
    "pending": live(Status.PENDING_CANCEL),
    "pending 40": live(Status.PENDING_CANCEL, 40.0),
    "filled": live(Status.FILLED, 100.0),
    "cancelled 40": live(Status.CANCELLED, 40.0),
    "rejected": live(Status.REJECTED),
    "expired 40": live(Status.EXPIRED, 40.0),
}
EVENTS = {
    "sent": Sent(ID, T),
    "accepted": Accepted(ID, T),
    "rejected": Rejected(ID, T, "insufficient buying power"),
    "fill 40": fill(40.0),
    "fill 70": fill(70.0),
    "fill 100": fill(100.0),
    "fill 120": fill(120.0),
    "cancel": CancelRequested(ID, T),
    "cancelled": Cancelled(ID, T),
    "refused": CancelRefused(ID, T, "order is already filled"),
    "expired": Expired(ID, T),
}
CODES = {
    "sub": Status.SUBMITTED,
    "acc": Status.ACCEPTED,
    "part": Status.PARTLY_FILLED,
    "pend": Status.PENDING_CANCEL,
    "fill": Status.FILLED,
    "canc": Status.CANCELLED,
    "rej": Status.REJECTED,
    "exp": Status.EXPIRED,
}
TABLE = {
    #                sent acc  rej  f40  f70  f100 f120 cncl cncd ref  exp
    "new":          "sub  x    x    x    x    x    x    x    x    x    x",
    "submitted":    "x    acc  rej  part part fill x    x    canc x    exp",
    "accepted":     "x    =    rej  part part fill x    pend canc x    exp",
    "partly 40":    "x    =    x    =    part fill x    pend canc x    exp",
    "pending":      "x    =    x    pend pend fill x    =    canc acc  exp",
    "pending 40":   "x    =    x    =    pend fill x    =    canc part exp",
    "filled":       "x    =    x    =    =    =    x    x    x    =    x",
    "cancelled 40": "x    =    x    =    x    x    x    x    =    =    x",
    "rejected":     "x    x    =    x    x    x    x    x    x    =    x",
    "expired 40":   "x    =    x    =    x    x    x    x    x    =    =",
}  # fmt: skip
CASES = [
    (status, event, cell)
    for status, row in TABLE.items()
    for event, cell in zip(EVENTS, row.split(), strict=True)
]


def test_the_table_covers_every_status():
    assert {o.status for o in STATUSES.values()} == set(Status)


@pytest.mark.parametrize(("status", "event", "cell"), CASES)
def test_each_event_in_each_status(status, event, cell):
    order = STATUSES[status]
    if cell == "x":
        with pytest.raises(InvalidTransition):
            apply(order, EVENTS[event])
        return
    after = apply(order, EVENTS[event])
    if cell == "=":
        assert after == order
        return
    assert after.status is CODES[cell]
    report = EVENTS[event]
    if isinstance(report, FillReport):
        # A new total is recorded, with the average price of every share filled.
        assert after.filled_quantity == report.filled_quantity
        assert after.average_price == report.average_price
    else:
        assert (after.filled_quantity, after.average_price) == (
            order.filled_quantity,
            order.average_price,
        )
    assert after.order == order.order


# The stories behind the rules.


def test_an_order_sent_accepted_and_filled_in_two_parts():
    events = [Sent(ID, T), Accepted(ID, T), fill(30.0, 600.0), fill(100.0, 600.7)]
    order = replay(make_order(), events)
    assert order.status is Status.FILLED
    assert (order.filled_quantity, order.average_price) == (100.0, 600.7)
    assert not order.is_open
    assert order.remaining == 0


def test_an_order_is_never_sent_twice():
    # The answer to a send can be lost. The engine then asks Alpaca for the order by
    # its client order id; it never sends it again blindly.
    with pytest.raises(InvalidTransition):
        replay(make_order(), [Sent(ID, T), Sent(ID, T)])


def test_a_fill_can_arrive_before_the_order_is_accepted():
    order = replay(make_order(), [Sent(ID, T), fill(100.0)])
    assert order.status is Status.FILLED
    # The acceptance, arriving late, is old news.
    assert apply(order, Accepted(ID, T)) == order


def test_a_report_received_twice_changes_nothing():
    once = replay(make_order(), [Sent(ID, T), Accepted(ID, T), fill(40.0)])
    twice = replay(make_order(), [Sent(ID, T), Accepted(ID, T), fill(40.0), fill(40.0)])
    assert once == twice


def test_a_fill_missed_while_disconnected_is_covered_by_the_next_total():
    # Fills of 30 and 20 shares; the one that took the total to 50 was lost.
    order = replay(
        make_order(), [Sent(ID, T), Accepted(ID, T), fill(30.0), fill(70.0, 600.4)]
    )
    assert (order.status, order.filled_quantity, order.average_price) == (
        Status.PARTLY_FILLED,
        70.0,
        600.4,
    )
    assert order.remaining == 30.0


def test_an_order_waiting_for_its_cancel_is_still_open_and_can_fill():
    # Decision 0019: until Alpaca confirms a cancel, Guard counts the order as open,
    # so a closing order sent meanwhile cannot take the position through zero.
    order = replay(
        make_order(), [Sent(ID, T), Accepted(ID, T), fill(40.0), CancelRequested(ID, T)]
    )
    assert order.status is Status.PENDING_CANCEL and order.is_open
    order = apply(order, fill(70.0))
    assert order.status is Status.PENDING_CANCEL and order.remaining == 30.0
    order = apply(order, Cancelled(ID, T))
    assert order.status is Status.CANCELLED and not order.is_open
    assert order.filled_quantity == 70.0  # the shares filled before stay


def test_a_fill_that_wins_the_race_against_a_cancel():
    order = replay(
        make_order(),
        [
            Sent(ID, T),
            Accepted(ID, T),
            CancelRequested(ID, T),
            fill(100.0),
            CancelRefused(ID, T, "order is already filled"),
        ],
    )
    assert order.status is Status.FILLED


def test_a_refused_cancel_puts_the_order_back_to_work():
    start = [Sent(ID, T), Accepted(ID, T)]
    refused = [CancelRequested(ID, T), CancelRefused(ID, T)]
    assert replay(make_order(), start + refused).status is Status.ACCEPTED
    partly = replay(make_order(), [*start, fill(40.0), *refused])
    assert partly.status is Status.PARTLY_FILLED


def test_an_order_not_yet_accepted_cannot_be_cancelled():
    # Alpaca cancels by its own order id, which the engine has only once the order is
    # accepted. An order whose send is unanswered is first asked for, then cancelled.
    with pytest.raises(InvalidTransition):
        replay(make_order(), [Sent(ID, T), CancelRequested(ID, T)])


def test_more_shares_than_ordered_stop_the_engine():
    order = replay(make_order(), [Sent(ID, T), Accepted(ID, T)])
    with pytest.raises(InvalidTransition):
        apply(order, fill(100.000000001))


def test_a_fill_after_a_confirmed_cancel_stops_the_engine():
    order = replay(
        make_order(),
        [Sent(ID, T), Accepted(ID, T), CancelRequested(ID, T), Cancelled(ID, T)],
    )
    with pytest.raises(InvalidTransition):
        apply(order, fill(10.0))


def test_a_day_order_left_at_the_close_expires_with_its_fills():
    order = replay(
        make_order(), [Sent(ID, T), Accepted(ID, T), fill(25.0), Expired(ID, T)]
    )
    assert (order.status, order.filled_quantity) == (Status.EXPIRED, 25.0)


def test_fractional_shares_fill_to_the_last_decimal():
    order = make_order(quantity=0.123456789)
    after = replay(order, [Sent(ID, T), fill(0.1), fill(0.123456789)])
    assert after.status is Status.FILLED and after.remaining == 0


# Random reports in any order: the rules either take a report or refuse it, and never
# leave an order its own checks reject.

POOL = [
    Sent(ID, T),
    Accepted(ID, T),
    Rejected(ID, T),
    CancelRequested(ID, T),
    Cancelled(ID, T),
    CancelRefused(ID, T),
    Expired(ID, T),
    fill(10.0, 600.1),
    fill(35.5, 600.2),
    fill(60.0, 600.3),
    fill(99.999999999, 600.4),
    fill(100.0, 600.5),
    fill(130.0, 600.6),
]


def test_random_reports_never_make_an_impossible_order():
    rng = random.Random(21)
    taken = refused = 0
    for _ in range(3000):
        order = LiveOrder(make_order())
        for event in rng.choices(POOL, k=rng.randint(1, 12)):
            try:
                after = apply(order, event)
            except InvalidTransition:
                refused += 1
                continue
            taken += 1
            assert isinstance(after, LiveOrder) and after.order == order.order
            assert after.filled_quantity >= order.filled_quantity
            if order.status in FINISHED:
                assert after == order  # an ended order stays as it ended
            if order.filled_quantity > 0:
                assert after.status not in (
                    Status.NEW,
                    Status.SUBMITTED,
                    Status.ACCEPTED,
                )
            order = after
    assert taken > 1000 and refused > 1000


# The order and the reports check themselves.


def test_apply_refuses_another_order_s_report():
    order = live(Status.ACCEPTED)
    with pytest.raises(InvalidTransition, match="other-id"):
        apply(order, fill(10.0, client_order_id="other-id"))


def test_replay_of_no_events_is_a_new_order():
    assert replay(make_order(), []) == LiveOrder(make_order())


def test_a_live_order_is_frozen():
    with pytest.raises(FrozenInstanceError):
        live(Status.NEW).status = Status.FILLED


def test_open_and_finished_cover_every_status_but_new():
    assert OPEN | FINISHED == set(Status) - {Status.NEW}
    assert not OPEN & FINISHED


@pytest.mark.parametrize(
    ("status", "filled"),
    [
        (Status.NEW, 10.0),
        (Status.SUBMITTED, 10.0),
        (Status.ACCEPTED, 10.0),
        (Status.REJECTED, 10.0),
        (Status.PARTLY_FILLED, 0.0),
        (Status.PARTLY_FILLED, 100.0),
        (Status.FILLED, 99.0),
        (Status.PENDING_CANCEL, 100.0),
        (Status.CANCELLED, 100.0),
        (Status.EXPIRED, 100.0),
    ],
)
def test_a_status_that_contradicts_the_fills_is_refused(status, filled):
    with pytest.raises(ValueError):
        live(status, filled)


@pytest.mark.parametrize(
    ("filled", "price"),
    [
        (0.0, 600.0),  # a price without fills
        (10.0, None),  # fills without a price
        (10.0, 0.0),
        (10.0, math.inf),
        (101.0, 600.0),  # more than ordered
        (-1.0, 600.0),
        (math.nan, 600.0),
        (0.1234567891, 600.0),  # more decimals than Alpaca takes
    ],
)
def test_impossible_fills_are_refused(filled, price):
    # A cancelled order may have any number of shares filled but all, so only these
    # checks can refuse them.
    with pytest.raises(ValueError):
        LiveOrder(make_order(), Status.CANCELLED, filled, price)


def test_a_status_must_be_a_status():
    with pytest.raises(TypeError):
        LiveOrder(make_order(), "filled", 100.0, 600.0)


# Wrong on purpose: a time without a zone, and New York's time in October.
NAIVE = datetime(2026, 10, 19, 14, 0)  # noqa: DTZ001
NEW_YORK = datetime(2026, 10, 19, 10, 0, tzinfo=timezone(timedelta(hours=-4)))


@pytest.mark.parametrize("time", [NAIVE, NEW_YORK])
def test_a_report_s_time_must_be_utc(time):
    with pytest.raises(ValueError):
        Accepted(ID, time)


def test_a_report_needs_an_order_id():
    with pytest.raises(ValueError):
        Cancelled("", T)


@pytest.mark.parametrize(
    ("total", "price"),
    [
        (0.0, 600.0),
        (-5.0, 600.0),
        (math.nan, 600.0),
        (True, 600.0),
        (10.0, 0.0),
        (10.0, math.nan),
    ],
)
def test_a_fill_report_needs_shares_and_a_price(total, price):
    with pytest.raises(ValueError):
        FillReport(ID, T, filled_quantity=total, average_price=price)
