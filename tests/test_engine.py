from dataclasses import replace
from datetime import UTC, date, datetime, timedelta

import pytest

from qfin_alphaguard.backtest.engine import (
    BacktestError,
    LookAheadError,
    SessionClose,
    run_backtest,
)
from qfin_alphaguard.events import (
    Bar,
    Cancellation,
    Fill,
    Order,
    OrderIntent,
    RiskAction,
    RiskEvent,
    Side,
)
from qfin_alphaguard.guard.check import GuardDecision
from qfin_alphaguard.strategy import Strategy

# 6 October 2026 is a summer-time session: 13:30 to 20:00 UTC.
OPEN = datetime(2026, 10, 6, 13, 30, tzinfo=UTC)
NEXT_OPEN = datetime(2026, 10, 7, 13, 30, tzinfo=UTC)


def at(minute, day_open=OPEN):
    return day_open + timedelta(minutes=minute)


def bar(ticker, minute, open=100.0, close=None, high=None, low=None, day_open=OPEN):
    close = open if close is None else close
    return Bar(
        ticker=ticker,
        start=at(minute, day_open),
        seconds=60,
        open=open,
        high=high if high is not None else max(open, close) + 0.5,
        low=low if low is not None else min(open, close) - 0.5,
        close=close,
        volume=1000,
    )


def buy(view, ticker="A", quantity=10, limit=200.0):
    return OrderIntent(ticker, Side.BUY, quantity, limit, view.now)


def sell(view, ticker="A", quantity=10, limit=1.0):
    return OrderIntent(ticker, Side.SELL, quantity, limit, view.now)


class Script(Strategy):
    """Answers with the decisions planned for each moment and records what it saw."""

    def __init__(self, plan=None):
        self.plan = plan or {}  # {moment: function of the view -> decisions}
        self.calls = []

    def on_session_start(self, view):
        self.calls.append(("start", view.now))

    def on_bars(self, bars, view):
        self.calls.append(("bars", view.now, tuple(bars)))
        decide = self.plan.get(view.now)
        return decide(view) if decide else ()

    def on_fill(self, fill, view):
        self.calls.append(
            ("fill", view.now, fill.client_order_id, dict(view.positions))
        )

    def on_session_end(self, view):
        self.calls.append(("end", view.now))


class NextOpenBroker:
    """Fills a waiting order in full at the next open of its stock, within its limit."""

    def __init__(self, commission=0.35):
        self.commission = commission
        self.waiting = {}
        self.cancelled = []

    def submit(self, order):
        self.waiting[order.client_order_id] = order

    def cancel(self, client_order_id, time):
        self.cancelled.append((client_order_id, time))
        del self.waiting[client_order_id]

    def on_bar(self, bar):
        fills = []
        for client_order_id, order in list(self.waiting.items()):
            intent = order.intent
            if intent.ticker != bar.ticker or order.sent_at > bar.start:
                continue
            if intent.side is Side.BUY:
                within = bar.open <= intent.limit_price
            else:
                within = bar.open >= intent.limit_price
            if within:
                fills.append(self.fill(order, bar))
                del self.waiting[client_order_id]
        return fills

    def fill(self, order, bar):
        intent = order.intent
        return Fill(
            order.client_order_id,
            intent.ticker,
            intent.side,
            intent.quantity,
            bar.open,
            self.commission,
            bar.start,
        )


class AllowAll:
    def check(self, intent, view):
        return GuardDecision(intent)


def run(feed, strategy, broker=None, guard=None, cash=10_000.0):
    return run_backtest(
        feed, strategy, broker or NextOpenBroker(), guard or AllowAll(), cash
    )


def records(result, kind):
    return [record for record in result.log if isinstance(record, kind)]


# --- Time and the look-ahead rule ------------------------------------------


def test_the_strategy_sees_each_slice_once_all_its_bars_have_ended():
    strategy = Script()
    feed = [bar("A", 0), bar("B", 0), bar("A", 1), bar("B", 1)]

    run(feed, strategy)

    seen = [call for call in strategy.calls if call[0] == "bars"]
    assert seen == [("bars", at(1), ("A", "B")), ("bars", at(2), ("A", "B"))]


def test_an_order_fills_on_the_next_bar_not_on_the_one_it_was_decided_from():
    strategy = Script({at(1): lambda view: [buy(view)]})
    feed = [bar("A", 0, open=100.0), bar("A", 1, open=101.0)]

    result = run(feed, strategy)

    (order,) = records(result, Order)
    (fill,) = records(result, Fill)
    assert order.sent_at == at(1)  # the end of the bar it was decided from
    assert (fill.time, fill.price) == (at(1), 101.0)  # the next bar's open


def test_fills_are_booked_and_reported_before_the_strategy_sees_the_slice():
    strategy = Script({at(1): lambda view: [buy(view)]})

    run([bar("A", 0), bar("A", 1)], strategy)

    fill_call, bars_call = strategy.calls[2], strategy.calls[3]
    assert fill_call == ("fill", at(2), "bt-0000001", {"A": 10})
    assert bars_call[:2] == ("bars", at(2))


def test_a_fill_dated_before_its_order_was_sent_stops_the_backtest():
    class Backdating(NextOpenBroker):
        def fill(self, order, bar):
            return replace(
                super().fill(order, bar), time=order.sent_at - timedelta(minutes=1)
            )

    strategy = Script({at(1): lambda view: [buy(view)]})
    with pytest.raises(LookAheadError, match="before it was sent"):
        run([bar("A", 0), bar("A", 1)], strategy, Backdating())


def test_a_fill_at_a_price_that_never_traded_stops_the_backtest():
    class Invented(NextOpenBroker):
        def fill(self, order, bar):
            return replace(super().fill(order, bar), price=bar.high + 0.01)

    strategy = Script({at(1): lambda view: [buy(view)]})
    with pytest.raises(BacktestError, match="never traded"):
        run([bar("A", 0), bar("A", 1)], strategy, Invented())


def test_a_fill_beyond_the_limit_stops_the_backtest():
    class IgnoresLimits(NextOpenBroker):
        def on_bar(self, bar):
            orders = [o for o in self.waiting.values() if o.intent.ticker == bar.ticker]
            self.waiting.clear()
            return [self.fill(order, bar) for order in orders]

    strategy = Script({at(1): lambda view: [buy(view, limit=100.0)]})
    with pytest.raises(BacktestError, match="limit"):
        run([bar("A", 0), bar("A", 1, open=101.0)], strategy, IgnoresLimits())


def test_fills_beyond_the_order_quantity_stop_the_backtest():
    class Doubles(NextOpenBroker):
        def fill(self, order, bar):
            return replace(super().fill(order, bar), quantity=order.intent.quantity * 2)

    strategy = Script({at(1): lambda view: [buy(view)]})
    with pytest.raises(BacktestError, match="exceeds"):
        run([bar("A", 0), bar("A", 1)], strategy, Doubles())


def test_a_cancelled_order_cannot_fill():
    class KeepsCancelled(NextOpenBroker):
        def cancel(self, client_order_id, time):
            self.cancelled.append((client_order_id, time))  # but leaves it waiting

    strategy = Script(
        {
            at(1): lambda view: [buy(view, limit=1.0)],  # far below the market
            at(2): lambda view: [Cancellation("bt-0000001", view.now)],
            at(3): lambda view: [],
        }
    )
    feed = [bar("A", 0), bar("A", 1), bar("A", 2, open=0.9, close=0.9)]
    with pytest.raises(BacktestError, match="not an open order"):
        run(feed, strategy, KeepsCancelled())


def test_a_decision_dated_before_now_is_refused():
    strategy = Script(
        {at(1): lambda view: [OrderIntent("A", Side.BUY, 10, 200.0, at(0))]}
    )
    with pytest.raises(LookAheadError, match="dated"):
        run([bar("A", 0)], strategy)


def test_bars_out_of_time_order_are_refused():
    with pytest.raises(LookAheadError, match="out of time order"):
        run([bar("A", 1), bar("A", 0)], Script())


def test_two_bars_for_one_stock_at_the_same_time_are_refused():
    with pytest.raises(BacktestError, match="two bars"):
        run([bar("A", 0), bar("A", 0)], Script())


def test_bars_outside_the_regular_session_are_refused():
    pre_market = bar("A", -60)  # 12:30 UTC, before the 13:30 open
    with pytest.raises(BacktestError, match="outside the session"):
        run([pre_market], Script())


def test_bars_on_a_day_the_exchange_was_closed_are_refused():
    christmas = datetime(2025, 12, 25, 14, 30, tzinfo=UTC)
    with pytest.raises(BacktestError, match="closed"):
        run([bar("A", 0, day_open=christmas)], Script())


# --- Guard -----------------------------------------------------------------


def risk(view, intent, action, rule="max_order_usd"):
    return RiskEvent(view.now, rule, action, "test", intent.ticker)


class Reject:
    def check(self, intent, view):
        return GuardDecision(None, risk(view, intent, RiskAction.REJECT))


class Halve:
    def check(self, intent, view):
        smaller = replace(intent, quantity=intent.quantity // 2)
        return GuardDecision(smaller, risk(view, intent, RiskAction.REDUCE))


class HaltOnSecond:
    """Approves the first intent and halts trading at the second."""

    def __init__(self):
        self.asked = 0

    def check(self, intent, view):
        self.asked += 1
        if self.asked == 1:
            return GuardDecision(intent)
        event = risk(view, intent, RiskAction.HALT, "daily_loss_limit_pct")
        return GuardDecision(None, event)


def test_a_rejected_intent_never_reaches_the_broker():
    broker = NextOpenBroker()
    strategy = Script({at(1): lambda view: [buy(view)]})

    result = run([bar("A", 0), bar("A", 1)], strategy, broker, Reject())

    assert broker.waiting == {} and records(result, Order) == []
    (event,) = records(result, RiskEvent)
    assert event.action is RiskAction.REJECT


def test_a_reduced_intent_is_sent_with_fewer_shares():
    strategy = Script({at(1): lambda view: [buy(view, quantity=10)]})

    result = run([bar("A", 0), bar("A", 1)], strategy, guard=Halve())

    (order,) = records(result, Order)
    assert order.intent.quantity == 5
    assert result.positions == {"A": 5}


def test_a_halt_cancels_open_orders_and_refuses_later_intents():
    guard = HaltOnSecond()
    strategy = Script(
        {
            at(1): lambda view: [buy(view, limit=1.0)],  # stays open, far below
            at(2): lambda view: [buy(view)],  # halts trading
            at(3): lambda view: [buy(view)],  # refused without asking Guard
        }
    )
    feed = [bar("A", 0), bar("A", 1), bar("A", 2)]

    result = run(feed, strategy, NextOpenBroker(), guard)

    assert guard.asked == 2
    (cancelled,) = records(result, Cancellation)
    assert (cancelled.client_order_id, cancelled.reason) == ("bt-0000001", "halt")
    assert [event.rule for event in records(result, RiskEvent)] == [
        "daily_loss_limit_pct",
        "halt",
    ]
    assert len(records(result, Order)) == 1


@pytest.mark.parametrize(
    "change",
    [
        lambda intent: replace(intent, limit_price=intent.limit_price + 1),
        lambda intent: replace(intent, quantity=intent.quantity + 1),
        lambda intent: replace(intent, ticker="B"),
    ],
)
def test_guard_may_only_reduce_the_quantity(change):
    class Changes:
        def check(self, intent, view):
            return GuardDecision(change(intent), risk(view, intent, RiskAction.REDUCE))

    strategy = Script({at(1): lambda view: [buy(view)]})
    with pytest.raises(BacktestError, match="only reduce"):
        run([bar("A", 0), bar("A", 1)], strategy, guard=Changes())


def test_guard_cannot_change_an_intent_without_saying_why():
    class Quiet:
        def check(self, intent, view):
            return GuardDecision(replace(intent, quantity=1))

    strategy = Script({at(1): lambda view: [buy(view)]})
    with pytest.raises(BacktestError, match="without saying why"):
        run([bar("A", 0), bar("A", 1)], strategy, guard=Quiet())


def test_guard_cannot_date_its_event_in_the_past():
    class Backdates:
        def check(self, intent, view):
            event = RiskEvent(at(0), "max_order_usd", RiskAction.REJECT, "test")
            return GuardDecision(None, event)

    strategy = Script({at(1): lambda view: [buy(view)]})
    with pytest.raises(LookAheadError, match="dated its event"):
        run([bar("A", 0)], strategy, guard=Backdates())


# --- Orders, sessions and the books ----------------------------------------


def test_a_strategy_can_cancel_its_open_order():
    broker = NextOpenBroker()
    strategy = Script(
        {
            at(1): lambda view: [buy(view, limit=1.0)],
            at(2): lambda view: [
                Cancellation(view.open_orders[0].client_order_id, view.now)
            ],
        }
    )

    result = run([bar("A", 0), bar("A", 1)], strategy, broker)

    assert broker.cancelled == [("bt-0000001", at(2))]
    (cancelled,) = records(result, Cancellation)
    assert cancelled.client_order_id == "bt-0000001"


def test_cancelling_an_order_that_is_not_open_is_refused():
    strategy = Script({at(1): lambda view: [Cancellation("bt-0000009", view.now)]})
    with pytest.raises(BacktestError, match="not open"):
        run([bar("A", 0)], strategy)


def test_an_answer_that_is_not_a_decision_is_refused():
    strategy = Script({at(1): lambda view: ["buy A"]})
    with pytest.raises(TypeError, match="OrderIntent or Cancellation"):
        run([bar("A", 0)], strategy)


def test_open_orders_are_cancelled_at_the_close_and_each_day_is_valued():
    broker = NextOpenBroker()
    strategy = Script({at(1): lambda view: [buy(view, limit=1.0)]})  # never fills
    feed = [bar("A", 0), bar("A", 0, day_open=NEXT_OPEN)]

    result = run(feed, strategy, broker)

    close = datetime(2026, 10, 6, 20, 0, tzinfo=UTC)
    (cancelled,) = records(result, Cancellation)
    assert (cancelled.time, cancelled.reason) == (close, "session close")
    assert [c.day for c in result.closes] == [date(2026, 10, 6), date(2026, 10, 7)]
    assert all(c.value == 10_000.0 for c in result.closes)
    assert [call[0] for call in strategy.calls] == ["start", "bars", "end"] * 2
    assert strategy.calls[0] == ("start", OPEN)


def test_a_half_day_closes_at_1800_utc():
    half_day = datetime(2025, 11, 28, 14, 30, tzinfo=UTC)  # winter time

    result = run([bar("A", 0, day_open=half_day)], Script())

    assert result.closes[0].time == datetime(2025, 11, 28, 18, 0, tzinfo=UTC)


def test_partial_fills_add_up_to_the_order():
    class InThrees(NextOpenBroker):
        def __init__(self):
            super().__init__()
            self.done = []

        def on_bar(self, bar):
            fills = []
            for order in list(self.waiting.values()):
                if order.intent.ticker != bar.ticker:
                    continue
                done = sum(
                    f.quantity
                    for f in self.done
                    if f.client_order_id == order.client_order_id
                )
                part = min(3, order.intent.quantity - done)
                fill = replace(self.fill(order, bar), quantity=part)
                self.done.append(fill)
                fills.append(fill)
                if done + part == order.intent.quantity:
                    del self.waiting[order.client_order_id]
            return fills

    strategy = Script({at(1): lambda view: [buy(view, quantity=7)]})
    feed = [bar("A", minute) for minute in range(5)]

    result = run(feed, strategy, InThrees())

    assert [fill.quantity for fill in records(result, Fill)] == [3, 3, 1]
    assert result.positions == {"A": 7}
    assert records(result, Cancellation) == []  # nothing left open at the close


def test_a_round_trip_books_cash_and_commissions():
    strategy = Script(
        {at(1): lambda view: [buy(view, quantity=10)], at(2): lambda view: [sell(view)]}
    )
    feed = [bar("A", 0), bar("A", 1, open=100.0), bar("A", 2, open=101.0)]

    result = run(feed, strategy, NextOpenBroker(commission=0.35), cash=1_000.0)

    assert result.positions == {}
    assert result.cash == pytest.approx(1_000.0 - 1_000.0 + 1_010.0 - 0.70)
    assert result.closes[-1].value == pytest.approx(result.cash)


def test_a_held_position_is_valued_at_the_last_close():
    strategy = Script({at(1): lambda view: [buy(view, quantity=10)]})
    feed = [bar("A", 0), bar("A", 1, open=100.0, close=105.0)]

    result = run(feed, strategy, NextOpenBroker(commission=0.0), cash=1_000.0)

    assert result.closes[-1].value == pytest.approx(1_000.0 - 1_000.0 + 10 * 105.0)


def test_order_ids_are_numbered_in_the_order_they_are_sent():
    strategy = Script(
        {
            at(1): lambda view: [buy(view, "A"), buy(view, "B")],
            at(2): lambda view: [buy(view, "A")],
        }
    )
    feed = [bar(t, m) for m in range(3) for t in ("A", "B")]

    result = run(feed, strategy)

    ids = [order.client_order_id for order in records(result, Order)]
    assert ids == ["bt-0000001", "bt-0000002", "bt-0000003"]


def test_the_same_inputs_give_the_same_fingerprint_and_others_do_not():
    def once(price):
        strategy = Script(
            {at(1): lambda view: [buy(view)], at(2): lambda view: [sell(view)]}
        )
        feed = [bar("A", 0), bar("A", 1), bar("A", 2, open=price)]
        return run(feed, strategy).fingerprint()

    assert once(101.0) == once(101.0)
    assert once(101.0) != once(102.0)


def test_session_close_records_cash_and_value():
    result = run([bar("A", 0)], Script(), cash=5_000.0)
    assert result.closes == (
        SessionClose(
            date(2026, 10, 6),
            datetime(2026, 10, 6, 20, 0, tzinfo=UTC),
            5_000.0,
            5_000.0,
        ),
    )
