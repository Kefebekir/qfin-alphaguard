import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from qfin_alphaguard.events import (
    Bar,
    Cancellation,
    DecisionNote,
    Fill,
    Order,
    OrderIntent,
    RiskAction,
    RiskEvent,
    Side,
)
from qfin_alphaguard.guard.check import GuardDecision
from qfin_alphaguard.guard.config import load_guard_config
from qfin_alphaguard.guard.rules import Guard
from qfin_alphaguard.live.engine import (
    Cancel,
    EngineError,
    Flatten,
    LiveEngine,
    SessionStart,
    Step,
    Stop,
    Submit,
    Tick,
    replay_day,
)
from qfin_alphaguard.live.journal import Journal
from qfin_alphaguard.live.orders import (
    Accepted,
    Cancelled,
    CancelRefused,
    CancelRequested,
    FillReport,
    InvalidTransition,
    Sent,
    Status,
)
from qfin_alphaguard.strategy import Strategy

DAY = date(2026, 10, 19)
OPEN = datetime(2026, 10, 19, 13, 30, tzinfo=UTC)  # 09:30 in New York
CLOSE = datetime(2026, 10, 19, 20, 0, tzinfo=UTC)
START = SessionStart(DAY, OPEN - timedelta(minutes=5), 30_000.0, 30_000.0)
QQQ_GUARD = load_guard_config(Path(__file__).parents[1] / "guard_qqq.yaml")
ID1 = "qqq-20261019-0001"
ID2 = "qqq-20261019-0002"


def bar(minute, close=600.0, ticker="QQQ"):
    return Bar(
        ticker,
        OPEN + timedelta(minutes=minute),
        60,
        close,
        close + 0.5,
        close - 0.5,
        close,
        1000,
    )


def seen(minute, delay=0.2):
    """When the engine sees the bar of `minute`: just after it ends."""
    return OPEN + timedelta(minutes=minute + 1, seconds=delay)


class Scripted(Strategy):
    """A strategy that answers each call with what `script(bars, view)` returns."""

    def __init__(self, script=lambda bars, view: ()):
        self.script = script
        self.starts, self.fills, self.calls = [], [], []

    def on_session_start(self, view):
        self.starts.append((view.now, view.cash, dict(view.positions)))

    def on_bars(self, bars, view):
        self.calls.append((view.now, dict(bars)))
        return list(self.script(bars, view))

    def on_fill(self, fill, view):
        self.fills.append((fill, dict(view.positions)))


class Approve:
    def __init__(self):
        self.views = []

    def check(self, intent, view):
        self.views.append(view)
        return GuardDecision(intent)


def buy(quantity, view, limit=603.0, reason="decision"):
    return OrderIntent("QQQ", Side.BUY, quantity, limit, view.now, reason)


def sell(quantity, view, limit=597.0, reason="decision"):
    return OrderIntent("QQQ", Side.SELL, quantity, limit, view.now, reason)


def once(*answers):
    """A script that answers its n-th call with answers[n], then nothing."""
    calls = iter(answers)

    def script(bars, view):
        answer = next(calls, ())
        return answer(view) if callable(answer) else answer

    return script


def engine(script=lambda bars, view: (), guard=None, start=START):
    strategy = Scripted(script)
    return LiveEngine(strategy, guard or Approve(), start, collar_bps=50.0), strategy


# Orders out.


def test_a_decision_becomes_an_order_sent_once_written():
    core, _ = engine(once(lambda v: [buy(50, v)]))
    step = core.on_bars([bar(30)], seen(30))
    intent = buy(50, core.view(seen(30)))
    order = Order(ID1, intent, sent_at=seen(30))
    assert step.records == (intent, order, Sent(ID1, seen(30)))
    assert step.actions == (Submit(order),)
    assert core.orders[ID1].status is Status.SUBMITTED


def test_order_ids_are_numbered_by_day():
    core, _ = engine(once(lambda v: [buy(10, v)], lambda v: [sell(10, v)]))
    core.on_bars([bar(30)], seen(30))
    core.on_order_event(FillReport(ID1, seen(30, 1), 10.0, 600.0), seen(30, 1))
    core.on_bars([bar(31)], seen(31))
    assert list(core.orders) == [ID1, ID2]


def test_an_order_waits_while_another_for_its_ticker_is_open():
    core, _ = engine(once(lambda v: [buy(10, v)], lambda v: [sell(10, v)]))
    core.on_bars([bar(30)], seen(30))
    assert core.on_bars([bar(31)], seen(31)).actions == ()
    filled = core.on_order_event(FillReport(ID1, seen(31, 1), 10.0, 600.0), seen(31, 1))
    assert core.portfolio.positions.get("QQQ", 0.0) == 10.0
    assert [type(a) for a in filled.actions] == [Submit]


def test_the_strategy_sees_the_account_it_starts_with():
    start = SessionStart(DAY, OPEN, 29_400.0, 30_000.0, {"QQQ": 1.0}, {"QQQ": 600.0})
    core, strategy = engine(start=start)
    assert strategy.starts == [(OPEN, 29_400.0, {"QQQ": 1.0})]
    assert core.view(OPEN).last_prices["QQQ"] == 600.0


def test_now_is_when_the_engine_saw_the_bar():
    # So Guard's stale data rule measures how late the feed is.
    guard = Approve()
    core, _ = engine(once(lambda v: [buy(1, v)]), guard)
    core.on_bars([bar(30)], seen(30, delay=20))
    view = guard.views[0]
    assert view.now == seen(30, delay=20)
    assert view.price_times["QQQ"] == bar(30).end


def test_a_stale_bar_is_refused_new_risk_by_guard():
    core, _ = engine(once(lambda v: [buy(1, v)]), Guard(QQQ_GUARD))
    step = core.on_bars([bar(30)], seen(30, delay=16))  # guard_qqq: 15 seconds
    assert [type(r) for r in step.records] == [OrderIntent, RiskEvent]
    assert step.records[1].rule == "stale_data_sec" and step.actions == ()


@pytest.mark.parametrize(
    ("bars", "now"),
    [
        ([], seen(30)),
        ([bar(30)], bar(30).end - timedelta(seconds=1)),  # before it ended
        ([bar(30), bar(30)], seen(30)),
        ([bar(30), bar(31, ticker="SPY")], seen(31)),
        ([Bar("QQQ", CLOSE, 60, 600, 601, 599, 600, 1)], CLOSE + timedelta(minutes=2)),
    ],
)
def test_bars_that_cannot_be_a_slice_stop_the_engine(bars, now):
    core, _ = engine()
    with pytest.raises(EngineError):
        core.on_bars(bars, now)


def test_a_clock_that_goes_back_keeps_the_last_time():
    core, strategy = engine()
    core.on_bars([bar(30)], seen(30))
    core.on_tick(Tick(seen(30) - timedelta(milliseconds=5)))
    assert strategy.calls[-1][0] == seen(30)


@pytest.mark.parametrize(
    "answer",
    [
        lambda v: [
            OrderIntent("QQQ", Side.BUY, 1, 603.0, v.now - timedelta(seconds=1))
        ],
        lambda v: ["buy"],
        lambda v: [Cancellation("nobody", v.now)],
    ],
)
def test_a_decision_the_engine_cannot_take_stops_it(answer):
    core, _ = engine(once(answer))
    with pytest.raises(EngineError):
        core.on_bars([bar(30)], seen(30))


def test_a_note_goes_into_the_journal_and_nowhere_else():
    def note(v):
        return [
            DecisionNote(
                "QQQ",
                v.now,
                "flat",
                "inside the noise band: stay flat",
                {"price": 600.0},
            )
        ]

    core, _ = engine(once(note))
    step = core.on_bars([bar(30)], seen(30))
    assert [type(r) for r in step.records] == [DecisionNote] and step.actions == ()


def test_a_tick_lets_the_strategy_act_without_a_bar():
    core, strategy = engine(
        once((), lambda v: [sell(1, v, reason="close at the day's end")])
    )
    core.on_bars([bar(30)], seen(30))  # a price to act on
    tick = Tick(CLOSE - timedelta(seconds=55))
    step = core.on_tick(tick)
    assert strategy.calls[-1] == (tick.time, {})
    (submit,) = step.actions
    assert submit.order.sent_at == tick.time


# Guard.


def test_guard_s_rejection_is_written_and_nothing_goes():
    class Reject:
        def check(self, intent, view):
            event = RiskEvent(
                view.now, "max_order_usd", RiskAction.REJECT, "too big", "QQQ"
            )
            return GuardDecision(None, event)

    core, _ = engine(once(lambda v: [buy(500, v)]), Reject())
    step = core.on_bars([bar(30)], seen(30))
    assert [type(r) for r in step.records] == [OrderIntent, RiskEvent]
    assert step.actions == () and core.orders == {}


def test_guard_s_cut_is_the_order_sent():
    core, _ = engine(once(lambda v: [buy(200, v)]), Guard(QQQ_GUARD))
    step = core.on_bars([bar(30)], seen(30))
    (submit,) = step.actions
    assert submit.order.intent.quantity < 200  # max_order_usd 70,000 at 603
    assert any(
        isinstance(r, RiskEvent) and r.action is RiskAction.REDUCE for r in step.records
    )


def test_a_guard_that_breaks_its_contract_stops_the_engine():
    class Reprice:
        def check(self, intent, view):
            event = RiskEvent(view.now, "x", RiskAction.REDUCE, "x", "QQQ")
            cheaper = OrderIntent(
                intent.ticker, intent.side, intent.quantity, 1.0, intent.time
            )
            return GuardDecision(cheaper, event)

    core, _ = engine(once(lambda v: [buy(1, v)]), Reprice())
    with pytest.raises(EngineError, match="only reduce"):
        core.on_bars([bar(30)], seen(30))


def test_a_halt_from_guard_stops_the_engine():
    class Halt:
        def check(self, intent, view):
            event = RiskEvent(view.now, "daily_loss_limit_pct", RiskAction.HALT, "stop")
            return GuardDecision(None, event)

    core, strategy = engine(once(lambda v: [buy(1, v)], lambda v: [buy(1, v)]), Halt())
    step = core.on_bars([bar(30)], seen(30))
    assert [type(r) for r in step.records] == [OrderIntent, RiskEvent, Stop]
    assert core.stopped is not None
    calls = len(strategy.calls)
    assert core.on_bars([bar(31)], seen(31)) == Step()
    assert len(strategy.calls) == calls  # no more decisions
    assert core.last_prices["QQQ"] == bar(31).close  # but prices still follow the feed


# Fills.


def sent_and_accepted(script):
    core, strategy = engine(script)
    core.on_bars([bar(30)], seen(30))
    core.on_order_event(Accepted(ID1, seen(30, 0.3)), seen(30, 0.3))
    return core, strategy


def test_fills_come_as_totals_and_are_booked_as_new_shares():
    core, strategy = sent_and_accepted(once(lambda v: [buy(100, v)]))
    first = core.on_order_event(
        FillReport(ID1, seen(30, 0.4), 40.0, 600.0), seen(30, 0.4)
    )
    second = core.on_order_event(
        FillReport(ID1, seen(30, 0.5), 100.0, 600.3), seen(30, 0.5)
    )
    assert first.records == (
        Fill(ID1, "QQQ", Side.BUY, 40.0, 600.0, 0.0, seen(30, 0.4)),
    )
    (fill,) = second.records
    assert fill.quantity == 60.0 and fill.price == pytest.approx(600.5)
    assert core.portfolio.positions["QQQ"] == 100.0
    assert core.portfolio.cash == pytest.approx(30_000 - 100 * 600.3)
    assert strategy.fills[-1][1] == {"QQQ": 100.0}  # the strategy sees the fill booked
    assert core.orders[ID1].status is Status.FILLED


def test_a_total_received_twice_books_nothing_twice():
    core, _ = sent_and_accepted(once(lambda v: [buy(100, v)]))
    core.on_order_event(FillReport(ID1, seen(30, 0.4), 40.0, 600.0), seen(30, 0.4))
    again = core.on_order_event(
        FillReport(ID1, seen(30, 0.4), 40.0, 600.0), seen(30, 0.5)
    )
    assert again == Step() and core.portfolio.positions["QQQ"] == 40.0


def test_a_tiny_fraction_takes_the_average_price():
    core, _ = engine(once(lambda v: [buy(1.000000001, v)]))
    core.on_bars([bar(30)], seen(30))
    core.on_order_event(FillReport(ID1, seen(30, 1), 1.0, 600.0), seen(30, 1))
    step = core.on_order_event(
        FillReport(ID1, seen(30, 2), 1.000000001, 600.0), seen(30, 2)
    )
    (fill,) = step.records
    assert fill.quantity == 0.000000001 and fill.price > 0


def test_an_event_for_an_order_the_engine_did_not_make_stops_it():
    core, _ = engine()
    with pytest.raises(EngineError, match="did not make"):
        core.on_order_event(Accepted("by-hand", seen(30)), seen(30))


def test_an_event_the_order_rules_refuse_stops_the_engine():
    core, _ = sent_and_accepted(once(lambda v: [buy(100, v)]))
    with pytest.raises(InvalidTransition):
        core.on_order_event(Sent(ID1, seen(31)), seen(31))


# Cancels, and the replacement that waits for them (Efe: like the backtest).


def test_the_strategy_and_guard_see_only_orders_still_open():
    core, _ = engine(
        once(lambda v: [buy(100, v)], lambda v: [Cancellation(ID1, v.now)])
    )
    core.on_bars([bar(30)], seen(30))
    assert [o.client_order_id for o in core.view(seen(30)).open_orders] == [ID1]
    core.on_order_event(Accepted(ID1, seen(30, 0.3)), seen(30, 0.3))
    core.on_bars([bar(31)], seen(31))  # asks for the cancel
    assert [o.client_order_id for o in core.view(seen(31)).open_orders] == [ID1]
    core.on_order_event(Cancelled(ID1, seen(31, 0.3)), seen(31, 0.3))
    assert core.view(seen(31, 0.3)).open_orders == ()


def test_a_cancel_is_asked_for_once():
    core, _ = sent_and_accepted(
        once(
            lambda v: [buy(100, v)],
            lambda v: [Cancellation(ID1, v.now)],
            lambda v: [Cancellation(ID1, v.now)],
        )
    )
    step = core.on_bars([bar(31)], seen(31))
    assert step.records == (Cancellation(ID1, seen(31)), CancelRequested(ID1, seen(31)))
    assert step.actions == (Cancel(ID1),)
    again = core.on_bars([bar(32)], seen(32))
    assert again.actions == ()  # already pending
    assert core.orders[ID1].is_open  # until Alpaca confirms it


def test_a_replacement_waits_for_its_cancel():
    core, _ = sent_and_accepted(
        once(
            lambda v: [buy(100, v)],
            lambda v: [Cancellation(ID1, v.now), buy(80, v, limit=606.0)],
        )
    )
    step = core.on_bars([bar(31, close=604.0)], seen(31))
    assert step.actions == (Cancel(ID1),)  # nothing new goes yet
    done = core.on_order_event(Cancelled(ID1, seen(31, 0.4)), seen(31, 0.4))
    intent, order, _ = done.records
    assert (intent.side, intent.quantity) == (Side.BUY, 80.0)
    assert intent.limit_price == 607.02  # 604 plus the 50 bps collar, on the cent below
    assert intent.time == seen(31, 0.4) and "after its cancel" in intent.reason
    assert done.actions == (Submit(order),) and order.client_order_id == ID2


def test_a_replacement_is_only_what_the_old_order_did_not_fill():
    # The old order bought 30 more shares before its cancel landed.
    core, _ = sent_and_accepted(
        once(lambda v: [buy(100, v)], lambda v: [Cancellation(ID1, v.now), buy(80, v)])
    )
    core.on_order_event(FillReport(ID1, seen(30, 1), 10.0, 600.0), seen(30, 1))
    core.on_bars([bar(32)], seen(32))  # holds 10; wants 10 + 80 = 90
    core.on_order_event(FillReport(ID1, seen(32, 0.3), 40.0, 600.0), seen(32, 0.3))
    done = core.on_order_event(Cancelled(ID1, seen(32, 0.4)), seen(32, 0.4))
    intent = done.records[0]
    assert (intent.side, intent.quantity) == (Side.BUY, 50.0)


def test_an_old_order_that_filled_past_the_target_is_undone():
    # At the day's end the strategy wants flat; the open buy filled while being cancelled.
    core, _ = sent_and_accepted(
        once(lambda v: [buy(100, v)], lambda v: [Cancellation(ID1, v.now)])
    )
    core.on_bars([bar(31)], seen(31))
    filled = core.on_order_event(
        FillReport(ID1, seen(31, 0.3), 100.0, 600.0), seen(31, 0.3)
    )
    intent = filled.records[1]
    assert (intent.side, intent.quantity) == (Side.SELL, 100.0)
    assert filled.actions[0].order.client_order_id == ID2
    refused = core.on_order_event(CancelRefused(ID1, seen(31, 0.4)), seen(31, 0.4))
    assert refused == Step()  # the fill won the race; nothing more


def test_a_refused_cancel_keeps_the_replacement_waiting():
    core, _ = sent_and_accepted(
        once(
            lambda v: [buy(100, v)], lambda v: [Cancellation(ID1, v.now), sell(0.5, v)]
        )
    )
    core.on_bars([bar(31)], seen(31))
    step = core.on_order_event(CancelRefused(ID1, seen(31, 0.4)), seen(31, 0.4))
    assert step == Step() and core.orders[ID1].status is Status.ACCEPTED
    filled = core.on_order_event(
        FillReport(ID1, seen(31, 2), 100.0, 600.0), seen(31, 2)
    )
    intent = filled.records[1]
    assert (intent.side, intent.quantity) == (Side.SELL, 100.5)  # to -0.5, as wanted


def test_a_cancel_that_ends_where_the_strategy_wanted_sends_nothing():
    core, _ = sent_and_accepted(
        once(lambda v: [buy(100, v)], lambda v: [Cancellation(ID1, v.now)])
    )
    core.on_bars([bar(31)], seen(31))
    assert core.on_order_event(Cancelled(ID1, seen(31, 0.4)), seen(31, 0.4)) == Step()
    assert core.waiting == {}


# Stopping (Efe: close, then stop).


def test_an_order_not_yet_accepted_is_cancelled_once_it_is():
    core, _ = engine(
        once(lambda v: [buy(100, v)], lambda v: [Cancellation(ID1, v.now)])
    )
    core.on_bars([bar(30)], seen(30))
    step = core.on_bars([bar(31)], seen(31))
    assert step.actions == () and core.orders[ID1].status is Status.SUBMITTED
    accepted = core.on_order_event(Accepted(ID1, seen(31, 0.3)), seen(31, 0.3))
    assert accepted.records == (CancelRequested(ID1, seen(31, 0.3)),)
    assert accepted.actions == (Cancel(ID1),)


def test_an_order_that_ends_before_it_is_accepted_needs_no_cancel():
    core, _ = engine(
        once(lambda v: [buy(100, v)], lambda v: [Cancellation(ID1, v.now)])
    )
    core.on_bars([bar(30)], seen(30))
    core.on_bars([bar(31)], seen(31))
    step = core.on_order_event(
        FillReport(ID1, seen(31, 0.3), 100.0, 600.0), seen(31, 0.3)
    )
    assert not any(isinstance(a, Cancel) for a in step.actions)
    assert core.cancel_when_accepted == set()


def test_a_stop_cancels_every_open_order_and_ends_decisions():
    core, strategy = sent_and_accepted(once(lambda v: [buy(100, v)]))
    step = core.stop(Stop(seen(31), "reconciliation differs"))
    assert step.records == (
        Stop(seen(31), "reconciliation differs"),
        CancelRequested(ID1, seen(31)),
    )
    assert step.actions == (Cancel(ID1),)
    calls = len(strategy.calls)
    core.on_bars([bar(31)], seen(31, 1))
    assert len(strategy.calls) == calls
    assert core.stop(Stop(seen(32), "again")) == Step()


def test_a_stop_cancels_an_order_not_yet_accepted_once_it_is():
    core, _ = engine(once(lambda v: [buy(100, v)]))
    core.on_bars([bar(30)], seen(30))
    step = core.stop(Stop(seen(30, 0.25), "lost data"))
    assert step.actions == ()  # Alpaca has not said it has the order
    accepted = core.on_order_event(Accepted(ID1, seen(30, 0.3)), seen(30, 0.3))
    assert accepted.actions == (Cancel(ID1),)


def test_after_a_stop_fills_are_booked_but_nothing_is_sent():
    core, _ = sent_and_accepted(
        once(
            lambda v: [buy(100, v)], lambda v: [Cancellation(ID1, v.now), sell(100, v)]
        )
    )
    core.on_bars([bar(31)], seen(31))
    core.stop(Stop(seen(31, 1), "lost data"))
    step = core.on_order_event(FillReport(ID1, seen(31, 2), 30.0, 600.0), seen(31, 2))
    assert [type(r) for r in step.records] == [Fill] and step.actions == ()
    # The cancel confirmed: the sale the strategy wanted is not sent either.
    assert core.on_order_event(Cancelled(ID1, seen(31, 3)), seen(31, 3)) == Step()


def test_flatten_closes_what_alpaca_holds_judged_by_alpaca_s_positions():
    guard = Approve()
    core, _ = sent_and_accepted(once(lambda v: [buy(100, v)]))
    core.on_order_event(FillReport(ID1, seen(30, 1), 100.0, 600.0), seen(30, 1))
    core.on_bars([bar(31, close=602.0)], seen(31))
    core.guard = guard
    core.stop(Stop(seen(31, 1), "reconciliation differs"))
    alpaca = {"QQQ": 70.0}  # the engine believes 100
    step = core.flatten(Flatten(seen(31, 2), alpaca))
    _, intent, order, _ = step.records
    assert (intent.side, intent.quantity, intent.limit_price) == (
        Side.SELL,
        70.0,
        598.99,
    )
    assert intent.reason == "close after a stop"
    assert dict(guard.views[-1].positions) == alpaca
    assert step.actions == (Submit(order),)


def test_flatten_waits_for_open_orders_to_end():
    core, _ = sent_and_accepted(once(lambda v: [buy(100, v)]))
    core.stop(Stop(seen(31), "lost data"))
    step = core.flatten(Flatten(seen(31, 1), {"QQQ": 40.0}))
    assert step.actions == ()  # the buy's cancel is not confirmed
    core.on_order_event(Cancelled(ID1, seen(31, 2)), seen(31, 2))
    assert len(core.flatten(Flatten(seen(31, 3), {"QQQ": 40.0})).actions) == 1


def test_flatten_only_after_a_stop():
    core, _ = engine()
    with pytest.raises(EngineError):
        core.flatten(Flatten(seen(31), {"QQQ": 1.0}))


@pytest.mark.parametrize(
    "start",
    [
        SessionStart(date(2026, 10, 18), OPEN, 30_000.0, 30_000.0),  # a Sunday
        SessionStart(DAY, CLOSE, 30_000.0, 30_000.0),
    ],
)
def test_an_engine_needs_a_session_still_to_trade(start):
    with pytest.raises(EngineError):
        engine(start=start)


# Replaying a day from its journal.


class Runner:
    """Writes as the runner will: what the engine is told, then what it answers."""

    def __init__(self, path, script):
        self.journal = Journal(path)
        self.core, _ = engine(script)
        self.journal.write(START, at=START.time)
        self.actions = []

    def answer(self, step, at):
        for record in step.records:
            self.journal.write(record, at=at)
        self.actions += step.actions

    def bars(self, bars, at):
        for b in bars:
            self.journal.write(b, at=at)
        self.answer(self.core.on_bars(bars, at), at)

    def event(self, event, at):
        self.journal.write(event, at=at)
        self.answer(self.core.on_order_event(event, at), at)

    def stop(self, stop):
        self.answer(self.core.stop(stop), stop.time)


def a_day(path):
    def note(v):
        return DecisionNote(
            "QQQ",
            v.now,
            "long",
            "above the band: buy",
            {"price": 600.0, "lower": float("nan")},
        )

    script = once(
        lambda v: [note(v), buy(100, v)],
        (),
        lambda v: [Cancellation(ID1, v.now), sell(20, v)],
    )
    run = Runner(path, script)
    run.bars([bar(30)], seen(30))
    run.event(Accepted(ID1, seen(30, 0.3)), seen(30, 0.3))
    run.event(FillReport(ID1, seen(30, 0.4), 60.0, 600.1), seen(30, 0.4))
    run.bars([bar(31)], seen(31))
    run.bars([bar(32)], seen(32))
    run.event(Cancelled(ID1, seen(32, 0.5)), seen(32, 0.5))
    run.stop(Stop(seen(33), "a test"))
    run.journal.close()
    return run


def replay(path, script_answers=None):
    with Journal(path) as journal:
        script = once(
            lambda v: [
                DecisionNote(
                    "QQQ",
                    v.now,
                    "long",
                    "above the band: buy",
                    {"price": 600.0, "lower": float("nan")},
                ),
                buy(100, v),
            ],
            (),
            lambda v: [Cancellation(ID1, v.now), sell(20, v)],
        )
        return replay_day(journal.entries, lambda start: engine(script, start=start)[0])


def test_a_day_replayed_from_its_journal_is_the_same_day(tmp_path):
    run = a_day(tmp_path / "day.jsonl")
    again, leftover = replay(tmp_path / "day.jsonl")
    assert leftover == []
    assert again.orders == run.core.orders
    assert (
        dict(again.portfolio.positions)
        == dict(run.core.portfolio.positions)
        == {"QQQ": 60.0}
    )
    assert again.stopped == run.core.stopped
    assert run.actions[1:3] == [Cancel(ID1), Submit(again.orders[ID2].order)]


def test_a_journal_that_does_not_match_its_decisions_stops_the_engine(tmp_path):
    path = tmp_path / "day.jsonl"
    a_day(path)
    lines = path.read_text().splitlines()
    first = next(i for i, line in enumerate(lines) if '"type":"OrderIntent"' in line)
    record = json.loads(lines[first])
    assert record["quantity"] == 100
    lines[first] = json.dumps({**record, "quantity": 90}, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(EngineError, match="the journal holds"):
        replay(path)


def test_a_step_cut_short_by_a_crash_is_returned_unwritten(tmp_path):
    path = tmp_path / "day.jsonl"
    a_day(path)
    lines = path.read_text().splitlines(keepends=True)
    # Cut after the first bar: its decision's note, intent, order and Sent were never written.
    first_bar = next(i for i, line in enumerate(lines) if '"type":"Bar"' in line)
    path.write_text("".join(lines[: first_bar + 1]))
    core, leftover = replay(path)
    assert [type(r).__name__ for r in leftover] == [
        "DecisionNote",
        "OrderIntent",
        "Order",
        "Sent",
    ]
    assert core.orders[ID1].status is Status.SUBMITTED  # to be asked about


def test_a_journal_without_a_start_cannot_be_replayed():
    with pytest.raises(EngineError, match="start"):
        replay_day([(seen(30), bar(30))], lambda start: None)
