import asyncio
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from qfin_alphaguard.data.eodhd import MINUTE_SCHEMA
from qfin_alphaguard.events import Bar, Cancellation, OrderIntent, Side
from qfin_alphaguard.guard.check import GuardDecision
from qfin_alphaguard.guard.config import load_guard_config
from qfin_alphaguard.live.alpaca_broker import (
    Account,
    AlpacaOrder,
    BrokerError,
    OutcomeUnknown,
)
from qfin_alphaguard.live.engine import (
    CancelAll,
    Flatten,
    LiveEngine,
    SessionStart,
    Tick,
    replay_day,
)
from qfin_alphaguard.live.journal import Journal, JournalError
from qfin_alphaguard.live.orders import (
    Accepted,
    CancelRefused,
    Rejected,
    Sent,
    Status,
)
from qfin_alphaguard.live.runner import (
    STREAM_GRACE,
    Backup,
    Bars,
    Clock,
    DayFiles,
    Events,
    Fatal,
    Runner,
    RunnerError,
    StreamState,
    alert_to,
    check_session,
    feed_bars,
    feed_trades,
    keep_time,
    past_sessions,
    prepare,
)
from qfin_alphaguard.live.streams import StreamError
from qfin_alphaguard.live.warmup import sessions_before
from qfin_alphaguard.sessions import sessions
from qfin_alphaguard.strategy import Strategy

DAY = date(2026, 10, 19)
SESSION = sessions(DAY, DAY)[0]
OPEN, CLOSE = SESSION.open, SESSION.close
START = SessionStart(DAY, OPEN - timedelta(minutes=30), 30_000.0, 30_000.0)
QQQ_GUARD = load_guard_config(Path(__file__).parents[1] / "guard_qqq.yaml")
ID1 = "qqq-20261019-0001"
ID2 = "qqq-20261019-0002"


def bar(minute, close=600.0):
    start = OPEN + timedelta(minutes=minute)
    return Bar("QQQ", start, 60, close, close + 0.5, close - 0.5, close, 1000)


def seen(minute, delay=0.2):
    return OPEN + timedelta(minutes=minute + 1, seconds=delay)


class FakeClock:
    def __init__(self, at):
        self.at = at

    def now(self):
        return self.at

    async def sleep(self, seconds):
        self.at += timedelta(seconds=seconds)
        await asyncio.sleep(0)


class FakeAlpaca:
    """Alpaca's paper account as the runner sees it: orders, positions, the account."""

    def __init__(self, clock):
        self.clock = clock
        self.orders = {}  # cid -> [order, status, filled, average]
        self.held = {}
        self.foreign = []  # open orders placed by hand
        self.lose_answers = 0  # sends placed whose answers are lost
        self.lose_cancel_answers = 0  # cancels asked for whose answers are lost
        self.lose_sends = 0  # sends lost on their way
        self.refusal = None  # what Alpaca says when it refuses a send
        self.down = False
        self.fill_when_asked = False  # an open order asked about has filled
        self.cancel_when_asked = True  # a cancel asked about is confirmed

    def _check(self):
        if self.down:
            raise BrokerError("Alpaca read /v2/orders failed: connection refused")

    def snapshot(self, cid):
        order, status, filled, average = self.orders[cid]
        intent = order.intent
        return AlpacaOrder(
            "b-" + cid,
            cid,
            intent.ticker,
            intent.side,
            intent.quantity,
            status,
            filled,
            average,
            self.clock.now(),
        )

    def submit(self, order):
        self._check()
        cid = order.client_order_id
        if self.lose_sends:
            self.lose_sends -= 1
            raise OutcomeUnknown(f"send of {cid}: timed out")
        if self.refusal:
            return [Rejected(cid, self.clock.now(), self.refusal)]
        self.orders[cid] = [order, "accepted", 0.0, None]
        if self.lose_answers:
            self.lose_answers -= 1
            raise OutcomeUnknown(f"send of {cid}: connection reset")
        return self.snapshot(cid).events()

    def cancel(self, cid):
        self._check()
        if cid not in self.orders:
            raise BrokerError(f"cancel of {cid}: Alpaca has no such order")
        if self.orders[cid][1] == "filled":
            return [CancelRefused(cid, self.clock.now(), "order is not cancelable")]
        self.orders[cid][1] = "pending_cancel"
        if self.lose_cancel_answers:
            self.lose_cancel_answers -= 1
            raise OutcomeUnknown(f"cancel of {cid}: timed out")
        return []

    def fill(self, cid, total, price):
        """A fill at Alpaca: what the trade updates stream then sends."""
        order, _, filled, _ = self.orders[cid]
        sign = 1 if order.intent.side is Side.BUY else -1
        ticker = order.intent.ticker
        self.held[ticker] = round(
            self.held.get(ticker, 0.0) + sign * (total - filled), 9
        )
        if not self.held[ticker]:
            del self.held[ticker]
        done = total == order.intent.quantity
        self.orders[cid] = [
            order,
            "filled" if done else "partially_filled",
            total,
            price,
        ]
        return self.snapshot(cid).events()

    def order(self, cid):
        self._check()
        if cid not in self.orders:
            return None
        order, status, _, _ = self.orders[cid]
        if status == "pending_cancel" and self.cancel_when_asked:
            self.orders[cid][1] = "canceled"
        elif status in ("accepted", "new") and self.fill_when_asked:
            self.fill(cid, order.intent.quantity, order.intent.limit_price)
        elif (
            status in ("accepted", "new", "partially_filled")
            and self.clock.now() >= CLOSE
        ):
            self.orders[cid][1] = "expired"  # a day order, at the close
        return self.snapshot(cid)

    def open_orders(self):
        self._check()
        mine = [
            self.snapshot(cid)
            for cid, (_, status, _, _) in self.orders.items()
            if status in ("accepted", "new", "partially_filled", "pending_cancel")
        ]
        return mine + self.foreign

    def positions(self):
        self._check()
        return dict(self.held)

    def account(self):
        self._check()
        return Account(
            "ACTIVE", "USD", 30_000.0, 30_000.0, 30_000.0, 120_000.0, False, True
        )


class Scripted(Strategy):
    def __init__(self, script):
        self.script = script
        self.calls = []

    def on_bars(self, bars, view):
        self.calls.append((view.now, list(bars)))
        return list(self.script(bars, view))


class Approve:
    def check(self, intent, view):
        return GuardDecision(intent)


def once(*answers):
    calls = iter(answers)

    def script(bars, view):
        answer = next(calls, ())
        return answer(view) if callable(answer) else answer

    return script


def buy(quantity, view):
    return OrderIntent("QQQ", Side.BUY, quantity, 603.0, view.now, "decision")


def runner(tmp_path, script=lambda b, v: (), start=START, alpaca=None, clock=None):
    clock = clock or FakeClock(OPEN)
    alpaca = alpaca or FakeAlpaca(clock)
    strategy = Scripted(script)
    core = LiveEngine(strategy, Approve(), start, collar_bps=50.0)
    journal = Journal(tmp_path / "journal.jsonl")
    journal.write(start, at=start.time)
    alerts = []
    run = Runner(
        core,
        start,
        journal,
        alpaca,
        alerts.append,
        now=clock.now,
        sleep=clock.sleep,
        blocking_sleep=lambda s: None,
    )
    return run, alpaca, alerts, clock, strategy


class TimedQueue:
    """The runner's queue, moving the fake clock to each input's time as it is taken."""

    def __init__(self, clock):
        self.clock = clock
        self.queue = asyncio.Queue()

    def put_nowait(self, item):
        self.queue.put_nowait(item)

    async def get(self):
        item = await self.queue.get()
        self.clock.at = max(self.clock.at, item.at)
        return item


def play(run, clock, *items):
    async def go():
        queue = TimedQueue(clock)
        for item in items:
            queue.put_nowait(item)
        return await run.run(queue)

    return asyncio.run(go())


CLOSED = Clock(CLOSE + timedelta(minutes=2), closed=True)


def written(run):
    """The journal's entries, once the runner has let it go."""
    run.journal.close()
    with Journal(run.journal.path) as journal:
        return journal.entries


def replayed(run, script):
    run.journal.close()
    with Journal(run.journal.path) as journal:
        core, leftover = replay_day(
            journal.entries,
            lambda s: LiveEngine(Scripted(script), Approve(), s, collar_bps=50.0),
        )
    return core, leftover


async def until(done, what, timeout=5.0):
    """Waits until done() holds, looking every 5 ms; fails after timeout seconds.

    The runner calls Alpaca in a worker thread, so on a loaded machine a fixed sleep
    can end before the call does.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not done():
        if loop.time() > deadline:
            pytest.fail(f"waited {timeout} s for {what}")
        await asyncio.sleep(0.005)


def accepted(run, cid):
    """The core has heard Alpaca accept cid: the send and its answer are done."""
    order = run.core.orders.get(cid)
    return order is not None and order.status is Status.ACCEPTED


# A day.


def test_a_day_goes_through_the_journal_to_alpaca_and_back(tmp_path):
    run, alpaca, alerts, clock, _ = runner(tmp_path, once(lambda v: [buy(100, v)]))

    async def day():
        queue = TimedQueue(clock)
        task = asyncio.create_task(run.run(queue))
        queue.put_nowait(StreamState("trades", True, seen(29)))
        queue.put_nowait(Bars((bar(30),), seen(30)))
        await until(lambda: accepted(run, ID1), f"{ID1} to be accepted")
        assert alpaca.orders[ID1][1] == "accepted"  # sent, and accepted
        queue.put_nowait(Events(tuple(alpaca.fill(ID1, 40.0, 600.0)), seen(30, 0.5)))
        queue.put_nowait(Events(tuple(alpaca.fill(ID1, 100.0, 600.02)), seen(30, 0.7)))
        queue.put_nowait(Bars((bar(31),), seen(31)))
        queue.put_nowait(CLOSED)
        return await task

    assert asyncio.run(day()) == "closed"
    assert run.core.orders[ID1].status is Status.FILLED
    assert run.core.portfolio.positions["QQQ"] == 100.0
    # The script never sold: Alpaca still holds it at the close.
    assert alerts == ["held overnight, to be closed at the next open: {'QQQ': 100.0}"]
    core, leftover = replayed(run, once(lambda v: [buy(100, v)]))
    assert leftover == [] and core.orders == run.core.orders


def test_each_input_is_written_before_the_core_sees_it(tmp_path):
    run, _, alerts, clock, _ = runner(tmp_path, once(lambda v: [buy(1, v)]))
    assert play(run, clock, Bars((bar(30),), seen(30)), CLOSED) == "closed"
    entries = written(run)
    kinds = [type(e.record).__name__ for e in entries]
    times = {type(e.record).__name__: e.at for e in entries}
    assert kinds == [
        "SessionStart",
        "Bar",
        "OrderIntent",
        "Order",
        "Sent",
        "Accepted",  # Alpaca's answer to the send
        "Expired",  # asked about at the close
    ]
    assert times["Bar"] == times["Order"] == seen(30)
    assert alerts == []


def test_the_day_s_last_minute_lets_the_strategy_act_without_its_bar(tmp_path):
    run, _, _, clock, strategy = runner(tmp_path)
    at = CLOSE - timedelta(seconds=55)
    play(run, clock, Bars((bar(30),), seen(30)), Clock(at, last_minute=True), CLOSED)
    assert strategy.calls[-1] == (at, [])
    assert Tick(at) in [e.record for e in written(run)]


# A send whose answer is lost, refused, or never arrives.


def test_a_send_whose_answer_was_lost_is_found_by_reconciliation(tmp_path):
    run, alpaca, alerts, clock, _ = runner(tmp_path, once(lambda v: [buy(100, v)]))
    alpaca.lose_answers = 1
    assert play(run, clock, Bars((bar(30),), seen(30)), CLOSED) == "closed"
    assert Accepted(ID1, seen(30)) in [e.record for e in written(run)]  # reconciliation
    assert run.core.orders[ID1].status is Status.EXPIRED and alerts == []


def test_a_send_answered_late_is_caught_up_with_its_fill(tmp_path):
    run, alpaca, alerts, clock, _ = runner(tmp_path, once(lambda v: [buy(100, v)]))
    alpaca.lose_answers = 1
    alpaca.fill_when_asked = True  # by the time it is asked about, it has filled
    assert play(run, clock, Bars((bar(30),), seen(30)), CLOSED) == "closed"
    assert run.core.orders[ID1].status is Status.FILLED
    assert run.core.portfolio.positions["QQQ"] == 100.0
    assert alerts == ["held overnight, to be closed at the next open: {'QQQ': 100.0}"]


def test_a_send_lost_on_its_way_was_not_placed(tmp_path):
    run, alpaca, alerts, clock, _ = runner(tmp_path, once(lambda v: [buy(100, v)]))
    alpaca.lose_sends = 1
    assert play(run, clock, Bars((bar(30),), seen(30)), CLOSED) == "closed"
    assert run.core.orders[ID1].status is Status.REJECTED
    assert alerts == [] and alpaca.orders == {}


def test_a_send_alpaca_refuses_ends_the_order_and_the_day_goes_on(tmp_path):
    run, alpaca, alerts, clock, _ = runner(tmp_path, once(lambda v: [buy(100, v)]))
    alpaca.refusal = "insufficient buying power"
    assert play(run, clock, Bars((bar(30),), seen(30)), CLOSED) == "closed"
    assert run.core.orders[ID1].status is Status.REJECTED and alerts == []


def test_a_cancel_whose_answer_was_lost_is_settled_before_the_next_input(tmp_path):
    script = once(lambda v: [buy(100, v)], lambda v: [Cancellation(ID1, v.now)])
    run, alpaca, alerts, clock, _ = runner(tmp_path, script)
    alpaca.lose_cancel_answers = 1
    result = play(
        run,
        clock,
        Bars((bar(30),), seen(30)),
        Bars((bar(31),), seen(31)),
        Bars((bar(32),), seen(32)),
        CLOSED,
    )
    assert result == "closed" and alerts == []
    kinds = [type(e.record).__name__ for e in written(run)]
    bars = [i for i, k in enumerate(kinds) if k == "Bar"]
    assert bars[1] < kinds.index("Cancelled") < bars[2]


def test_a_position_carried_from_yesterday_agrees_and_the_day_goes_on(tmp_path):
    start = SessionStart(
        DAY, START.time, 24_000.0, 30_000.0, {"QQQ": 10.0}, {"QQQ": 600.0}
    )
    run, alpaca, alerts, clock, _ = runner(tmp_path, start=start)
    alpaca.held = {"QQQ": 10.0}
    assert play(run, clock, Bars((bar(30),), seen(30)), CLOSED) == "closed"
    assert alerts == ["held overnight, to be closed at the next open: {'QQQ': 10.0}"]


def test_a_difference_at_the_close_stops_the_engine(tmp_path):
    run, alpaca, alerts, clock, _ = runner(tmp_path, once(lambda v: [buy(100, v)]))

    async def day():
        queue = TimedQueue(clock)
        task = asyncio.create_task(run.run(queue))
        queue.put_nowait(Bars((bar(30),), seen(30)))
        await until(lambda: accepted(run, ID1), f"{ID1} to be accepted")
        queue.put_nowait(Events(tuple(alpaca.fill(ID1, 100.0, 600.0)), seen(30, 1)))
        await until(
            lambda: run.core.portfolio.positions.get("QQQ") == 100.0,
            "the engine to hold 100 QQQ",
        )
        alpaca.held["QQQ"] = 50.0  # half sold by hand, unheard
        queue.put_nowait(CLOSED)
        return await task

    assert asyncio.run(day()) == "stopped"
    assert "QQQ: 100.0 shares here, 50.0 at Alpaca" in alerts[0]


def test_an_order_alpaca_has_lost_at_the_close_stops_the_engine(tmp_path):
    run, alpaca, alerts, clock, _ = runner(tmp_path, once(lambda v: [buy(100, v)]))

    async def day():
        queue = TimedQueue(clock)
        task = asyncio.create_task(run.run(queue))
        queue.put_nowait(Bars((bar(30),), seen(30)))
        await until(lambda: accepted(run, ID1), f"{ID1} to be accepted")
        del alpaca.orders[ID1]
        queue.put_nowait(CLOSED)
        return await task

    assert asyncio.run(day()) == "stopped"
    assert "is open here and unknown to Alpaca" in alerts[0]


# What stops the engine, and the close that follows.


def test_a_position_is_closed_by_alpaca_s_numbers_after_a_stop(tmp_path):
    run, alpaca, alerts, clock, _ = runner(tmp_path, once(lambda v: [buy(100, v)]))
    alpaca.fill_when_asked = True

    async def day():
        queue = TimedQueue(clock)
        task = asyncio.create_task(run.run(queue))
        queue.put_nowait(Bars((bar(30, close=602.0),), seen(30)))
        await until(lambda: accepted(run, ID1), f"{ID1} to be accepted")
        queue.put_nowait(Events(tuple(alpaca.fill(ID1, 60.0, 600.0)), seen(30, 1)))
        alpaca.held["QQQ"] = 75.0  # Alpaca holds more than the engine heard of
        queue.put_nowait(Fatal("bars: the plan does not include the feed", seen(31)))
        return await task

    assert asyncio.run(day()) == "stopped"
    assert (
        alerts[0] == "stopping: RunnerError: bars: the plan does not include the feed"
    )
    close = run.core.orders[ID2].order.intent
    assert (close.side, close.quantity, close.reason) == (
        Side.SELL,
        75.0,
        "close after a stop",
    )
    assert close.limit_price == 598.99  # 602 less the 50 bps collar, on the cent above
    assert run.core.orders[ID1].status is Status.CANCELLED
    assert alerts[-1] == "nothing is held; the engine has stopped"
    core, leftover = replayed(run, once(lambda v: [buy(100, v)]))
    assert leftover == [] and core.orders == run.core.orders and core.stopped


def test_a_close_that_does_not_fill_is_tried_again_then_alerted(tmp_path):
    start = SessionStart(
        DAY, START.time, 24_000.0, 30_000.0, {"QQQ": 10.0}, {"QQQ": 600.0}
    )
    run, alpaca, alerts, clock, _ = runner(tmp_path, start=start)
    alpaca.held = {"QQQ": 10.0}  # yesterday's close did not fill
    assert (
        play(run, clock, Bars((bar(30),), seen(30)), Fatal("lost", seen(31)))
        == "stopped"
    )
    closes = [
        o
        for o in run.core.orders.values()
        if o.order.intent.reason == "close after a stop"
    ]
    assert len(closes) == 3  # one a round, each cancelled when it did not fill
    assert all(o.status is Status.CANCELLED for o in closes)
    assert alerts[-1] == "still held after 3 rounds of closing: {'QQQ': 10.0}"
    records = [e.record for e in written(run)]
    assert sum(isinstance(r, CancelAll) for r in records) == 3
    assert sum(isinstance(r, Flatten) for r in records) == 3


def test_open_orders_the_engine_did_not_make_are_left_to_a_person(tmp_path):
    run, alpaca, alerts, clock, _ = runner(tmp_path)
    alpaca.held = {"QQQ": 5.0}
    alpaca.foreign = [
        AlpacaOrder("b-x", "by-hand", "QQQ", Side.SELL, 5.0, "new", 0.0, None, OPEN)
    ]
    assert play(run, clock, CLOSED) == "stopped"
    assert "by-hand is open at Alpaca, an order the engine did not make" in alerts[0]
    assert alerts[-1] == "open orders the engine did not make, ['by-hand']: not closing"
    assert run.core.orders == {}


def test_a_difference_at_the_start_stops_and_closes(tmp_path):
    start = SessionStart(DAY, START.time, 30_000.0, 30_000.0, {}, {"QQQ": 600.0})
    run, alpaca, alerts, clock, strategy = runner(tmp_path, start=start)
    alpaca.held = {"QQQ": 3.0}  # the engine started flat
    alpaca.fill_when_asked = True
    assert play(run, clock, Bars((bar(30),), seen(30)), CLOSED) == "stopped"
    assert "QQQ: 0.0 shares here, 3.0 at Alpaca" in alerts[0]
    assert alerts[-1] == "nothing is held; the engine has stopped"
    assert run.core.orders[ID1].order.intent.side is Side.SELL
    assert strategy.calls == []  # stopped before the first bar


def test_an_alpaca_that_cannot_be_reached_is_left_to_a_person(tmp_path):
    run, alpaca, alerts, clock, _ = runner(tmp_path)
    alpaca.down = True
    assert play(run, clock, CLOSED) == "stopped"
    assert alerts[0].startswith("stopping: BrokerError")
    assert alerts[-1].startswith("the stop itself failed, BrokerError")


def test_an_update_the_order_rules_refuse_stops_the_engine(tmp_path):
    run, _, alerts, clock, _ = runner(tmp_path, once(lambda v: [buy(100, v)]))
    result = play(
        run,
        clock,
        Bars((bar(30),), seen(30)),
        Events((Sent(ID1, seen(30, 1)),), seen(30, 1)),
    )
    assert result == "stopped" and alerts[0].startswith("stopping: InvalidTransition")
    assert run.core.orders[ID1].status is Status.CANCELLED
    assert alerts[-1] == "nothing is held; the engine has stopped"


def test_an_update_for_an_order_the_engine_did_not_make_stops_it(tmp_path):
    run, _, alerts, clock, _ = runner(tmp_path)
    assert (
        play(run, clock, Events((Accepted("by-hand", seen(30)),), seen(30)))
        == "stopped"
    )
    assert "did not make" in alerts[0]


# The streams.


def test_trade_updates_that_reconnect_are_caught_up(tmp_path):
    run, alpaca, alerts, clock, _ = runner(tmp_path, once(lambda v: [buy(100, v)]))

    async def day():
        queue = TimedQueue(clock)
        task = asyncio.create_task(run.run(queue))
        queue.put_nowait(Bars((bar(30),), seen(30)))
        queue.put_nowait(StreamState("trades", False, seen(30, 1)))
        await until(
            lambda: accepted(run, ID1) and "trades" in run.down_since,
            f"{ID1} to be accepted and the trades stream to be down",
        )
        alpaca.fill(ID1, 100.0, 600.1)  # filled while the stream was down: not heard
        queue.put_nowait(StreamState("trades", True, seen(30, 5)))
        queue.put_nowait(Bars((bar(31),), seen(31)))
        queue.put_nowait(CLOSED)
        return await task

    assert asyncio.run(day()) == "closed"
    kinds = [type(e.record).__name__ for e in written(run)]
    assert kinds.index("FillReport") < kinds.index("Bar", kinds.index("Accepted"))
    assert run.core.orders[ID1].status is Status.FILLED
    assert run.core.portfolio.positions["QQQ"] == 100.0
    assert len(alerts) == 1 and alerts[0].startswith("held overnight")


def test_a_stream_down_too_long_while_the_market_is_open_stops_the_engine(tmp_path):
    run, _, alerts, clock, _ = runner(tmp_path)
    down = OPEN + timedelta(minutes=10)
    result = play(
        run,
        clock,
        StreamState("bars", False, down),
        Clock(down + STREAM_GRACE - timedelta(seconds=1)),
        Clock(down + STREAM_GRACE),
    )
    assert result == "stopped" and "bars stream has been down" in alerts[0]


def test_a_stream_down_before_the_open_is_given_its_grace_from_the_open(tmp_path):
    run, _, alerts, clock, _ = runner(
        tmp_path, clock=FakeClock(OPEN - timedelta(hours=1))
    )
    result = play(
        run,
        clock,
        StreamState("bars", False, OPEN - timedelta(minutes=20)),
        Clock(OPEN + STREAM_GRACE - timedelta(seconds=1)),
        StreamState("bars", True, OPEN + STREAM_GRACE),
        Clock(OPEN + 2 * STREAM_GRACE),
        CLOSED,
    )
    assert result == "closed" and alerts == []


# The morning.


class FakeData:
    def __init__(self):
        self.asked = 0

    def minute_bars(self, ticker, first, last, feed="sip"):
        self.asked += 1
        rows = []
        for i, s in enumerate(sessions_before(DAY)):
            level = 600.0 + 3.0 * (i % 4) - 1.5 * (i % 3)  # closes that move
            for minute in range(0, 390, 30):
                start = s.open + timedelta(minutes=minute)
                rows.append(
                    (start, "QQQ", level, level + 1, level - 1, level + 0.5, 1000)
                )
        return pl.DataFrame(rows, schema=MINUTE_SCHEMA, orient="row")


def test_the_morning_starts_from_alpaca_s_account(tmp_path):
    files = DayFiles(tmp_path, DAY)
    clock = FakeClock(OPEN - timedelta(minutes=30))
    alpaca = FakeAlpaca(clock)
    alpaca.held = {"QQQ": 2.0}
    data = FakeData()
    core, start, journal = prepare(DAY, files, QQQ_GUARD, alpaca, data, now=clock.now)
    journal.close()
    assert (start.cash, start.last_close_value) == (30_000.0, 30_000.0)
    assert dict(start.positions) == {"QQQ": 2.0}
    assert dict(start.prices) == {"QQQ": 608.0}  # the last session's last close
    assert len(core.strategy.closes) == core.strategy.closes.maxlen  # warmed
    assert files.warm_up.exists() and data.asked == 1
    with Journal(files.journal) as again:
        assert again.records == [start]


def test_a_restarted_morning_replays_its_journal_on_the_same_history(tmp_path):
    files = DayFiles(tmp_path, DAY)
    clock = FakeClock(OPEN - timedelta(minutes=30))
    alpaca, data = FakeAlpaca(clock), FakeData()
    core, start, journal = prepare(DAY, files, QQQ_GUARD, alpaca, data, now=clock.now)
    journal.write(bar(0, close=620.0), at=seen(0))
    core.on_bars([bar(0, close=620.0)], seen(0))
    journal.write(bar(29, close=640.0), at=seen(29))  # 10:00: the first decision
    step = core.on_bars([bar(29, close=640.0)], seen(29))
    journal.close()  # a crash: the bar written, the decision not
    assert type(step.records[0]).__name__ == "DecisionNote"
    again, start2, journal = prepare(DAY, files, QQQ_GUARD, alpaca, data, now=clock.now)
    journal.close()
    assert start2 == start and data.asked == 1  # the kept history, not a new fetch
    assert again.orders == core.orders
    with Journal(files.journal) as kept:
        assert kept.records[3:] == list(step.records)  # the leftovers, written


def test_only_one_engine_may_run_a_day(tmp_path):
    files = DayFiles(tmp_path, DAY)
    clock = FakeClock(OPEN)
    _, _, journal = prepare(
        DAY, files, QQQ_GUARD, FakeAlpaca(clock), FakeData(), now=clock.now
    )
    with pytest.raises(JournalError, match="another engine"):
        prepare(DAY, files, QQQ_GUARD, FakeAlpaca(clock), FakeData(), now=clock.now)
    journal.close()


def test_the_runner_trades_only_the_paper_account(tmp_path):
    live = replace(QQQ_GUARD, live_trading=True)
    clock = FakeClock(OPEN)
    with pytest.raises(RunnerError, match="paper"):
        prepare(DAY, DayFiles(tmp_path, DAY), live, FakeAlpaca(clock), FakeData())


def test_an_account_that_cannot_trade_does_not_start(tmp_path):
    clock = FakeClock(OPEN)
    alpaca = FakeAlpaca(clock)
    alpaca.account = lambda: Account("ACTIVE", "USD", 1.0, 1.0, 1.0, 1.0, True, True)
    with pytest.raises(RunnerError, match="cannot trade"):
        prepare(DAY, DayFiles(tmp_path, DAY), QQQ_GUARD, alpaca, FakeData())
    # The journal is free again for the next try.
    Journal(DayFiles(tmp_path, DAY).journal).close()


def test_history_is_kept_beside_the_journal(tmp_path):
    files = DayFiles(tmp_path, DAY)
    data = FakeData()
    first = past_sessions(data, DAY, files.warm_up)
    second = past_sessions(data, DAY, files.warm_up)
    assert first == second and data.asked == 1
    assert [s.day for s, _ in first] == [s.day for s in sessions_before(DAY)]


# The feeds.


def test_the_bars_feed_reconnects_and_says_so():
    clock = FakeClock(OPEN)
    tries = []

    async def stream(key, secret, symbols, feed, on_connected):
        tries.append(feed)
        if len(tries) == 1:
            raise OSError("network unreachable")
        on_connected()
        yield bar(0)
        if len(tries) == 3:
            raise asyncio.CancelledError  # the session is over

    queue = asyncio.Queue()
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(feed_bars(queue, ("k", "s"), clock.now, clock.sleep, stream))
    got = []
    while not queue.empty():
        got.append(queue.get_nowait())
    assert [type(g).__name__ for g in got] == [
        "StreamState",  # down
        "StreamState",  # up
        "Bars",
        "StreamState",  # the stream ended
        "StreamState",  # up again
        "Bars",
    ]
    assert [g.up for g in got if isinstance(g, StreamState)] == [
        False,
        True,
        False,
        True,
    ]
    assert tries == ["iex"] * 3


def test_a_trade_update_that_cannot_be_read_is_fatal():
    clock = FakeClock(OPEN)

    async def stream(key, secret, on_connected):
        on_connected()
        raise BrokerError("a trade update the engine does not expect: 'replaced'")
        yield  # an async generator

    queue = asyncio.Queue()
    asyncio.run(feed_trades(queue, ("k", "s"), clock.now, clock.sleep, stream))
    up, fatal = queue.get_nowait(), queue.get_nowait()
    assert up == StreamState("trades", True, OPEN)
    assert isinstance(fatal, Fatal) and "replaced" in fatal.reason


def test_a_refused_trade_stream_is_tried_again():
    clock = FakeClock(OPEN)
    tries = []

    async def stream(key, secret, on_connected):
        tries.append(1)
        if len(tries) < 3:
            raise StreamError("trade updates refused the keys")
        raise asyncio.CancelledError
        yield

    queue = asyncio.Queue()
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(feed_trades(queue, ("k", "s"), clock.now, clock.sleep, stream))
    assert len(tries) == 3


def test_the_clock_ticks_the_last_minute_once_then_closes():
    clock = FakeClock(CLOSE - timedelta(minutes=2))
    queue = asyncio.Queue()
    asyncio.run(keep_time(queue, SESSION, clock.now, clock.sleep))
    got = []
    while not queue.empty():
        got.append(queue.get_nowait())
    ticks = [g for g in got if g.last_minute]
    assert len(ticks) == 1 and ticks[0].at == CLOSE - timedelta(seconds=55)
    assert got[-1].closed
    assert (
        CLOSE + timedelta(minutes=2)
        <= got[-1].at
        < CLOSE + timedelta(minutes=2, seconds=6)
    )
    assert sum(g.closed for g in got) == 1


# On the server: the day's folder in S3, alerts e-mailed, and the check.


class FakeS3:
    def __init__(self):
        self.objects = {}  # (bucket, key) -> bytes
        self.fail_uploads = False

    def download_file(self, bucket, key, filename):
        from botocore.exceptions import ClientError

        if (bucket, key) not in self.objects:
            raise ClientError(
                {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"
            )
        Path(filename).write_bytes(self.objects[(bucket, key)])

    def upload_file(self, filename, bucket, key):
        from botocore.exceptions import EndpointConnectionError

        if self.fail_uploads:
            raise EndpointConnectionError(
                endpoint_url="https://s3.eu-north-1.amazonaws.com"
            )
        self.objects[(bucket, key)] = Path(filename).read_bytes()


def test_the_day_s_folder_goes_to_s3_and_comes_back(tmp_path):
    s3 = FakeS3()
    files = DayFiles(tmp_path / "server-1", DAY)
    files.folder.mkdir(parents=True)
    files.journal.write_text('{"seq":1}\n')
    files.warm_up.write_bytes(b"parquet")
    Backup(files, "bucket", s3).push()
    assert set(s3.objects) == {
        ("bucket", "data/live/2026-10-19/journal.jsonl"),
        ("bucket", "data/live/2026-10-19/warm_up.parquet"),
    }
    # A new task, on a new disk, finds the day where the first left it.
    again = DayFiles(tmp_path / "server-2", DAY)
    assert Backup(again, "bucket", s3).pull() == ["journal.jsonl", "warm_up.parquet"]
    assert again.journal.read_text() == '{"seq":1}\n'


def test_a_pull_never_overwrites_what_the_disk_has(tmp_path):
    s3 = FakeS3()
    s3.objects[("bucket", "data/live/2026-10-19/journal.jsonl")] = b"older\n"
    files = DayFiles(tmp_path, DAY)
    files.folder.mkdir(parents=True)
    files.journal.write_text("newer\n")
    assert Backup(files, "bucket", s3).pull() == []
    assert files.journal.read_text() == "newer\n"


def test_a_failed_backup_does_not_stop_the_day(tmp_path, caplog):
    s3 = FakeS3()
    s3.fail_uploads = True
    files = DayFiles(tmp_path, DAY)
    files.folder.mkdir(parents=True)
    files.journal.write_text("x\n")
    Backup(files, "bucket", s3).push()  # does not raise
    assert "backup of journal.jsonl failed" in caplog.text


class FakeSNS:
    def __init__(self, fail=False):
        self.sent = []
        self.fail = fail

    def publish(self, **message):
        from botocore.exceptions import ClientError

        if self.fail:
            raise ClientError({"Error": {"Code": "AuthorizationError"}}, "Publish")
        self.sent.append(message)


def test_an_alert_is_written_and_e_mailed(tmp_path):
    sns = FakeSNS()
    files = DayFiles(tmp_path, DAY)
    alert = alert_to(files.alert, "arn:aws:sns:eu-north-1:1:qfin-alerts", sns)
    alert("stopping: lost data")
    assert files.alert.read_text().endswith(" stopping: lost data\n")
    assert sns.sent == [
        {
            "TopicArn": "arn:aws:sns:eu-north-1:1:qfin-alerts",
            "Subject": "qfin live 2026-10-19: alert",
            "Message": "stopping: lost data",
        }
    ]


def test_an_alert_that_cannot_be_e_mailed_is_still_written(tmp_path, caplog):
    files = DayFiles(tmp_path, DAY)
    alert_to(files.alert, "arn:aws:sns:eu-north-1:1:qfin-alerts", FakeSNS(fail=True))(
        "x"
    )
    assert files.alert.exists() and "could not be e-mailed" in caplog.text


def connecting(*names_that_fail):
    """Streams that connect at once and then wait, or never connect."""

    def bars(key, secret, symbols, feed, on_connected):
        async def stream():
            if "bars" not in names_that_fail:
                on_connected()
            await asyncio.sleep(3600)
            yield

        return stream()

    def trades(key, secret, on_connected):
        async def stream():
            if "trades" not in names_that_fail:
                on_connected()
            await asyncio.sleep(3600)
            yield

        return stream()

    return bars, trades


def check(tmp_path, alpaca, *failing, seconds=0.5):
    bars, trades = connecting(*failing)
    return asyncio.run(
        check_session(
            DAY,
            tmp_path,
            QQQ_GUARD,
            ("k", "s"),
            seconds=seconds,
            broker=alpaca,
            data=FakeData(),
            bars=bars,
            trades=trades,
            now=alpaca.clock.now,
        )
    )


def test_the_check_reads_reconciles_and_connects_without_trading(tmp_path):
    alpaca = FakeAlpaca(FakeClock(OPEN - timedelta(hours=1)))
    said = check(tmp_path, alpaca)
    assert said.startswith("checked: account 30,000.00 USD, held nothing")
    assert "15 closes" in said and "both streams connected" in said
    assert alpaca.orders == {}
    assert (tmp_path / "check" / DAY.isoformat() / "journal.jsonl").exists()
    assert not DayFiles(tmp_path, DAY).journal.exists()  # the day's own is untouched


def test_the_check_fails_on_a_difference_or_a_stream_that_does_not_connect(tmp_path):
    alpaca = FakeAlpaca(FakeClock(OPEN))
    answers = iter([{"QQQ": 1.0}])
    # 1 share in the morning, 2 when reconciled: someone traded in between.
    alpaca.positions = lambda: next(answers, {"QQQ": 2.0})
    with pytest.raises(RunnerError, match="reconciliation"):
        check(tmp_path / "a", alpaca)
    with pytest.raises(RunnerError, match="trades"):
        check(tmp_path / "b", FakeAlpaca(FakeClock(OPEN)), "trades")
