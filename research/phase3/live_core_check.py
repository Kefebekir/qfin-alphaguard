"""Phase 3: does the live engine's core decide as the backtest's engine does?

The QQQ strategy (momentum.py) behind Guard (guard_qqq.yaml), on IEX's stored minutes of
QQQ (the live feed, decision 0021), twice:
- the backtest's engine, with its simulated broker filling at the next bar;
- the live engine's core (live/engine.py), one LiveEngine a session. Each bar reaches it
  0.2 s after it ended, as Alpaca's stream sends them. Alpaca's answers are played as they
  would come: each order accepted at once, then its fills as totals, at the times and
  prices the backtest filled the same order; a cancel confirmed at once; what is still
  open at the close expired.

No fees and no spread in either. Each session the core starts from the account as the
backtest left it, as the live engine starts from Alpaca's account each morning: the core's
books, built from fill totals, can differ from the backtest's by floating-point crumbs,
and a crumb in the cash can move a fractional order by a billionth of a share.
The end-of-day books are compared too. PARTICIPATION is the share of a bar's volume an order may take (0.1, the
backtest's own, by default); a small one leaves orders open for minutes, so the strategy
cancels and replaces them, and the live core must wait for each cancel. The checks:
1. every decision note (choice, reason, the numbers seen), every order (side, shares,
   limit) and every Guard event is the same, in the same order;
2. each session's journal, written as JSON and read back, replayed through a fresh core
   (engine.replay_day), gives the same session: the same orders and the same books.

Usage: live_core_check.py ROOT FIRST LAST GUARD_QQQ_YAML [PARTICIPATION]
"""

import copy
import json
import math
import sys
import time
from collections import defaultdict, deque
from datetime import date, timedelta
from itertools import groupby
from pathlib import Path

import polars as pl

from qfin_alphaguard.backtest.broker import CostModel, NextBarBroker
from qfin_alphaguard.backtest.engine import SessionClose, run_backtest
from qfin_alphaguard.events import (
    Bar,
    DecisionNote,
    Fill,
    Order,
    RiskEvent,
    round_shares,
)
from qfin_alphaguard.guard.config import load_guard_config
from qfin_alphaguard.guard.rules import Guard
from qfin_alphaguard.live.engine import (
    Cancel,
    LiveEngine,
    SessionStart,
    Submit,
    replay_day,
)
from qfin_alphaguard.live.journal import decode, encode
from qfin_alphaguard.live.orders import Accepted, Cancelled, Expired, FillReport
from qfin_alphaguard.momentum import NoiseAreaMomentum
from qfin_alphaguard.sessions import sessions

ROOT = Path(sys.argv[1])
FIRST, LAST = date.fromisoformat(sys.argv[2]), date.fromisoformat(sys.argv[3])
LIMITS = load_guard_config(sys.argv[4])
PARTICIPATION = float(sys.argv[5]) if len(sys.argv) > 5 else 0.1
CASH = 30_000.0
DELAY = timedelta(milliseconds=200)  # a bar reaches the engine this long after it ends
FREE = CostModel(sec_fee_rate=0, cat_per_share=0, half_spread_bps=0)

calendar = {s.day: s for s in sessions(FIRST, LAST)}
minutes = (
    pl.concat(
        [
            pl.read_parquet(ROOT / "intraday_iex" / "QQQ" / f"{year}.parquet")
            for year in range(FIRST.year, LAST.year + 1)
        ]
    )
    .filter(pl.col("start").dt.date().is_between(FIRST, LAST))
    .sort("start")
)
bars = [
    Bar("QQQ", r["start"], 60, r["open"], r["high"], r["low"], r["close"], r["volume"])
    for r in minutes.iter_rows(named=True)
]
bars = [
    b
    for b in bars
    if b.start.date() in calendar
    and calendar[b.start.date()].open <= b.start
    and b.end <= calendar[b.start.date()].close
]
print(
    f"{len(bars):,} IEX minutes of QQQ, {FIRST} to {LAST}; an order takes at most "
    f"{PARTICIPATION:.1%} of a bar's volume"
)

# 1. The backtest.


class TellingBroker(NextBarBroker):
    """The backtest's broker, noting which bar each fill came from.

    A fill's time does not tell: one at a bar's open is dated its start, the end of the
    bar before.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.bar_ends: dict[str, list] = defaultdict(list)

    def on_bar(self, bar: Bar) -> list[Fill]:
        fills = super().on_bar(bar)
        for fill in fills:
            self.bar_ends[fill.client_order_id].append(bar.end)
        return fills


clock = time.monotonic()
broker = TellingBroker(FREE, participation=PARTICIPATION)
bt = run_backtest(bars, NoiseAreaMomentum(), broker, Guard(LIMITS), CASH)
print(f"backtest: {time.monotonic() - clock:.0f} s")
bt_order_ids = [r.client_order_id for r in bt.log if isinstance(r, Order)]
bt_fills: dict[str, list[tuple]] = defaultdict(list)  # (the bar's end, the fill)
bt_accounts = {}  # each session's close: cash, value and positions, as the backtest had them
bt_held: dict[str, float] = {}
for r in bt.log:
    if isinstance(r, Fill):
        bar_end = broker.bar_ends[r.client_order_id][len(bt_fills[r.client_order_id])]
        bt_fills[r.client_order_id].append((bar_end, r))
        shares = r.quantity if r.side.value == "buy" else -r.quantity
        bt_held[r.ticker] = round_shares(bt_held.get(r.ticker, 0.0) + shares)
    elif isinstance(r, SessionClose):
        bt_accounts[r.day] = (r.cash, r.value, {t: s for t, s in bt_held.items() if s})

# 2. The live core, a session at a time.


class Alpaca:
    """Alpaca's answers to one session's core, played from the backtest's fills."""

    sent = 0  # orders sent over all sessions: the n-th is the backtest's n-th
    cancels = 0
    partial_reports = 0

    def __init__(self, core: LiveEngine, start: SessionStart) -> None:
        self.core = core
        self.entries: list = [(start.time, start)]  # the session's journal
        self.due: dict[
            str, deque
        ] = {}  # each live order's fills, as the backtest made them
        self.totals: dict[
            str, tuple[float, float]
        ] = {}  # shares and their cost, so far

    def run(self, step, at) -> None:
        for record in step.records:
            self.entries.append((at, record))
            outputs.append(record)
        for action in step.actions:
            if isinstance(action, Submit):
                cid = action.order.client_order_id
                self.due[cid] = deque(bt_fills[bt_order_ids[Alpaca.sent]])
                Alpaca.sent += 1
                self.deliver(Accepted(cid, at), at)
            elif isinstance(action, Cancel):
                Alpaca.cancels += 1
                self.due.pop(action.client_order_id, None)
                self.deliver(Cancelled(action.client_order_id, at), at)

    def deliver(self, event, at) -> None:
        self.entries.append((at, event))
        self.run(self.core.on_order_event(event, at), at)

    def bar(self, bar: Bar) -> None:
        """The fills made during the bar's minute, then the bar itself."""
        fill_at = bar.end + DELAY / 2
        for cid, fills in list(self.due.items()):
            while fills and fills[0][0] <= bar.end:
                _, fill = fills.popleft()
                shares, cost = self.totals.get(cid, (0.0, 0.0))
                shares = round_shares(shares + fill.quantity)
                cost += fill.quantity * fill.price
                self.totals[cid] = (shares, cost)
                Alpaca.partial_reports += shares < self.core.orders[cid].quantity
                self.deliver(FillReport(cid, fill_at, shares, cost / shares), fill_at)
        at = bar.end + DELAY
        self.entries.append((at, bar))
        self.run(self.core.on_bars([bar], at), at)

    def close(self, at) -> None:
        """What is still open at the close expires."""
        for cid, order in list(self.core.orders.items()):
            if order.is_open:
                self.deliver(Expired(cid, at), at)


def replayed(entries, before):
    """The session's journal, as JSON and back, through a fresh core."""
    written = [(at, decode(json.loads(json.dumps(encode(r))))) for at, r in entries]

    def make(start):
        return LiveEngine(
            copy.deepcopy(before), Guard(LIMITS), start, LIMITS.price_collar_bps
        )

    return replay_day(written, make)


clock = time.monotonic()
strategy = NoiseAreaMomentum()
guard = Guard(LIMITS)
outputs: list = []
cash, value, held, prices = CASH, CASH, {}, {}
replayed_days = 0
largest_cash = largest_shares = 0.0
for day, day_bars in groupby(bars, key=lambda b: b.start.date()):
    session = calendar[day]
    start = SessionStart(
        day,
        session.open - timedelta(minutes=5),
        cash,
        value,
        held,
        {t: prices[t] for t in held},
    )
    before = copy.deepcopy(strategy)
    core = LiveEngine(strategy, guard, start, collar_bps=LIMITS.price_collar_bps)
    alpaca = Alpaca(core, start)
    for bar in day_bars:
        alpaca.bar(bar)
    closed = session.close + timedelta(seconds=1)
    alpaca.close(closed)
    strategy.on_session_end(core.view(closed))
    again, leftover = replayed(alpaca.entries, before)
    assert leftover == [], (day, leftover)
    assert again.orders == core.orders, day
    assert dict(again.portfolio.positions) == dict(core.portfolio.positions), day
    assert again.portfolio.cash == core.portfolio.cash, day
    # The next session starts from the backtest's account, as from Alpaca's.
    prices = dict(core.last_prices)
    cash, value, held = bt_accounts[day]
    largest_cash = max(largest_cash, abs(core.portfolio.cash - cash))
    for ticker in set(held) | set(core.portfolio.positions):
        gap = abs(held.get(ticker, 0.0) - core.portfolio.positions.get(ticker, 0.0))
        largest_shares = max(largest_shares, gap)
    replayed_days += 1
print(f"live core and replays: {time.monotonic() - clock:.0f} s")


# 3. The comparison.
def same_number(a, b):
    return (math.isnan(a) and math.isnan(b)) or math.isclose(
        a, b, rel_tol=1e-9, abs_tol=1e-9
    )


def first_difference(name, ours, theirs, same):
    if len(ours) != len(theirs):
        print(f"   {name}: {len(theirs):,} in the backtest, {len(ours):,} live")
    for i, (a, b) in enumerate(zip(ours, theirs, strict=False)):
        if not same(a, b):
            print(f"   {name} {i} differs:\n      live     {a}\n      backtest {b}")
            return False
    return len(ours) == len(theirs)


bt_notes = [r for r in bt.log if isinstance(r, DecisionNote)]
live_notes = [r for r in outputs if isinstance(r, DecisionNote)]
notes_ok = first_difference(
    "note",
    live_notes,
    bt_notes,
    lambda a, b: (
        (a.ticker, a.choice, a.reason) == (b.ticker, b.choice, b.reason)
        and a.time - b.time == DELAY
        and a.seen.keys() == b.seen.keys()
        and all(same_number(a.seen[k], b.seen[k]) for k in a.seen)
    ),
)
bt_orders = [r for r in bt.log if isinstance(r, Order)]
live_orders = [r for r in outputs if isinstance(r, Order)]
largest = max(
    (
        abs(a.intent.quantity - b.intent.quantity)
        for a, b in zip(live_orders, bt_orders, strict=False)
    ),
    default=0.0,
)
orders_ok = first_difference(
    "order",
    live_orders,
    bt_orders,
    lambda a, b: (
        (a.intent.side, a.intent.limit_price) == (b.intent.side, b.intent.limit_price)
        and abs(a.intent.quantity - b.intent.quantity) <= 1e-6
        and a.sent_at - b.sent_at == DELAY
    ),
)
bt_risk = [(r.rule, r.action) for r in bt.log if isinstance(r, RiskEvent)]
live_risk = [(r.rule, r.action) for r in outputs if isinstance(r, RiskEvent)]
risk_ok = first_difference("Guard event", live_risk, bt_risk, lambda a, b: a == b)
print(
    f"{len(calendar)} sessions in the calendar, {replayed_days} with bars, each replayed from its "
    f"journal: same orders and books"
)
print(
    f"decision notes: {len(live_notes):,} live, {len(bt_notes):,} backtest: "
    f"{'all the same' if notes_ok else 'DIFFERENT'}"
)
print(
    f"orders: {len(live_orders):,} live, {len(bt_orders):,} backtest: "
    f"{'all the same' if orders_ok else 'DIFFERENT'} (largest difference in shares {largest:.1e}); "
    f"fills reported in part {Alpaca.partial_reports:,} times; cancels {Alpaca.cancels:,}"
)
print(
    f"Guard events: {len(live_risk):,} live, {len(bt_risk):,} backtest: "
    f"{'all the same' if risk_ok else 'DIFFERENT'}"
)
print(
    f"books at each close, live against backtest: cash within {largest_cash:.1e} USD, "
    f"positions within {largest_shares:.1e} shares; at the end {value:,.2f} USD"
)
