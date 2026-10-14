"""What Guard's limits do to the QQQ strategy (Phase 2, step 4; decision 0019).

Runs momentum.py's NoiseAreaMomentum on the stored QQQ minutes in 2016-2026
from 30,000 USD (Alpaca's fees, QQQ's measured half spread of 0.3 bps):

1. no Guard (a stand-in that sends every order): what the strategy asks for;
2. Guard with guard.yaml, the nightly plan's limits;
3. Guard with guard_qqq.yaml, as written;
4. guard_qqq.yaml with max_order_usd lifted: the file's 70,000 USD is for a
   30,000 USD account, and over ten years this one grows several times over,
   so the cap would cut every later order. This run shows the other limits;
5. run 4 with a daily loss limit of 5, 4, 3 and 2%, the alternatives to
   guard_qqq.yaml's none.

For each run: the result, Guard's events by rule, the sessions that ended
with a position still held, and the days whose fills differ from run 1. For
run 1, what the limits have to allow: orders a day and a minute, the largest
position an order asks for, and how far the account was below its last
close at its decisions.

Usage: guard_needs.py ROOT GUARD_YAML GUARD_QQQ_YAML
"""

import math
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from qfin_alphaguard.backtest.broker import CostModel, NextBarBroker
from qfin_alphaguard.backtest.engine import SessionClose, _Engine, _slices
from qfin_alphaguard.data.intraday import minute_feed
from qfin_alphaguard.events import DecisionNote, Fill, Order, RiskEvent, Side
from qfin_alphaguard.guard.check import GuardDecision
from qfin_alphaguard.guard.config import load_guard_config
from qfin_alphaguard.guard.rules import Guard
from qfin_alphaguard.momentum import NoiseAreaMomentum

ROOT = Path(sys.argv[1])
NIGHTLY, QQQ = load_guard_config(sys.argv[2]), load_guard_config(sys.argv[3])
CASH = 30_000.0
FIRST, LAST = date(2016, 1, 1), date(2026, 10, 9)
UNCAPPED = replace(QQQ, max_order_usd=1e12)
RUNS = {
    "no Guard: what the strategy asks for": None,
    "guard.yaml, the nightly plan's limits": NIGHTLY,
    "guard_qqq.yaml as written": QQQ,
    "guard_qqq.yaml, max_order_usd lifted": UNCAPPED,
    **{
        f"guard_qqq.yaml, max_order_usd lifted, daily loss limit {pct}%": replace(
            UNCAPPED, daily_loss_limit_pct=pct
        )
        for pct in (5.0, 4.0, 3.0, 2.0)
    },
}


class SendsEverything:
    """Not Guard: lets every intent through."""

    def check(self, intent, view):
        return GuardDecision(intent)


def run(limits):
    """The backtest's log, and why it stopped if it did.

    The engine's own loop rather than run_backtest, so that a run that stops
    still leaves its log.
    """
    feed = minute_feed(
        ROOT, {year: ["QQQ"] for year in range(FIRST.year, LAST.year + 1)}, FIRST, LAST
    )
    guard = SendsEverything() if limits is None else Guard(limits)
    costs = CostModel(half_spread_bps=0.3)
    engine = _Engine(NoiseAreaMomentum(), NextBarBroker(costs), guard, CASH, ())
    try:
        for bars in _slices(feed):
            engine.handle(bars)
        engine.finish()
    except ValueError as error:
        return SimpleNamespace(log=engine.log, stopped=(engine.session.day, str(error)))
    return SimpleNamespace(log=engine.log, stopped=None)


def fills_by_day(log):
    days = defaultdict(list)
    for record in log:
        if isinstance(record, Fill):
            days[record.time.date()].append(
                (record.time, record.side, record.quantity, record.price)
            )
    return days


def held_overnight(log):
    """How many sessions ended with a position still held."""
    held, count = 0.0, 0
    for record in log:
        if isinstance(record, Fill):
            shares = record.quantity if record.side is Side.BUY else -record.quantity
            held = round(held + shares, 9)
        elif isinstance(record, SessionClose):
            count += held != 0
    return count


def report(name, result, free_fills):
    closes = [r for r in result.log if isinstance(r, SessionClose)]
    if not closes:
        raise SystemExit(f"no sessions: no QQQ minutes under {ROOT}?")
    values = np.array([c.value for c in closes])
    r = values / np.concatenate([[CASH], values[:-1]]) - 1
    print(f"-- {name}")
    if result.stopped is None:
        growth = np.prod(1 + r) ** (252 / len(r)) - 1
        sharpe = r.mean() / r.std(ddof=1) * math.sqrt(252)
        drawdown = 1 - (values / np.maximum.accumulate(values)).min()
        print(
            f"   {growth:+.1%} a year, Sharpe {sharpe:.2f}, worst fall from a high "
            f"{drawdown:.1%}, {values[-1]:,.0f} USD at the end"
        )
    else:
        day, why = result.stopped
        print(f"   STOPPED on {day}: {why}")
    print(f"   sessions that ended with a position held: {held_overnight(result.log)}")
    events = Counter(
        f"{e.rule} {e.action.value}" for e in result.log if isinstance(e, RiskEvent)
    )
    print(f"   Guard's events: {dict(events.most_common()) or 'none'}")
    fills = fills_by_day(result.log)
    differ = sorted(
        day for day in fills.keys() | free_fills.keys() if fills[day] != free_fills[day]
    )
    if differ:
        shown = ", ".join(str(day) for day in differ[:5])
        more = f" and {len(differ) - 5} more" if len(differ) > 5 else ""
        print(f"   days whose fills differ from no Guard: {len(differ)}: {shown}{more}")
    else:
        print("   fills identical to no Guard on every day")
    stops = sorted(
        {
            e.time.date()
            for e in result.log
            if isinstance(e, RiskEvent) and e.rule == "daily_loss_limit_pct"
        }
    )
    if stops:
        print(f"   days the loss limit stopped new risk: {[str(d) for d in stops]}")


def needs(log):
    """What the strategy asks of the limits, from the run without Guard."""
    orders, per_minute = defaultdict(int), Counter()
    held, cash, value, asked, losses = 0.0, CASH, CASH, [], []
    last_close = CASH
    for record in log:
        if isinstance(record, Fill):
            shares = record.quantity if record.side is Side.BUY else -record.quantity
            held = round(held + shares, 9)
            cash -= shares * record.price + record.commission
        elif isinstance(record, DecisionNote):
            value = cash + held * record.seen["price"]
            losses.append((record.time.date(), 1 - value / last_close))
        elif isinstance(record, Order):
            intent = record.intent
            orders[record.sent_at.date()] += 1
            per_minute[record.sent_at] += 1
            after = held + (
                intent.quantity if intent.side is Side.BUY else -intent.quantity
            )
            if abs(after) > abs(held):  # it adds to the position
                asked.append((abs(after) * intent.limit_price / value, record.sent_at))
        elif isinstance(record, SessionClose):
            last_close = record.value
    per_day = list(orders.values())
    print("   what the limits have to allow:")
    print(
        f"   orders a day: at most {max(per_day)}, 99th percentile "
        f"{np.percentile(per_day, 99):.0f}, median {np.median(per_day):.0f}"
    )
    print(f"   orders in one minute: at most {max(per_minute.values())}")
    ratios = np.array([ratio for ratio, _ in asked])
    worst, when = max(asked)
    print(
        "   the position an order asks for, at its limit price, as a multiple of "
        f"the account at the decision: at most {worst:.3f} ({when:%Y-%m-%d}); "
        f"above 2.1: {(ratios > 2.1).sum()} of {len(ratios)} orders"
    )
    print(
        "   worst account value at a decision: "
        f"{max(loss for _, loss in losses):.2%} below the last close"
    )
    for limit in (0.02, 0.03, 0.04, 0.05):
        days = {day for day, loss in losses if loss >= limit}
        print(
            f"   days the account was {limit:.0%} or more below its last close at "
            f"a decision: {len(days)}"
        )


free = run(None)
free_fills = fills_by_day(free.log)
for name, limits in RUNS.items():
    result = free if limits is None else run(limits)
    report(name, result, free_fills)
    if result is free:
        needs(free.log)
