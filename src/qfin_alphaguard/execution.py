"""Equal-spaced child orders: when the day's trades go out, without a model.

The nightly plan says what to hold, and the band (rebalance.py) which trades
to make today. This strategy only chooses when to make them (decision 0001),
on decision 0013's schedule:

1. At start_after_open_min after the open, 10:30 New York time, once the
   opening's wide spreads have narrowed, the band decides the day's trades at
   the latest prices. If a price is still missing, it tries again on the next
   bar, until the deadline.
2. A trade is split into N child orders, at most max_children, each worth at
   least min_trade_usd and min_gap_min minutes apart. They go at equal
   intervals from then to deadline_before_close_min before the close: 10:30,
   13:00 and 15:30.
3. Each child is what is left of the trade over the children left. A child
   still open when the next is due is cancelled, and its rest goes with the
   next one; the last child takes all that is left.
4. A child is a limit order at the last price plus Guard's price collar for
   a buy, minus it for a sale, rounded to the cent towards the last price: it
   fills at once on an ordinary minute and never breaks the collar. A buy
   child worth less than min_trade_usd, Alpaca's smallest buy, waits for the
   next child; a sale of any size goes.

This is the baseline that Phase 2's timing model has to beat (Gate M2).
"""

import math
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta

from qfin_alphaguard.events import (
    Bar,
    Cancellation,
    DailyPlan,
    Fill,
    OrderIntent,
    Side,
    round_shares,
)
from qfin_alphaguard.rebalance import trades_at_open
from qfin_alphaguard.strategy import Decision, Strategy, View


class EqualSpaced(Strategy):
    """Each day's trades from the band, in equal-spaced child orders."""

    def __init__(self, plans: Mapping[date, DailyPlan], collar_bps: float) -> None:
        self.plans = plans  # the plan for each trading day
        self.collar = collar_bps / 10_000  # Guard's price_collar_bps, as a fraction
        self.plan: DailyPlan | None = None  # today's
        self.decided = False  # whether the band has decided today's trades
        self.left: dict[str, float] = {}  # shares still to trade; negative to sell
        self.due: dict[str, list[datetime]] = {}  # when each child still to send goes
        self.children: dict[str, int] = {}  # how many children each trade has

    def on_session_start(self, view: View) -> None:
        self.plan = self.plans.get(view.session.day)
        self.decided = False
        self.left, self.due, self.children = {}, {}, {}

    def on_bars(self, bars: Mapping[str, Bar], view: View) -> Sequence[Decision]:
        if self.plan is None:
            return ()
        settings = self.plan.execution
        start = view.session.open + timedelta(minutes=settings.start_after_open_min)
        deadline = view.session.close - timedelta(
            minutes=settings.deadline_before_close_min
        )
        if not self.decided and start <= view.now <= deadline:
            self.decide(view, deadline)
        decisions: list[Decision] = []
        for ticker, due in self.due.items():
            if due and due[0] <= view.now:
                due.pop(0)
                decisions += self.send(ticker, view)
        return decisions

    def on_fill(self, fill: Fill, view: View) -> None:
        if fill.ticker in self.left:
            done = fill.quantity if fill.side is Side.BUY else -fill.quantity
            self.left[fill.ticker] = round_shares(self.left[fill.ticker] - done)

    def decide(self, view: View, deadline: datetime) -> None:
        """The band's trades for today, and when each child goes.

        Only a missing price waits for the next bar. Any other error of the
        band, such as a plan without volatility, is a bug and stops the run.
        """
        needed = set(self.plan.target_weights) | set(view.positions)
        if any(ticker not in view.last_prices for ticker in needed):
            return  # a price is still missing: try again on the next bar
        trades = trades_at_open(self.plan, view.positions, view.last_prices, view.cash)
        self.decided = True
        settings = self.plan.execution
        gap = timedelta(minutes=settings.min_gap_min)
        for ticker, change in trades.items():
            worth = abs(change) * view.last_prices[ticker]
            n = math.floor(worth / settings.min_trade_usd)
            n = min(n, settings.max_children)
            if gap:
                n = min(n, 1 + math.floor((deadline - view.now) / gap))
            n = max(n, 1)
            step = (deadline - view.now) / (n - 1) if n > 1 else timedelta(0)
            self.left[ticker] = change
            self.due[ticker] = [view.now + k * step for k in range(n)]
            self.children[ticker] = n

    def send(self, ticker: str, view: View) -> list[Decision]:
        """The child due now, after cancelling the stock's open child."""
        decisions: list[Decision] = [
            Cancellation(order.client_order_id, view.now, "its rest goes with the next")
            for order in view.open_orders
            if order.intent.ticker == ticker
        ]
        left = self.left[ticker]
        children_left = len(self.due[ticker]) + 1
        quantity = round_shares(abs(left) / children_left)
        price = view.last_prices[ticker]
        buy = left > 0
        minimum = self.plan.execution.min_trade_usd
        if quantity == 0 or (buy and quantity * price < minimum):
            return decisions
        if buy:
            limit = math.floor(round(price * (1 + self.collar) * 100, 6)) / 100
        else:
            limit = math.ceil(round(price * (1 - self.collar) * 100, 6)) / 100
        number = self.children[ticker] - len(self.due[ticker])
        reason = f"child {number} of {self.children[ticker]}"
        side = Side.BUY if buy else Side.SELL
        decisions.append(OrderIntent(ticker, side, quantity, limit, view.now, reason))
        return decisions
