"""Guard: each order intent against the limits in guard.yaml (decision 0014).

Guard stands between the strategy and the broker. It judges an intent on two
things only: what the View shows at that moment, and the orders it has let
through itself that day. It does not rely on the strategy to count them.

1. Six rules drop an intent (REJECT), checked in this order:
   - daily_loss_limit_pct: the account has fallen this far below its value
     at the last close. Nothing more goes out that day, even after a
     recovery; it is not a halt, which would stop a backtest for good.
   - stale_data_sec: the stock's last price is older than this, or missing.
   - price_collar_bps: the limit price lies further than this from the last
     price, on either side.
   - max_orders_per_min: this many orders went out in the last 60 seconds.
   - max_trades_per_symbol_day: this many orders in the stock went out today.
   - no_same_day_reversal: an order in the stock went the other way today.
2. Four rules cut its quantity (REDUCE), together:
   - max_order_usd: the order's value at its limit price.
   - allow_short: a sale is at most the shares held and not already in open
     sales.
   - max_weight: the stock at its last price, with its open buys and this
     one at their limits, as a share of the account's value.
   - max_gross_exposure: every position at its last price and every open
     buy at its limit, as a multiple of the account's value. An open sale
     counts only once it fills: if the buy filled and the sale did not, the
     account would borrow.
   The intent goes out with the smallest quantity they allow, and the event
   names the rule that set it. If that is no shares, the intent is dropped.

live_trading is not Guard's to check: a backtest never reaches a broker, and
the live engine reads it before it connects to one (Phase 3).
"""

import math
from dataclasses import replace
from datetime import date, timedelta

from qfin_alphaguard.events import (
    SHARE_DECIMALS,
    OrderIntent,
    RiskAction,
    RiskEvent,
    Side,
    round_shares,
)
from qfin_alphaguard.guard.check import GuardDecision
from qfin_alphaguard.guard.config import GuardConfig
from qfin_alphaguard.strategy import View

# A limit set on the cent at the collar's edge can lie past it by a
# floating-point crumb: 20 * 1.005 is 20.099999999999998, and the limit 20.10.
COLLAR_TOLERANCE_BPS = 1e-6
MINUTE = timedelta(minutes=1)


class Guard:
    """The limits of guard.yaml, applied to every order intent of one account."""

    def __init__(self, limits: GuardConfig) -> None:
        self.limits = limits
        self.day: date | None = None  # the session the two records below are for
        self.sent: list[OrderIntent] = []  # what Guard let through that day
        self.stopped = False  # whether the daily loss limit was reached that day

    def check(self, intent: OrderIntent, view: View) -> GuardDecision:
        """Let `intent` through, cut its quantity or drop it."""
        if view.session.day != self.day:
            self.day, self.sent, self.stopped = view.session.day, [], False
        dropped = self._dropped(intent, view)
        if dropped is not None:
            rule, detail = dropped
            return GuardDecision(None, _event(intent, RiskAction.REJECT, rule, detail))
        quantity, rule, detail = self._allowed(intent, view)
        if quantity == 0:
            detail += "; no shares can go"
            return GuardDecision(None, _event(intent, RiskAction.REJECT, rule, detail))
        if quantity < intent.quantity:
            cut = replace(intent, quantity=quantity)
            self.sent.append(cut)
            detail += f"; {quantity} of the {intent.quantity} shares go"
            return GuardDecision(cut, _event(intent, RiskAction.REDUCE, rule, detail))
        self.sent.append(intent)
        return GuardDecision(intent)

    def _dropped(self, intent: OrderIntent, view: View) -> tuple[str, str] | None:
        """The first rule that drops `intent`, and why; None if none does."""
        limits = self.limits
        ticker = intent.ticker
        start = view.last_close_value
        if start is None or not start > 0:
            raise ValueError(
                f"the daily loss limit needs the value at the last close, got {start!r}"
            )
        value = _value(view)
        if 100 * (1 - value / start) >= limits.daily_loss_limit_pct:
            self.stopped = True
        if self.stopped:
            return "daily_loss_limit_pct", (
                f"the account fell {limits.daily_loss_limit_pct:g}% below its "
                f"{start:,.2f} USD at the last close today; it is {value:,.2f} USD"
            )

        seen = view.price_times.get(ticker)
        if seen is None or ticker not in view.last_prices:
            return "stale_data_sec", f"no price for {ticker}"
        age = (intent.time - seen).total_seconds()
        if age > limits.stale_data_sec:
            return "stale_data_sec", (
                f"the last price of {ticker} is {age:g} seconds old; "
                f"at most {limits.stale_data_sec:g}"
            )

        last = view.last_prices[ticker]
        gap = abs(intent.limit_price - last) / last * 10_000
        if gap > limits.price_collar_bps + COLLAR_TOLERANCE_BPS:
            return "price_collar_bps", (
                f"the limit {intent.limit_price} is {gap:.1f} bps from the last "
                f"price {last}; at most {limits.price_collar_bps:g}"
            )

        recent = [order for order in self.sent if order.time > intent.time - MINUTE]
        if len(recent) >= limits.max_orders_per_min:
            return "max_orders_per_min", (
                f"{len(recent)} orders went out in the last minute; "
                f"at most {limits.max_orders_per_min}"
            )

        same = [order for order in self.sent if order.ticker == ticker]
        if len(same) >= limits.max_trades_per_symbol_day:
            return "max_trades_per_symbol_day", (
                f"{len(same)} orders in {ticker} went out today; "
                f"at most {limits.max_trades_per_symbol_day}"
            )
        if limits.no_same_day_reversal and any(
            order.side is not intent.side for order in same
        ):
            other = "sale" if intent.side is Side.BUY else "buy"
            return "no_same_day_reversal", f"a {other} of {ticker} went out today"
        return None

    def _allowed(self, intent: OrderIntent, view: View) -> tuple[float, str, str]:
        """The most shares the four cutting rules allow, the rule that sets it, why."""
        limits = self.limits
        ticker, limit, notional = intent.ticker, intent.limit_price, intent.notional
        caps: list[tuple[float, str, str]] = []
        if notional > limits.max_order_usd:
            detail = (
                f"{intent.quantity} shares of {ticker} at {limit} are "
                f"{notional:,.2f} USD; at most {limits.max_order_usd:,.2f}"
            )
            caps.append(
                (_shares(limits.max_order_usd / limit), "max_order_usd", detail)
            )
        held = view.positions.get(ticker, 0.0)
        if intent.side is Side.SELL:
            selling = sum(
                order.intent.quantity
                for order in view.open_orders
                if order.intent.ticker == ticker and order.intent.side is Side.SELL
            )
            free = max(round_shares(held - selling), 0.0)
            if not limits.allow_short and intent.quantity > free:
                detail = (
                    f"{held} shares of {ticker} held and {selling} in open sales: "
                    f"selling {intent.quantity} would sell short"
                )
                caps.append((free, "allow_short", detail))
            return _smallest(intent, caps)

        value = _value(view)
        buys = [
            order.intent for order in view.open_orders if order.intent.side is Side.BUY
        ]
        stock = held * view.last_prices[ticker] + sum(
            buy.notional for buy in buys if buy.ticker == ticker
        )
        room = limits.max_weight * value - stock
        if notional > room:
            detail = (
                f"{ticker} would be worth {stock + notional:,.2f} USD, over "
                f"{limits.max_weight:g} of the account's {value:,.2f} USD"
            )
            caps.append((_shares(room / limit), "max_weight", detail))
        gross = sum(
            abs(shares) * view.last_prices[name]
            for name, shares in view.positions.items()
        ) + sum(buy.notional for buy in buys)
        room = limits.max_gross_exposure * value - gross
        if notional > room:
            detail = (
                f"positions and buys would be worth {gross + notional:,.2f} USD, "
                f"over {limits.max_gross_exposure:g} times the account's "
                f"{value:,.2f} USD"
            )
            caps.append((_shares(room / limit), "max_gross_exposure", detail))
        return _smallest(intent, caps)


def _smallest(
    intent: OrderIntent, caps: list[tuple[float, str, str]]
) -> tuple[float, str, str]:
    """The smallest cap and its rule; the intent's own quantity if none is smaller."""
    if not caps:
        return intent.quantity, "", ""
    quantity, rule, detail = min(caps, key=lambda cap: cap[0])
    return min(quantity, intent.quantity), rule, detail


def _value(view: View) -> float:
    """The account's value: its cash and every position at its last price."""
    return view.cash + sum(
        shares * view.last_prices[ticker] for ticker, shares in view.positions.items()
    )


def _shares(quantity: float) -> float:
    """`quantity` cut down to the broker's 9 decimals, and never below 0.

    Rounded first, so a quotient that floating point leaves a crumb under a
    whole number of billionths is not cut by one: 984.90 USD at 100.50 buy
    9.8 shares, but 984.9 / 100.5 is 9.799999999999999.
    """
    scale = 10**SHARE_DECIMALS
    return max(round_shares(math.floor(round(quantity * scale, 3)) / scale), 0.0)


def _event(
    intent: OrderIntent, action: RiskAction, rule: str, detail: str
) -> RiskEvent:
    return RiskEvent(intent.time, rule, action, detail, intent.ticker)
