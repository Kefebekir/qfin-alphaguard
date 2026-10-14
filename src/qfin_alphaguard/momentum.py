"""Intraday momentum on the Nasdaq-100: the noise-area rules of decision 0011.

Every half hour from 10:00 New York time the strategy compares QQQ's last
price with its noise band and VWAP, and holds long, short or nothing until
the next half hour. It is flat at every close. Decision 0018 makes it the
intraday AI's decision engine; `research/intraday_pilot/qqq_momentum.py` is
the research it must reproduce.

The rules are four functions (Phase 2, step 2: Efe's):

- noise_band: how far from the open the price may wander on a normal day;
- daily_volatility: how much the last two weeks moved from close to close;
- next_position: long, short or flat, from the price, the band and VWAP;
- target_shares: how many shares that position is.

The class around them remembers what they need and turns a position into
orders:

- It learns from the feed: each session's moves away from its open, minute
  by minute, and its last close. It trades once it has `lookback` sessions
  behind it, and the close before them (15 closes for 14 returns).
- A decision uses the bar that ended at that moment, so its order fills on
  a later bar (the engine's rule). If that bar is missing, the decision is
  made on the first bar after it.
- An order is a limit order at the last price plus Guard's price collar
  (minus it for a sale), as EqualSpaced sends them: it fills at once on an
  ordinary minute.
- A long may be fractional; a short is whole shares, because Alpaca sells
  short only whole shares. Alpaca takes no order that turns a long into a
  short or back: the strategy closes the position first and opens the other
  side on the bar after the close has filled.
- One minute before the close it closes whatever is held, and makes no
  decision after that.
- Every decision, a kept position included, leaves a DecisionNote in the
  log: what it chose, why in words (describe), and the numbers it saw. So
  does the close at the day's end. A day without enough history leaves none.
"""

import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import pairwise

from qfin_alphaguard.events import (
    Bar,
    Cancellation,
    DecisionNote,
    OrderIntent,
    Side,
    round_shares,
)
from qfin_alphaguard.strategy import Decision, Strategy, View


@dataclass(frozen=True)
class MomentumSettings:
    """How the strategy trades. The risk limits stay in guard.yaml."""

    ticker: str = "QQQ"
    cap: float = 2.0  # most held, long or short, as a multiple of the account
    target_vol: float = 0.02  # each day is sized for this daily volatility
    lookback: int = 14  # sessions behind the noise band and the volatility
    every_min: int = 30  # minutes between decisions, from the open
    long_only: bool = False  # a cash account cannot sell short
    collar_bps: float = 50.0  # Guard's price_collar_bps

    def __post_init__(self) -> None:
        for name in ("cap", "target_vol", "collar_bps"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be a positive number, got {value!r}")
        for name in ("lookback", "every_min"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f"{name} must be a whole number of at least 2")


# The rules (Efe's). Plain functions of numbers: no engine, no clock.


def noise_band(
    past_moves: Sequence[Sequence[float]],
    minute: int,
    day_open: float,
    prev_close: float,
) -> tuple[float, float]:
    """The noise band's lower and upper edge at `minute` minutes after the open.

    `past_moves` are the last `lookback` sessions, oldest first. A session's
    moves are indexed by minutes since its open: how far its last price was
    from its open at the end of each minute, as abs(close / open - 1). A half
    day has fewer of them.

    sigma is the average of the past sessions' moves at `minute`, over the
    sessions that reach `minute`. Around today's open and yesterday's close:

        lower = min(day_open, prev_close) * (1 - sigma)
        upper = max(day_open, prev_close) * (1 + sigma)

    A move can be NaN (a session whose first minutes have no trade); it makes
    sigma, and so both edges, NaN, as in the research.
    """

    reached = [moves[minute] for moves in past_moves if minute < len(moves)]
    if not reached:
        return math.nan, math.nan
    sigma = sum(reached) / len(reached)
    lower = min(day_open, prev_close) * (1 - sigma)
    upper = max(day_open, prev_close) * (1 + sigma)
    return lower, upper


def daily_volatility(closes: Sequence[float]) -> float:
    """The standard deviation of the log returns between `closes`.

    `closes` are the last `lookback` + 1 sessions' closes, oldest first; the
    sample standard deviation (ddof=1) of their `lookback` log returns.
    """
    returns = [math.log(b / a) for a, b in pairwise(closes)]
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    return math.sqrt(variance)


def next_position(
    held: int, price: float, lower: float, upper: float, vwap: float
) -> int:
    """The position to hold until the next decision: 1 long, -1 short, 0 flat.

    `held` is the position the rules chose last time.

    - Flat: long above the band (price > upper), short below it
      (price < lower), otherwise flat.
    - Long: keep it while price >= max(upper, vwap), the tighter of the band's
      top and VWAP. Below it the long is stopped: short if the price is also
      below the band, otherwise flat.
    - Short: the mirror image, with min(lower, vwap).

    Compare exactly as written: a NaN edge makes a comparison false, so the
    rules then stay flat, or keep a long or short, as the research does.
    """
    if held == 0:
        if price > upper:
            return 1
        if price < lower:
            return -1
        return 0
    if held == 1:
        if price < max(upper, vwap):
            return -1 if price < lower else 0
        return 1
    if price > min(lower, vwap):
        return 1 if price > upper else 0
    return -1


def target_shares(
    position: int,
    account_value: float,
    price: float,
    volatility: float,
    cap: float,
    target_vol: float,
) -> float:
    """The shares to hold for `position`: positive long, negative short.

    The size is min(cap, target_vol / volatility) times the account's value,
    in shares at `price`. A long is rounded down to the broker's precision,
    9 decimals; a short is rounded down to whole shares, the only shorts
    Alpaca takes. Flat is 0. Rounding down never holds more than the size.
    """
    if position == 0:
        return 0.0
    size = min(cap, target_vol / volatility)
    shares = size * account_value / price
    if position > 0:
        return math.floor(round(shares * 1e9, 3)) / 1e9
    return -float(math.floor(shares))


# The strategy (Claude's skeleton around the rules).

CHOICES = {1: "long", 0: "flat", -1: "short"}


def describe(
    before: int,
    wanted: int,
    after: int,
    price: float,
    lower: float,
    upper: float,
    vwap: float,
) -> str:
    """A decision in words, for the decision log.

    `before` is the position the rules held, `wanted` what next_position
    chose, and `after` what is held: `wanted`, unless long_only turned a
    short into flat.
    """
    if math.isnan(lower) or math.isnan(upper):
        return "no noise band at this minute (a past session lacks it): no change"
    if wanted == -1 and after == 0:
        return "below the band, but long only: stay flat"
    if before == after:
        if after == 0:
            return "inside the noise band: stay flat"
        stop = max(upper, vwap) if after == 1 else min(lower, vwap)
        return f"the trend holds; its stop is at {stop:.2f}"
    if after == 1:
        return (
            "above the band: buy"
            if before == 0
            else "short stopped above the band: buy"
        )
    if after == -1:
        if before == 0:
            return "below the band: sell short"
        return "long stopped below the band: sell short"
    if before == 1:
        return "long stopped: price fell below the band's top or VWAP"
    return "short stopped: price rose above the band's bottom or VWAP"


class NoiseAreaMomentum(Strategy):
    """QQQ intraday momentum: the rules above, every `every_min` minutes."""

    def __init__(self, settings: MomentumSettings | None = None) -> None:
        self.settings = settings or MomentumSettings()
        lookback = self.settings.lookback
        self.moves: deque[list[float]] = deque(maxlen=lookback)  # past sessions
        self.closes: deque[float] = deque(maxlen=lookback + 1)  # their last closes
        self.prices: dict[int, float] = {}  # today's closes, by minute since open
        self.day_open: float | None = None  # today's first price
        self.flow = 0.0  # today's sum of typical price times volume, for VWAP
        self.volume = 0  # today's volume
        self.held = 0  # the position the rules chose last: -1, 0 or 1
        self.target = 0.0  # the shares that position is
        self.next_due: datetime | None = None  # the next decision's time
        self.closed = False  # the day's last minute: everything closed
        self.volatility: float | None = None  # today's, from past closes
        self.account_value = 0.0  # at today's open

    def on_session_start(self, view: View) -> None:
        self.prices, self.day_open = {}, None
        self.flow, self.volume = 0.0, 0
        self.held, self.target, self.closed = 0, 0.0, False
        self.next_due = view.session.open + timedelta(minutes=self.settings.every_min)
        ready = len(self.closes) == self.closes.maxlen
        self.volatility = daily_volatility(list(self.closes)) if ready else None
        self.account_value = view.cash + sum(
            shares * view.last_prices[ticker]
            for ticker, shares in view.positions.items()
        )

    def on_bars(self, bars: Mapping[str, Bar], view: View) -> Sequence[Decision]:
        bar = bars.get(self.settings.ticker)
        if bar is not None:
            self.record(bar, view)
        if self.closed or self.day_open is None:
            return ()
        notes: list[Decision] = []
        if view.now >= view.session.close - timedelta(minutes=1):
            self.closed = True
            self.held, self.target = 0, 0.0
            if self.tradable():
                notes.append(self.closing_note(view))
            return notes + self.move(view)
        if self.next_due is not None and view.now >= self.next_due:
            every = timedelta(minutes=self.settings.every_min)
            while self.next_due <= view.now:
                self.next_due += every
            if self.next_due >= view.session.close:
                self.next_due = None  # no decision at or after the close
            if self.tradable():
                notes.append(self.decide(view))
        return notes + self.move(view)

    def on_session_end(self, view: View) -> None:
        if self.day_open is None:
            return  # no bar today: the session teaches nothing
        last = max(self.prices)
        moves, price = [], math.nan
        for minute in range(last + 1):
            price = self.prices.get(minute, price)  # a missing minute keeps the last
            moves.append(abs(price / self.day_open - 1))
        self.moves.append(moves)
        self.closes.append(self.prices[last])

    def record(self, bar: Bar, view: View) -> None:
        """Remember the bar: its close by minute, and its part of VWAP."""
        minute = int((bar.start - view.session.open).total_seconds() // 60)
        if self.day_open is None:
            self.day_open = bar.open
        self.prices[minute] = bar.close
        self.flow += (bar.high + bar.low + bar.close) / 3 * bar.volume
        self.volume += bar.volume

    def tradable(self) -> bool:
        """Enough history for the band and the volatility, and a day to trade."""
        return (
            len(self.moves) == self.moves.maxlen
            and self.volatility is not None
            and self.volatility > 0
            and self.account_value > 0
        )

    def decide(self, view: View) -> DecisionNote:
        """The rules' position now, the shares it is, and the note saying why.

        A position the rules keep keeps its shares: as in the research, it is
        sized once, when it is taken, and not traded again until it changes.
        """
        minute = max(self.prices)
        price = self.prices[minute]
        lower, upper = noise_band(
            list(self.moves), minute, self.day_open, self.closes[-1]
        )
        vwap = self.flow / self.volume if self.volume else price
        before = self.held
        wanted = next_position(before, price, lower, upper, vwap)
        held = max(wanted, 0) if self.settings.long_only else wanted
        if held != before:
            self.held = held
            self.target = target_shares(
                held,
                self.account_value,
                price,
                self.volatility,
                self.settings.cap,
                self.settings.target_vol,
            )
        reason = describe(before, wanted, held, price, lower, upper, vwap)
        seen = {
            "price": price,
            "lower": lower,
            "upper": upper,
            "vwap": vwap,
            "shares": self.target,
            "volatility": self.volatility,
            "account": self.account_value,
        }
        return DecisionNote(self.settings.ticker, view.now, CHOICES[held], reason, seen)

    def closing_note(self, view: View) -> DecisionNote:
        """The note for the close a minute before the session ends."""
        held = view.positions.get(self.settings.ticker, 0.0)
        reason = "the day's end: close the position" if held else "the day's end: flat"
        seen = {"price": view.last_prices[self.settings.ticker], "shares": 0.0}
        return DecisionNote(self.settings.ticker, view.now, "flat", reason, seen)

    def move(self, view: View) -> list[Decision]:
        """Orders that take the position towards the target, on every bar.

        An order still open is cancelled and sent again for what is left at
        the new price: it was decided at an earlier price. On an ordinary
        minute an order fills on the next bar, so nothing is open by then.
        Going from long to short or back takes two steps, because Alpaca
        takes no order that crosses zero: close now, and open the other side
        on the bar after the close has filled.
        """
        ticker = self.settings.ticker
        decisions: list[Decision] = [
            Cancellation(order.client_order_id, view.now, "a new decision")
            for order in view.open_orders
            if order.intent.ticker == ticker
        ]
        if ticker not in view.last_prices:
            return decisions
        held = view.positions.get(ticker, 0.0)
        if held and self.target and (held > 0) != (self.target > 0):
            change = -held  # close first; the other side comes after the fill
            reason = "close before turning"
        else:
            change = round_shares(self.target - held)
            reason = "decision" if not self.closed else "close at the day's end"
        if not change:
            return decisions
        price = view.last_prices[ticker]
        collar = self.settings.collar_bps / 10_000
        buy = change > 0
        if buy:
            limit = math.floor(round(price * (1 + collar) * 100, 6)) / 100
        else:
            limit = math.ceil(round(price * (1 - collar) * 100, 6)) / 100
        side = Side.BUY if buy else Side.SELL
        quantity = round_shares(abs(change))
        decisions.append(OrderIntent(ticker, side, quantity, limit, view.now, reason))
        return decisions
