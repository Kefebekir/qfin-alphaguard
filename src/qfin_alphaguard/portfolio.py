"""Cash and shares held, changed by fills and by corporate actions.

Prices are as traded, so a split changes the share count, and a dividend is
paid into cash on its ex-date. Shares may be fractional (decision 0012): every
change is rounded to the broker's precision, so fills add up exactly and a
position sold to nothing disappears.
"""

import math
from collections.abc import Mapping
from types import MappingProxyType

from qfin_alphaguard.events import Fill, Side, round_shares


class Portfolio:
    def __init__(self, cash: float) -> None:
        if not math.isfinite(cash):
            raise ValueError(f"cash must be a finite number, got {cash!r}")
        self._cash = float(cash)
        self._positions: dict[str, float] = {}
        # Built once: a read-only window that always shows the current positions.
        self.positions: Mapping[str, float] = MappingProxyType(self._positions)

    @property
    def cash(self) -> float:
        return self._cash

    def apply(self, fill: Fill) -> None:
        """Book a fill: shares in or out, and its value and commission in cash."""
        shares = fill.quantity if fill.side is Side.BUY else -fill.quantity
        held = round_shares(self._positions.get(fill.ticker, 0) + shares)
        if held:
            self._positions[fill.ticker] = held
        else:
            del self._positions[fill.ticker]
        self._cash -= shares * fill.price + fill.commission

    def split(self, ticker: str, ratio: float) -> None:
        """Shares follow a split, fractions included.

        Alpaca has kept the fractions a split leaves since 2021, rather than
        paying them in cash (decision 0012). Rounding keeps 3 shares at
        1-for-3 one share, not 0.999...
        """
        held = self._positions.get(ticker, 0)
        if not held:
            return
        after = round_shares(held * ratio)
        if after:
            self._positions[ticker] = after
        else:
            del self._positions[ticker]

    def pay_out(self, ticker: str, price: float) -> None:
        """Every share of `ticker` turned into cash at `price`, as in a takeover."""
        self._cash += self._positions.pop(ticker, 0) * price

    def dividend(self, ticker: str, amount: float) -> None:
        """Cash for each share held; a short position pays it instead."""
        self._cash += self._positions.get(ticker, 0) * amount

    def value(self, prices: Mapping[str, float]) -> float:
        """Cash plus every position at `prices`, which must cover them all."""
        missing = sorted(set(self._positions) - set(prices))
        if missing:
            raise ValueError(f"no price for held tickers: {', '.join(missing)}")
        held = sum(
            shares * prices[ticker] for ticker, shares in self._positions.items()
        )
        return self._cash + held
