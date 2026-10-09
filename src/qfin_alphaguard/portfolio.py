"""Cash and shares held, changed by fills and by corporate actions.

Prices are as traded, so a split changes the share count, and a dividend is
paid into cash on its ex-date.
"""

import math
from collections.abc import Mapping
from types import MappingProxyType

from qfin_alphaguard.events import Fill, Side


class Portfolio:
    def __init__(self, cash: float) -> None:
        if not math.isfinite(cash):
            raise ValueError(f"cash must be a finite number, got {cash!r}")
        self._cash = float(cash)
        self._positions: dict[str, int] = {}
        # Built once: a read-only window that always shows the current positions.
        self.positions: Mapping[str, int] = MappingProxyType(self._positions)

    @property
    def cash(self) -> float:
        return self._cash

    def apply(self, fill: Fill) -> None:
        """Book a fill: shares in or out, and its value and commission in cash."""
        shares = fill.quantity if fill.side is Side.BUY else -fill.quantity
        held = self._positions.get(fill.ticker, 0) + shares
        if held:
            self._positions[fill.ticker] = held
        else:
            del self._positions[fill.ticker]
        self._cash -= shares * fill.price + fill.commission

    def split(self, ticker: str, ratio: float, price: float) -> None:
        """Shares follow a split; a fraction of a share is paid in cash at `price`.

        `price` is per new share. Brokers pay such fractions as "cash in lieu".
        """
        held = self._positions.get(ticker, 0)
        if not held:
            return
        exact = round(held * ratio, 9)  # so 3 shares at 1-for-3 make 1, not 0.999...
        whole = math.trunc(exact)
        if whole:
            self._positions[ticker] = whole
        else:
            del self._positions[ticker]
        self._cash += (exact - whole) * price

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
