"""Cash and shares held, changed only by fills.

Prices here are as traded, so a split while a position is held needs its
share count adjusted; corporate actions arrive in Phase 1, step 5b.
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

    def value(self, prices: Mapping[str, float]) -> float:
        """Cash plus every position at `prices`, which must cover them all."""
        missing = sorted(set(self._positions) - set(prices))
        if missing:
            raise ValueError(f"no price for held tickers: {', '.join(missing)}")
        held = sum(
            shares * prices[ticker] for ticker, shares in self._positions.items()
        )
        return self._cash + held
