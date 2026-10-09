from datetime import UTC, datetime

import pytest

from qfin_alphaguard.events import Fill, Side
from qfin_alphaguard.portfolio import Portfolio

T = datetime(2026, 10, 6, 14, 0, tzinfo=UTC)


def fill(side, quantity, price, commission=0.0, ticker="A"):
    return Fill("bt-0000001", ticker, side, quantity, price, commission, T)


def test_a_buy_turns_cash_into_shares_and_pays_commission():
    portfolio = Portfolio(1_000.0)
    portfolio.apply(fill(Side.BUY, 5, 100.0, commission=0.35))
    assert portfolio.positions == {"A": 5}
    assert portfolio.cash == pytest.approx(1_000.0 - 500.0 - 0.35)


def test_selling_every_share_closes_the_position():
    portfolio = Portfolio(1_000.0)
    portfolio.apply(fill(Side.BUY, 5, 100.0))
    portfolio.apply(fill(Side.SELL, 5, 110.0, commission=0.35))
    assert portfolio.positions == {}
    assert portfolio.cash == pytest.approx(1_000.0 + 50.0 - 0.35)


def test_the_value_is_cash_plus_each_position_at_its_price():
    portfolio = Portfolio(1_000.0)
    portfolio.apply(fill(Side.BUY, 5, 100.0))
    portfolio.apply(fill(Side.BUY, 2, 50.0, ticker="B"))
    assert portfolio.value({"A": 110.0, "B": 40.0, "C": 1.0}) == pytest.approx(
        1_000.0 - 600.0 + 550.0 + 80.0
    )


def test_valuing_needs_a_price_for_every_holding():
    portfolio = Portfolio(1_000.0)
    portfolio.apply(fill(Side.BUY, 5, 100.0))
    with pytest.raises(ValueError, match="no price"):
        portfolio.value({})


def test_positions_cannot_be_changed_from_outside():
    portfolio = Portfolio(1_000.0)
    with pytest.raises(TypeError):
        portfolio.positions["A"] = 100


def test_the_books_record_a_sale_beyond_the_holding_as_short():
    # Guard forbids short selling (allow_short: false); the books only record.
    portfolio = Portfolio(1_000.0)
    portfolio.apply(fill(Side.SELL, 3, 100.0))
    assert portfolio.positions == {"A": -3}


def test_cash_must_be_a_finite_number():
    with pytest.raises(ValueError):
        Portfolio(float("nan"))
