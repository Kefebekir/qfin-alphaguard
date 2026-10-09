from datetime import date

import numpy as np
import polars as pl
import pytest

from qfin_alphaguard.config import Config
from qfin_alphaguard.data.synthetic import generate_prices
from qfin_alphaguard.events import ExecutionSettings
from qfin_alphaguard.plan import (
    build_plan,
    build_plans,
    plan_from_json,
    plan_to_json,
    positions_for,
)

PRICES = generate_prices(Config())  # 30 stocks, 2015 to 2024, all index members
DAY = date(2024, 6, 3)


@pytest.mark.parametrize(
    ("capital", "count"), [(100_000, 33), (30_000, 10), (10_000, 3), (500, 1)]
)
def test_capital_sets_how_many_stocks_are_held(capital, count):
    # Each position must be big enough for 3 orders of at least 1,000 USD.
    assert positions_for(capital, ExecutionSettings()) == count


def test_a_plan_is_fully_invested_within_the_weight_limit():
    plan = build_plan(PRICES, DAY, capital=100_000, max_weight=0.25)
    weights = plan.target_weights.values()
    assert sum(weights) == pytest.approx(1.0)
    assert max(weights) <= 0.25 + 1e-9
    assert set(plan.target_weights) <= set(Config().tickers)
    assert all(plan.volatility[ticker] > 0 for ticker in plan.target_weights)


def test_volatility_is_each_stocks_own_over_the_last_year():
    plan = build_plan(PRICES, DAY, capital=100_000, max_weight=0.25)
    ko = PRICES.filter((pl.col("ticker") == "KO") & (pl.col("date") < DAY))
    returns = np.diff(np.log(ko.sort("date")["close"].to_numpy()))[-252:]
    # A daily figure, not annualised, and not shrunk towards the other stocks.
    assert plan.volatility["KO"] == pytest.approx(returns.std(ddof=1), rel=1e-9)


def test_no_stock_gets_more_than_the_weight_limit():
    # A very calm stock would take most of a minimum-variance portfolio.
    calm = PRICES.filter(pl.col("ticker") == "KO").sort("date")
    path = 100 * np.exp(
        np.cumsum(np.random.default_rng(1).normal(0, 0.001, calm.height))
    )
    prices = pl.concat(
        [
            PRICES.filter(pl.col("ticker") != "KO"),
            calm.with_columns(pl.Series("close", path)),
        ]
    )

    plan = build_plan(prices, DAY, capital=100_000, max_weight=0.25)

    assert plan.target_weights["KO"] == pytest.approx(0.25)
    assert max(plan.target_weights.values()) <= 0.25 + 1e-9


def test_capital_limits_the_number_of_stocks():
    plan = build_plan(PRICES, DAY, capital=30_000, max_weight=0.25)
    assert len(plan.target_weights) <= 10
    assert sum(plan.target_weights.values()) == pytest.approx(1.0)


def test_too_little_capital_for_the_limit_keeps_the_rest_in_cash():
    # 10,000 USD holds 3 stocks; at most 25% each, so 25% stays in cash.
    plan = build_plan(PRICES, DAY, capital=10_000, max_weight=0.25)
    assert sorted(plan.target_weights.values()) == pytest.approx([0.25, 0.25, 0.25])


def test_a_plan_uses_nothing_from_its_own_day_or_later():
    later = pl.col("date") >= DAY
    changed = PRICES.with_columns(
        pl.when(later)
        .then(pl.col("close") * 3)
        .otherwise(pl.col("close"))
        .alias("close")
    )
    original = build_plan(PRICES, DAY, capital=100_000, max_weight=0.25)
    assert build_plan(changed, DAY, capital=100_000, max_weight=0.25) == original
    shortened = PRICES.filter(~later)
    assert build_plan(shortened, DAY, capital=100_000, max_weight=0.25) == original


def test_without_a_year_of_returns_there_is_no_plan():
    # Bars from June 2015 are enough to choose the 2016 universe (126 days),
    # but not for a year of returns before 5 January 2016.
    recent = PRICES.filter(pl.col("date") >= date(2015, 6, 1))
    with pytest.raises(ValueError, match="year of returns"):
        build_plan(recent, date(2016, 1, 5), capital=30_000, max_weight=0.25)


def test_plans_for_many_days_are_the_plans_of_each_day():
    # KO leaves the index on the last session of 2023: it may be held in 2023
    # but not in 2024, whose universe comes from the members on that session.
    last_of_2023 = PRICES.filter(pl.col("date").dt.year() == 2023)["date"].max()
    prices = PRICES.with_columns(
        pl.when((pl.col("ticker") == "KO") & (pl.col("date") >= last_of_2023))
        .then(False)
        .otherwise(pl.col("sp500"))
        .alias("sp500")
    )
    days = [date(2023, 12, 1), date(2024, 1, 2), DAY]

    plans = build_plans(prices, days, capital=100_000, max_weight=0.25)

    assert plans == {day: build_plan(prices, day, 100_000, 0.25) for day in days}
    assert "KO" in plans[date(2023, 12, 1)].volatility
    assert "KO" not in plans[date(2024, 1, 2)].volatility


def test_a_plan_survives_a_trip_through_json():
    plan = build_plan(PRICES, DAY, capital=30_000, max_weight=0.25)
    assert plan_from_json(plan_to_json(plan)) == plan
