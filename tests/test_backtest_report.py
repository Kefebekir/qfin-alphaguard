import math
import statistics
from datetime import UTC, date, datetime, timedelta

import polars as pl
import pytest

from qfin_alphaguard.backtest.broker import CostModel
from qfin_alphaguard.backtest.engine import BacktestResult, SessionClose
from qfin_alphaguard.backtest.report import (
    benchmark,
    equal_weight,
    performance,
    render,
    summary,
    trading,
    wanted_without_minutes,
)
from qfin_alphaguard.backtest.run import Backtest
from qfin_alphaguard.events import (
    DailyPlan,
    Delisting,
    Fill,
    Order,
    OrderIntent,
    RiskAction,
    RiskEvent,
    Side,
)
from qfin_alphaguard.guard.config import GuardConfig

LIMITS = GuardConfig(
    allow_short=False,
    max_weight=0.25,
    max_gross_exposure=1.0,
    max_order_usd=2000,
    price_collar_bps=50,
    max_orders_per_min=10,
    max_trades_per_symbol_day=3,
    no_same_day_reversal=True,
    daily_loss_limit_pct=2.0,
    stale_data_sec=15,
    live_trading=False,
)
T = datetime(2026, 10, 6, 14, 30, tzinfo=UTC)
YEAR_END = [date(2016, 12, 28), date(2016, 12, 29), date(2016, 12, 30)]
NEW_YEAR = [date(2017, 1, 3), date(2017, 1, 4)]


# --- Performance ----------------------------------------------------------------


def test_performance_of_a_known_series():
    # From 100: +10%, -10%, +10%.
    result = performance([110.0, 99.0, 108.9], 100.0)
    returns = [0.1, -0.1, 0.1]
    deviation = statistics.stdev(returns)
    assert result.total_return == pytest.approx(0.089)
    assert result.annual_return == pytest.approx(1.089 ** (252 / 3) - 1)
    assert result.annual_volatility == pytest.approx(deviation * math.sqrt(252))
    assert result.sharpe == pytest.approx(
        statistics.mean(returns) / deviation * math.sqrt(252)
    )
    assert result.max_drawdown == pytest.approx(-0.1)  # 110 to 99
    assert result.winning_days == pytest.approx(2 / 3)
    assert result.worst_day == pytest.approx(-0.1)
    assert result.best_day == pytest.approx(0.1)


def test_a_value_that_never_moves_has_no_sharpe_ratio_and_no_winning_day():
    flat = performance([100.0, 100.0], 100.0)
    assert math.isnan(flat.sharpe)
    assert flat.winning_days == 0  # a day without a gain is not a winning day


@pytest.mark.parametrize("values", [[100.0], [100.0, 0.0], [-5.0, 100.0]])
def test_performance_needs_two_sessions_of_positive_values(values):
    with pytest.raises(ValueError, match="two sessions"):
        performance(values, 100.0)


# --- Trading ----------------------------------------------------------------------


def test_trading_counts_orders_turnover_costs_guard_and_payouts():
    intent = OrderIntent("A", Side.BUY, 10, 100.5, T)
    dropped = OrderIntent("B", Side.BUY, 1, 50.0, T)  # Guard let it not go
    log = (
        intent,
        dropped,
        Order("bt-0000001", intent, T),
        Fill("bt-0000001", "A", Side.BUY, 10, 100.0, 0.02, T + timedelta(minutes=1)),
        RiskEvent(T, "max_gross_exposure", RiskAction.REDUCE, "", "A"),
        RiskEvent(T, "daily_loss_limit_pct", RiskAction.REJECT, "", "B"),
        RiskEvent(T, "daily_loss_limit_pct", RiskAction.REJECT, "", "C"),
        Delisting("D", date(2026, 10, 7)),
    )
    closes = (
        SessionClose(date(2026, 10, 6), T, 0.0, 1_010.0),
        SessionClose(date(2026, 10, 7), T + timedelta(days=1), 0.0, 1_030.0),
    )
    result = BacktestResult(log, closes, 0.0, {})

    done = trading(result, CostModel(), 1_000.0)

    average, years = (1_000.0 + 1_010.0 + 1_030.0) / 3, 2 / 252
    assert done.orders == 1
    assert done.turnover == pytest.approx(1_000.0 / average / years)
    assert done.fees == pytest.approx(0.02)
    assert done.spread == pytest.approx(0.1)  # 1 bp of 1,000 USD
    assert done.costs_a_year == pytest.approx(0.12 / average / years)
    assert done.guard == {
        "daily_loss_limit_pct reject": 2,
        "max_gross_exposure reduce": 1,
    }
    assert done.delistings == 1


# --- Baselines ----------------------------------------------------------------------


def two_stocks(make_daily):
    """X: 10, 11, 12 to the year's end, then 12 and 18; Y flat at 20."""
    days = YEAR_END + NEW_YEAR
    x = make_daily("X", days, [10.0, 11.0, 12.0, 12.0, 18.0])
    y = make_daily("Y", days, [20.0] * 5)
    return pl.concat([x, y])


def test_equal_weight_buys_each_year_s_stocks_in_equal_parts_and_holds_them(
    make_daily,
):
    daily = two_stocks(make_daily)
    universes = {2016: ("X", "Y"), 2017: ("X", "Y")}
    days = [*YEAR_END[1:], *NEW_YEAR]

    values = equal_weight(daily, universes, days)

    # 2016, from the close of 28 December: half in X at 10, half in Y at 20.
    # 2017, from the close of 30 December: 1.10 split again, X at 12.
    assert values == pytest.approx([1.05, 1.10, 1.10, 0.55 / 12 * 18 + 0.55])


def test_a_stock_whose_bars_end_is_held_as_cash_at_its_last_close(make_daily):
    x = make_daily("X", YEAR_END, [10.0, 11.0, 12.0])
    z = make_daily("Z", YEAR_END[:2], [10.0, 5.0])  # no bar after 29 December
    daily = pl.concat([x, z])

    values = equal_weight(daily, {2016: ("X", "Z")}, YEAR_END[1:])

    assert values == pytest.approx([0.55 + 0.25, 0.60 + 0.25])


def test_spy_is_bought_at_the_close_before_the_first_day(make_daily):
    daily = make_daily("SPY", YEAR_END, [200.0, 210.0, 220.0], member=False)
    assert benchmark(daily, "SPY", YEAR_END[1:]) == pytest.approx([1.05, 1.10])


def test_a_benchmark_without_bars_or_with_a_gap_is_not_measured(make_daily):
    daily = make_daily("X", YEAR_END, [10.0, 11.0, 12.0])
    with pytest.raises(ValueError, match="no daily bars for SPY"):
        benchmark(daily, "SPY", YEAR_END[1:])
    gap = make_daily("SPY", [YEAR_END[0], YEAR_END[2]], [200.0, 220.0], member=False)
    with pytest.raises(ValueError, match="lacks daily bars"):
        benchmark(gap, "SPY", YEAR_END[1:])


# --- Data ---------------------------------------------------------------------------


def test_stock_days_a_plan_held_without_minutes_are_counted():
    d1, d2 = date(2016, 1, 5), date(2016, 1, 6)
    plans = {
        d1: DailyPlan(d1, "test", {"A": 0.2, "B": 0.2}),
        d2: DailyPlan(d2, "test", {"A": 0.2, "V": 0.1}),
    }
    minute_days = {"A": frozenset({d1, d2}), "B": frozenset({d1})}
    assert wanted_without_minutes(plans, minute_days) == 1  # V on the 6th


# --- The report -----------------------------------------------------------------------


def test_the_report_holds_every_run_the_baselines_and_the_data(store):
    raw, daily = store
    test = Backtest(daily, raw, LIMITS)
    result = test.run(1_000.0)

    report = summary(
        daily, test.inputs, test.plans(1_000.0), {1_000.0: result}, test.costs
    )

    assert (report["first"], report["last"], report["sessions"]) == (
        "2016-01-05",
        "2016-01-07",
        3,
    )
    assert report["plan"]["1000"]["fingerprint"] == result.fingerprint()
    assert set(report["baselines"]) == {"equal weight", "SPY"}
    assert report["missing baselines"] == {}
    assert report["data"] == {
        "delistings": 0,
        "plan stock-days": 6,
        "plan stock-days without minutes": 0,
    }
    text = render(report)
    for line in (
        "Plan, 1,000 USD",
        "Same stocks, equal weight",
        "SPY",
        "Guard at 1,000 USD: nothing",
        f"Fingerprint at 1,000 USD: {result.fingerprint()}",
    ):
        assert line in text


def test_a_benchmark_missing_from_the_store_is_reported_as_not_measured(store):
    raw, daily = store
    without_spy = daily.filter(pl.col("ticker") != "SPY")
    test = Backtest(without_spy, raw, LIMITS)
    result = test.run(1_000.0)

    report = summary(
        without_spy, test.inputs, test.plans(1_000.0), {1_000.0: result}, test.costs
    )

    assert report["missing baselines"] == {"SPY": "no daily bars for SPY"}
    assert "SPY" not in report["baselines"]
    assert "not measured: no daily bars for SPY" in render(report)


def test_a_run_that_missed_a_session_is_refused(store):
    raw, daily = store
    test = Backtest(daily, raw, LIMITS)
    result = test.run(1_000.0)
    short = BacktestResult(result.log, result.closes[:2], result.cash, result.positions)
    with pytest.raises(ValueError, match="missed some of the sessions"):
        summary(daily, test.inputs, test.plans(1_000.0), {1_000.0: short}, test.costs)
