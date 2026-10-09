import math
from dataclasses import FrozenInstanceError
from datetime import UTC, date, datetime, timedelta, timezone

import numpy as np
import pytest

from qfin_alphaguard.events import (
    Bar,
    Cancellation,
    DailyPlan,
    Dividend,
    ExecutionSettings,
    Fill,
    Order,
    OrderIntent,
    RiskEvent,
    Side,
    Signal,
    Split,
)

T = datetime(2027, 4, 12, 14, 35, tzinfo=UTC)
PLAN_DAY = date(2027, 4, 12)

# These two times are wrong on purpose: the tests check that events reject them.
NAIVE = datetime(2027, 4, 12, 14, 35)  # noqa: DTZ001
LONDON_SUMMER = datetime(2027, 4, 12, 15, 35, tzinfo=timezone(timedelta(hours=1)))


def make_bar(**changes):
    values = {
        "ticker": "AAPL",
        "start": T,
        "seconds": 60,
        "open": 100.0,
        "high": 101.0,
        "low": 99.5,
        "close": 100.5,
        "volume": 12_000,
    }
    values.update(changes)
    return Bar(**values)


def make_intent(**changes):
    values = {
        "ticker": "AAPL",
        "side": Side.BUY,
        "quantity": 10,
        "limit_price": 100.0,
        "time": T,
    }
    values.update(changes)
    return OrderIntent(**values)


def test_events_cannot_be_changed_after_creation():
    bar = make_bar()
    with pytest.raises(FrozenInstanceError):
        bar.close = 50.0


@pytest.mark.parametrize("start", [NAIVE, LONDON_SUMMER])
def test_times_must_be_utc(start):
    with pytest.raises(ValueError, match="UTC"):
        make_bar(start=start)


def test_bar_end_is_start_plus_length():
    assert make_bar().end == T + timedelta(minutes=1)


def test_bar_rejects_close_above_high():
    with pytest.raises(ValueError):
        make_bar(close=102.0)


def test_bar_rejects_nan_close():
    with pytest.raises(ValueError):
        make_bar(close=math.nan)


def test_bar_rejects_negative_volume():
    with pytest.raises(ValueError):
        make_bar(volume=-1)


@pytest.mark.parametrize("score", [-1.0, 0.0, 1.0])
def test_signal_accepts_scores_in_range(score):
    assert Signal("AAPL", T, score, "v1").score == score


@pytest.mark.parametrize("score", [1.5, -1.01, math.nan])
def test_signal_rejects_scores_out_of_range(score):
    with pytest.raises(ValueError):
        Signal("AAPL", T, score, "v1")


def test_signal_rejects_naive_time():
    with pytest.raises(ValueError, match="UTC"):
        Signal("AAPL", NAIVE, 0.5, "v1")


@pytest.mark.parametrize("quantity", [0, -5, 2.5])
def test_quantity_must_be_positive_whole_shares(quantity):
    with pytest.raises(ValueError):
        make_intent(quantity=quantity)


def test_numpy_integers_count_as_whole_shares():
    assert make_intent(quantity=np.int64(10)).quantity == 10


def test_side_must_be_the_enum_not_a_string():
    with pytest.raises(TypeError):
        make_intent(side="buy")


def test_intent_notional_is_quantity_times_limit_price():
    assert make_intent(quantity=10, limit_price=100.0).notional == 1000.0


def test_order_needs_an_id():
    with pytest.raises(ValueError):
        Order(client_order_id="", intent=make_intent(), sent_at=T)


def test_fill_rejects_negative_commission():
    with pytest.raises(ValueError):
        Fill("q-1", "AAPL", Side.BUY, 10, 100.0, -0.35, T)


def test_daily_plan_weights_are_read_only():
    plan = DailyPlan(PLAN_DAY, "v1", {"AAPL": 0.5, "MSFT": 0.5})
    with pytest.raises(TypeError):
        plan.target_weights["AAPL"] = 0.9


def test_daily_plan_ignores_later_changes_to_the_input_dict():
    weights = {"AAPL": 0.5}
    plan = DailyPlan(PLAN_DAY, "v1", weights)
    weights["AAPL"] = 0.9
    assert plan.target_weights["AAPL"] == 0.5


@pytest.mark.parametrize(
    "weights",
    [
        {"AAPL": -0.1},  # short position
        {"AAPL": 0.7, "MSFT": 0.4},  # leverage: sums to 1.1
    ],
)
def test_daily_plan_is_long_only_and_unlevered(weights):
    with pytest.raises(ValueError):
        DailyPlan(PLAN_DAY, "v1", weights)


def test_daily_plan_uses_default_execution_settings():
    plan = DailyPlan(PLAN_DAY, "v1", {"AAPL": 1.0})
    assert plan.execution == ExecutionSettings()


@pytest.mark.parametrize("band_k", [-1.0, math.nan, math.inf])
def test_the_band_is_a_number_of_days_and_not_negative(band_k):
    with pytest.raises(ValueError, match="band_k"):
        ExecutionSettings(band_k=band_k)


def test_execution_settings_reject_threshold_above_one():
    with pytest.raises(ValueError):
        ExecutionSettings(theta0=1.5)


def test_risk_action_must_be_the_enum_not_a_string():
    with pytest.raises(TypeError):
        RiskEvent(time=T, rule="max_order_usd", action="reject", detail="too big")


def test_a_cancellation_names_its_order_and_a_utc_time():
    assert Cancellation("bt-0000001", T, "session close").reason == "session close"
    with pytest.raises(ValueError):
        Cancellation("", T)
    with pytest.raises(ValueError, match="UTC"):
        Cancellation("bt-0000001", NAIVE)


@pytest.mark.parametrize("ratio", [0.0, -2.0, 1.0, math.nan])
def test_a_split_ratio_must_be_positive_and_not_one(ratio):
    with pytest.raises(ValueError):
        Split("NFLX", date(2025, 11, 17), ratio)


@pytest.mark.parametrize("amount", [0.0, -0.25, math.inf])
def test_a_dividend_must_be_positive(amount):
    with pytest.raises(ValueError):
        Dividend("AAPL", date(2025, 2, 10), amount)


def test_a_plans_volatility_must_be_positive():
    with pytest.raises(ValueError, match="volatility"):
        DailyPlan(PLAN_DAY, "v1", {"AAPL": 0.5}, volatility={"AAPL": 0.0})


def test_a_plan_with_volatility_needs_it_for_every_target_stock():
    with pytest.raises(ValueError, match="no volatility"):
        DailyPlan(PLAN_DAY, "v1", {"AAPL": 0.5, "MSFT": 0.5}, volatility={"AAPL": 0.01})
