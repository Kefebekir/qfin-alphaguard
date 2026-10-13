from dataclasses import replace
from datetime import date

import polars as pl
import pytest

from qfin_alphaguard.backtest import run as backtest_run
from qfin_alphaguard.backtest.run import (
    Backtest,
    delistings,
    first_day,
    holdable_actions,
    prepare,
    stored_minute_days,
)
from qfin_alphaguard.data.intraday import intraday_path
from qfin_alphaguard.events import Delisting, Dividend, RiskEvent, Split
from qfin_alphaguard.guard.config import GuardConfig
from qfin_alphaguard.sessions import session_days

# guard.yaml's limits today, written out so a hand-made change to the file
# cannot change what these tests mean.
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
D1, D2, D3 = date(2016, 1, 5), date(2016, 1, 6), date(2016, 1, 7)


# --- The days ---------------------------------------------------------------


def test_the_first_day_has_a_year_of_daily_returns_before_it(store):
    _, daily = store
    # Bars from 2 January 2015: 2015 had 252 sessions, so the 252nd return
    # before a day is complete on 5 January 2016 (plan.py's LOOKBACK_DAYS).
    assert first_day(daily) == D1


def test_bars_of_less_than_a_year_and_a_day_have_no_first_day(make_daily):
    days = session_days(date(2015, 1, 2), date(2015, 12, 31))
    with pytest.raises(ValueError, match="a year"):
        first_day(make_daily("A", days, [100.0] * len(days)))


def test_the_last_day_is_the_last_with_minutes(store):
    raw, daily = store
    inputs = prepare(daily, raw)  # the daily bars run one day longer
    assert inputs.days == (D1, D2, D3)
    assert dict(inputs.universes) == {2016: ("A", "B")}
    assert dict(inputs.minute_days) == {"A": {D1, D2, D3}, "B": {D1, D2, D3}}
    assert inputs.actions == ()


def test_the_days_asked_for_are_kept_inside_the_data(store):
    raw, daily = store
    inputs = prepare(daily, raw, first=date(2015, 6, 1), last=date(2016, 1, 6))
    assert inputs.days == (D1, D2)
    with pytest.raises(ValueError, match="no backtest"):
        prepare(daily, raw, first=D3, last=D1)


def test_a_day_needs_its_daily_bars_as_well_as_its_minutes(store):
    # The baselines are measured on the daily closes of the backtest's days.
    raw, daily = store
    inputs = prepare(daily.filter(pl.col("date") < D3), raw, last=D3)
    assert inputs.days == (D1, D2)


def test_without_stored_minutes_there_is_no_backtest(store, tmp_path):
    _, daily = store
    with pytest.raises(ValueError, match="no stored minutes"):
        prepare(daily, tmp_path / "empty")


# --- Minutes and delistings ---------------------------------------------------


def test_minute_days_are_read_as_the_feed_reads_them(store):
    raw, _ = store
    # A until 6 January, as for a stock that left the universe; B not needed.
    found = stored_minute_days(raw, {2016: {"A": D2}}, D1, D3)
    assert found == {"A": {D1, D2}}
    assert stored_minute_days(raw, {2016: {"A": D3}}, D2, D2) == {"A": {D2}}
    assert stored_minute_days(raw, {2016: {"C": D3}}, D1, D3) == {}


def test_a_stock_whose_minutes_end_is_delisted_on_the_next_session():
    days = (D1, D2, D3)
    minute_days = {"A": frozenset({D1, D2}), "B": frozenset(days)}
    assert delistings(minute_days, days) == [Delisting("A", D3)]


def test_a_stock_with_minutes_to_the_end_is_not_delisted():
    days = (D1, D2, D3)
    assert delistings({"B": frozenset(days)}, days) == []


def test_a_stock_is_not_delisted_before_its_minutes_begin():
    days = (D1, D2, D3)
    assert delistings({"NEW": frozenset({D3})}, days) == []


def test_a_gap_in_a_stock_s_minutes_pays_out_its_shares_on_the_first_day_missing():
    # A held stock without minutes can be neither sold nor priced afresh.
    days = (D1, D2, D3)
    assert delistings({"C": frozenset({D1, D3})}, days) == [Delisting("C", D2)]


# --- Splits and dividends -----------------------------------------------------


def split_and_odd_step(make_daily):
    """S splits 2 for 1 on 8 January 2016, and on 5 January its dividend
    factor falls: neither a split nor a dividend, as in EODHD's NVDA of 2015."""
    days = session_days(date(2016, 1, 4), date(2016, 1, 15))
    bars = make_daily("S", days, [100.0] * len(days))
    split_factor = [2.0 if day < date(2016, 1, 8) else 1.0 for day in days]
    adjustment = [
        (0.99 if day >= date(2016, 1, 5) else 1.0) / factor
        for day, factor in zip(days, split_factor, strict=True)
    ]
    return bars.with_columns(
        pl.Series("split_factor", split_factor), pl.Series("adjustment", adjustment)
    )


def test_only_the_days_a_stock_can_be_held_are_read_for_its_actions(make_daily):
    daily = split_and_odd_step(make_daily)
    windows = {2016: {"S": date(2016, 12, 31)}}
    # From 7 January, the odd step of 5 January is out of reach; the split is not.
    actions = holdable_actions(daily, windows, date(2016, 1, 7), date(2016, 1, 15))
    assert actions == [Split("S", date(2016, 1, 8), 2.0)]


def test_an_odd_step_on_a_day_a_stock_can_be_held_stops_the_backtest(make_daily):
    daily = split_and_odd_step(make_daily)
    windows = {2016: {"S": date(2016, 12, 31)}}
    with pytest.raises(ValueError, match="neither split nor dividend"):
        holdable_actions(daily, windows, date(2016, 1, 4), date(2016, 1, 15))


def test_an_action_on_the_first_day_is_seen(make_daily):
    # The session before the first day is read too, for the step into it.
    daily = split_and_odd_step(make_daily)
    windows = {2016: {"S": date(2016, 12, 31)}}
    actions = holdable_actions(daily, windows, date(2016, 1, 8), date(2016, 1, 15))
    assert actions == [Split("S", date(2016, 1, 8), 2.0)]


def test_a_stock_is_read_only_until_the_end_of_its_window(make_daily):
    daily = split_and_odd_step(make_daily)
    windows = {2016: {"S": date(2016, 1, 7)}}  # left before the split
    assert holdable_actions(daily, windows, date(2016, 1, 6), date(2016, 1, 15)) == []


def test_a_dividend_on_the_day_of_a_delisting_is_paid_before_the_payout(store):
    # A goes ex-dividend on 7 January, the first day without its minutes.
    raw, daily = store
    a_minutes = pl.read_parquet(intraday_path(raw, "A", 2016))
    a_minutes.filter(pl.col("start").dt.date() < D3).write_parquet(
        intraday_path(raw, "A", 2016)
    )
    dividend_day = pl.col("date") >= D3
    paid = daily.with_columns(
        pl.when((pl.col("ticker") == "A") & dividend_day)
        .then(pl.col("adjustment") * 1.01)
        .otherwise(pl.col("adjustment"))
        .alias("adjustment")
    )

    actions = prepare(paid, raw, last=D3).actions

    assert [type(action) for action in actions] == [Dividend, Delisting]
    assert {action.day for action in actions} == {D3}


# --- Runs ---------------------------------------------------------------------


def test_plans_are_built_once_for_capitals_that_hold_as_many_stocks(store, monkeypatch):
    raw, daily = store
    built = []

    def counting(prices, days, capital, max_weight):
        built.append(capital)
        return {}

    monkeypatch.setattr(backtest_run, "build_plans", counting)
    test = Backtest(daily, raw, LIMITS)
    test.plans(500.0)
    test.plans(30_000.0)  # past the universe's 50 stocks: the same plans
    test.plans(9.0)  # three stocks at most
    assert built == [500.0, 9.0]


def test_a_run_trades_every_day_through_guard_and_closes_each_session(store):
    raw, daily = store
    test = Backtest(daily, raw, LIMITS)

    result = test.run(1_000.0)

    assert [close.day for close in result.closes] == [D1, D2, D3]
    assert set(result.positions) == {"A", "B"}
    assert not any(isinstance(record, RiskEvent) for record in result.log)


def test_every_order_of_a_run_goes_through_guard(store):
    # With orders of at most 50 USD, Guard cuts the children of 1,000 USD.
    raw, daily = store
    test = Backtest(daily, raw, replace(LIMITS, max_order_usd=50.0))

    result = test.run(1_000.0)

    events = [record for record in result.log if isinstance(record, RiskEvent)]
    assert events
    assert {event.rule for event in events} == {"max_order_usd"}


def test_the_same_inputs_give_the_same_run(store):
    raw, daily = store
    first = Backtest(daily, raw, LIMITS).run(1_000.0)
    again = Backtest(daily, raw, LIMITS).run(1_000.0)
    other = Backtest(daily, raw, LIMITS).run(2_000.0)
    assert first.fingerprint() == again.fingerprint()
    assert first.fingerprint() != other.fingerprint()
