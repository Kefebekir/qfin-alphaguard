import math
import statistics
from datetime import date, timedelta
from itertools import pairwise

import pytest

from qfin_alphaguard.backtest.broker import CostModel, NextBarBroker
from qfin_alphaguard.backtest.engine import SessionClose, run_backtest
from qfin_alphaguard.events import (
    Bar,
    Cancellation,
    DecisionNote,
    Fill,
    OrderIntent,
    Side,
)
from qfin_alphaguard.guard.check import GuardDecision
from qfin_alphaguard.momentum import (
    MomentumSettings,
    NoiseAreaMomentum,
    daily_volatility,
    describe,
    next_position,
    noise_band,
    target_shares,
)
from qfin_alphaguard.sessions import sessions

NAN = math.nan

# The rules, one at a time, against numbers worked out by hand.


def test_noise_band_averages_the_past_moves_at_the_minute():
    past = [[0.0, 0.01, 0.02], [0.0, 0.03, 0.04]]
    lower, upper = noise_band(past, 1, day_open=100.0, prev_close=101.0)
    # sigma = (0.01 + 0.03) / 2 = 0.02, around min 100 and max 101
    assert lower == pytest.approx(100.0 * 0.98)
    assert upper == pytest.approx(101.0 * 1.02)


def test_noise_band_is_around_the_lower_and_higher_of_open_and_close():
    past = [[0.0, 0.01]]
    gap_up = noise_band(past, 1, day_open=105.0, prev_close=100.0)
    gap_down = noise_band(past, 1, day_open=95.0, prev_close=100.0)
    assert gap_up == pytest.approx((100.0 * 0.99, 105.0 * 1.01))
    assert gap_down == pytest.approx((95.0 * 0.99, 100.0 * 1.01))


def test_noise_band_leaves_out_sessions_that_end_before_the_minute():
    # The first past session is a half day: it has no move at minute 2.
    past = [[0.0, 0.01], [0.0, 0.03, 0.05], [0.0, 0.02, 0.07]]
    lower, upper = noise_band(past, 2, day_open=100.0, prev_close=100.0)
    assert lower == pytest.approx(100.0 * (1 - 0.06))
    assert upper == pytest.approx(100.0 * (1 + 0.06))


def test_a_minute_no_past_session_reaches_has_no_band():
    lower, upper = noise_band([[0.0, 0.01], [0.0, 0.02]], 5, 100.0, 100.0)
    assert math.isnan(lower) and math.isnan(upper)


def test_a_nan_move_makes_both_edges_nan():
    lower, upper = noise_band([[NAN, 0.01], [0.0, 0.03]], 0, 100.0, 100.0)
    assert math.isnan(lower) and math.isnan(upper)


def test_daily_volatility_is_the_sample_deviation_of_log_returns():
    closes = [100.0, 101.0, 99.0, 102.0, 102.5]
    returns = [math.log(b / a) for a, b in pairwise(closes)]
    assert daily_volatility(closes) == pytest.approx(statistics.stdev(returns))


def test_daily_volatility_of_steady_closes_is_zero():
    assert daily_volatility([100.0, 100.0, 100.0]) == 0.0


@pytest.mark.parametrize(
    ("held", "price", "lower", "upper", "vwap", "expected"),
    [
        # flat
        (0, 103.0, 98.0, 102.0, 100.0, 1),  # above the band: long
        (0, 97.0, 98.0, 102.0, 100.0, -1),  # below it: short
        (0, 101.0, 98.0, 102.0, 100.0, 0),  # inside it: flat
        (0, 102.0, 98.0, 102.0, 100.0, 0),  # on the edge is inside
        # long: its stop is the higher of the band's top and VWAP
        (1, 103.0, 98.0, 102.0, 100.0, 1),  # above both: keep it
        (1, 103.0, 98.0, 102.0, 104.0, 0),  # below VWAP: stopped, flat
        (1, 101.0, 98.0, 102.0, 100.0, 0),  # back in the band: flat
        (1, 97.0, 98.0, 102.0, 100.0, -1),  # through the band: short
        (1, 102.0, 98.0, 102.0, 100.0, 1),  # on the stop: kept
        # short: its stop is the lower of the band's bottom and VWAP
        (-1, 97.0, 98.0, 102.0, 100.0, -1),  # below both: keep it
        (-1, 97.0, 98.0, 102.0, 96.0, 0),  # above VWAP: stopped, flat
        (-1, 99.0, 98.0, 102.0, 100.0, 0),  # back in the band: flat
        (-1, 103.0, 98.0, 102.0, 100.0, 1),  # through the band: long
        (-1, 98.0, 98.0, 102.0, 100.0, -1),  # on the stop: kept
    ],
)
def test_next_position(held, price, lower, upper, vwap, expected):
    assert next_position(held, price, lower, upper, vwap) == expected


@pytest.mark.parametrize("held", [-1, 0, 1])
@pytest.mark.parametrize("vwap", [99.0, 101.0])
def test_a_nan_band_changes_nothing_as_in_the_research(held, vwap):
    # max(nan, vwap) is nan in this order, and every comparison with nan is
    # false: whichever side of VWAP the price is on, nothing changes.
    assert next_position(held, 100.0, NAN, NAN, vwap) == held


def test_target_shares_long_is_fractional_and_rounded_down():
    # size = min(2, 0.02 / 0.01) = 2; 2 * 30,000 / 700 = 85.714285714285...
    assert target_shares(1, 30_000.0, 700.0, 0.01, 2.0, 0.02) == 85.714285714


def test_target_shares_long_rounds_down_not_to_the_nearest():
    # 2,000 / 3 = 666.66666666666...: the nearest 9 decimals would be ...667,
    # a fraction more than the size allows.
    assert target_shares(1, 2_000.0, 3.0, 0.02, 1.0, 0.02) == 666.666666666


def test_target_shares_short_is_whole_shares_rounded_down():
    assert target_shares(-1, 30_000.0, 700.0, 0.01, 2.0, 0.02) == -85.0


def test_target_shares_takes_the_lower_of_the_cap_and_the_volatility_target():
    calm = target_shares(1, 10_000.0, 100.0, 0.005, 2.0, 0.02)  # 4x wanted, 2x cap
    wild = target_shares(1, 10_000.0, 100.0, 0.04, 2.0, 0.02)  # 0.5x
    assert calm == 200.0
    assert wild == 50.0


def test_target_shares_flat_is_none():
    assert target_shares(0, 30_000.0, 700.0, 0.01, 2.0, 0.02) == 0.0


def test_a_small_account_holds_a_fraction_long_and_nothing_short():
    # 500 USD at 1x in a 700 USD share: under one share.
    assert target_shares(1, 500.0, 700.0, 0.02, 1.0, 0.02) == 0.714285714
    assert target_shares(-1, 500.0, 700.0, 0.02, 1.0, 0.02) == 0.0


# The strategy in the engine, on made-up sessions.

DAYS = sessions(date(2026, 3, 2), date(2026, 4, 30))  # no half day among them
CASH = 30_000.0


class AllowAll:
    """Guard stand-in that sends every intent: step 4 sets the real limits."""

    def check(self, intent, view):
        return GuardDecision(intent)


def calm(day_number, minute):
    """A quiet session: within 0.1% of its open, which is a little up or down."""
    level = 100.0 * (1 + 0.002 * (-1) ** day_number)
    return level * (1 + 0.001 * math.sin(minute / 30))


def session_bars(session, price, ticker="QQQ"):
    """One bar a minute; each opens at the last close, so prices are smooth."""
    minutes = int((session.close - session.open).total_seconds() // 60)
    last = price(0)
    for minute in range(minutes):
        close = price(minute)
        yield Bar(
            ticker,
            session.open + timedelta(minutes=minute),
            60,
            open=last,
            high=max(last, close),
            low=min(last, close),
            close=close,
            volume=10_000,
        )
        last = close


def run(paths, settings=None, broker=None):
    """The strategy over DAYS: `paths` maps a day's number to its price path."""

    def feed():
        for number, session in enumerate(DAYS[: max(paths) + 1]):
            path = paths.get(number, lambda minute, n=number: calm(n, minute))
            yield from session_bars(session, path)

    strategy = NoiseAreaMomentum(settings)
    broker = broker or NextBarBroker(CostModel(half_spread_bps=0.0))
    return run_backtest(feed(), strategy, broker, AllowAll(), CASH), strategy


def trend(start, slope):
    """A day that opens at `start` and moves `slope` of it every minute."""
    return lambda minute: start * (1 + slope * minute)


def intents(result, day):
    return [
        record
        for record in result.log
        if isinstance(record, OrderIntent) and record.time.date() == day
    ]


def fills(result, day):
    return [
        record
        for record in result.log
        if isinstance(record, Fill) and record.time.date() == day
    ]


def ny_minutes(session, time):
    return int((time - session.open).total_seconds() // 60)


WARM = 15  # sessions before the first that can trade: 15 closes, 14 moves


def test_no_trade_before_fifteen_sessions_of_history():
    # The 15th session (number 14) trends hard, but the strategy cannot know
    # its band or volatility yet.
    result, _ = run({14: trend(100.0, 0.0002)})
    assert not any(isinstance(record, OrderIntent) for record in result.log)


def test_a_trend_up_buys_at_ten_and_closes_a_minute_before_the_close():
    session = DAYS[WARM]
    result, strategy = run({WARM: trend(100.0, 0.0002)})
    sent = intents(result, session.day)
    assert [ny_minutes(session, intent.time) for intent in sent] == [30, 389]
    buy, sell = sent
    assert (buy.side, sell.side) == (Side.BUY, Side.SELL)
    assert buy.quantity == sell.quantity  # flat at the end of the day
    # 2x the account at 10:00's price, as the cap binds on calm history
    price = trend(100.0, 0.0002)(29)
    assert buy.quantity == pytest.approx(2 * CASH / price, abs=1e-9)
    assert result.positions == {}
    assert strategy.held == 0


def test_orders_fill_on_the_bar_after_the_decision():
    session = DAYS[WARM]
    result, _ = run({WARM: trend(100.0, 0.0002)})
    times = [ny_minutes(session, fill.time) for fill in fills(result, session.day)]
    assert times == [30, 389]  # the 10:00 bar's open, and the last bar's open


def test_a_trend_down_sells_short_whole_shares():
    session = DAYS[WARM]
    result, _ = run({WARM: trend(100.0, -0.0002)})
    short, cover = intents(result, session.day)
    assert (short.side, cover.side) == (Side.SELL, Side.BUY)
    assert short.quantity == math.floor(2 * CASH / trend(100.0, -0.0002)(29))


def test_long_only_stays_out_of_a_trend_down():
    settings = MomentumSettings(long_only=True)
    result, _ = run({WARM: trend(100.0, -0.0002)}, settings)
    assert intents(result, DAYS[WARM].day) == []


def test_a_quiet_day_has_no_trade():
    result, _ = run({WARM + 1: lambda minute: calm(0, minute)})
    assert intents(result, DAYS[WARM + 1].day) == []


def test_decisions_come_only_every_half_hour():
    # Up for the first hour and a half, then down hard: the long is stopped
    # and turned short only at a half hour.
    def path(minute):
        top = 100.0 * (1 + 0.0002 * 90)
        if minute <= 90:
            return 100.0 * (1 + 0.0002 * minute)
        return top * (1 - 0.0004 * (minute - 90))

    session = DAYS[WARM]
    result, _ = run({WARM: path})
    times = [
        ny_minutes(session, intent.time) for intent in intents(result, session.day)
    ]
    flips = [t for t in times if t != 389 and t % 30 != 0]
    # The only order off a half hour opens the short after the long's close.
    assert all(
        intent.reason == "decision"
        for intent in intents(result, session.day)
        if ny_minutes(session, intent.time) in flips
    )
    assert all(t % 30 == 0 or t == 389 or t - 1 in times for t in times)


def test_turning_from_long_to_short_closes_first_then_opens():
    def path(minute):
        if minute <= 60:
            return 100.0 * (1 + 0.0003 * minute)
        top = 100.0 * (1 + 0.0003 * 60)
        return top * (1 - 0.001 * (minute - 60))

    session = DAYS[WARM]
    result, _ = run({WARM: path})
    sent = intents(result, session.day)
    reasons = [(ny_minutes(session, i.time), i.side, i.reason) for i in sent]
    closing = [r for r in reasons if r[2] == "close before turning"]
    assert closing, reasons
    minute = closing[0][0]
    # the close goes at a half hour, the short on the next bar after its fill
    assert minute % 30 == 0
    following = [r for r in reasons if r[0] == minute + 1]
    assert following and following[0][1] is Side.SELL
    # No order ever crosses zero: each fill leaves the position on one side.
    position = 0.0
    for fill in fills(result, session.day):
        before = position
        position += fill.quantity if fill.side is Side.BUY else -fill.quantity
        assert before * position >= 0 or abs(position) < 1e-9


def test_every_order_is_inside_guards_collar():
    session = DAYS[WARM]
    result, _ = run({WARM: trend(100.0, 0.0002)})
    for intent in intents(result, session.day):
        last = trend(100.0, 0.0002)(ny_minutes(session, intent.time) - 1)
        assert abs(intent.limit_price / last - 1) <= 0.005 + 0.0001


def test_the_account_value_at_the_open_and_the_volatility_size_the_day():
    # After a day up 8%, the band is wider and the volatility higher: the
    # next day needs a steeper trend to break out, and gets a smaller size.
    session = DAYS[WARM + 1]
    steep = trend(100.0, 0.001)
    result, strategy = run({WARM: trend(100.0, 0.0002), WARM + 1: steep})
    closes = [r for r in result.log if isinstance(r, SessionClose)]
    value_before = closes[WARM].value
    buy = intents(result, session.day)[0]
    size = min(2.0, 0.02 / strategy.volatility)
    assert size < 2.0  # the volatility, not the cap, sets it
    price = steep(ny_minutes(session, buy.time) - 1)  # the bar it was decided on
    expected = math.floor(round(size * value_before / price * 1e9, 3)) / 1e9
    assert buy.quantity == expected
    assert value_before > CASH  # the first day's gain is in it


def test_an_unfilled_order_is_cancelled_and_sent_again():
    class FillsLate(NextBarBroker):
        """Lets the first minute after each order pass without a fill."""

        def __init__(self):
            super().__init__(CostModel(half_spread_bps=0.0))
            self.skipped = set()

        def on_bar(self, bar):
            waiting = [o for o in self.waiting.values() if o.sent_at <= bar.start]
            new = [o for o in waiting if o.client_order_id not in self.skipped]
            if new:
                self.skipped.update(o.client_order_id for o in new)
                return []
            return super().on_bar(bar)

    session = DAYS[WARM]
    result, _ = run({WARM: trend(100.0, 0.0002)}, broker=FillsLate())
    cancelled = [
        r
        for r in result.log
        if isinstance(r, Cancellation) and r.time.date() == session.day
    ]
    assert cancelled
    assert result.positions == {}


def test_settings_are_checked():
    with pytest.raises(ValueError, match="cap"):
        MomentumSettings(cap=0)
    with pytest.raises(ValueError, match="lookback"):
        MomentumSettings(lookback=1)


# The decision log: a note for every decision, and for the day's close.


def notes(result, day):
    return [
        record
        for record in result.log
        if isinstance(record, DecisionNote) and record.time.date() == day
    ]


def test_a_note_for_every_half_hour_and_the_close():
    session = DAYS[WARM]
    result, _ = run({WARM: trend(100.0, 0.0002)})
    day = notes(result, session.day)
    assert [ny_minutes(session, n.time) for n in day] == [*range(30, 361, 30), 389]
    first, *kept, last = day
    assert (first.choice, first.reason) == ("long", "above the band: buy")
    assert all(
        n.choice == "long" and n.reason.startswith("the trend holds") for n in kept
    )
    assert (last.choice, last.reason) == ("flat", "the day's end: close the position")


def test_a_note_holds_what_the_decision_saw():
    session = DAYS[WARM]
    path = trend(100.0, 0.0002)
    result, _ = run({WARM: path})
    first = notes(result, session.day)[0]
    buy = intents(result, session.day)[0]
    assert first.seen["price"] == path(29)  # the bar that ended at 10:00
    assert first.seen["shares"] == buy.quantity
    assert first.seen["lower"] < first.seen["upper"] < first.seen["price"]
    assert first.seen["account"] == CASH
    assert first.seen["volatility"] > 0
    # the kept long's stop is the higher of the band's top and VWAP
    kept = notes(result, session.day)[1]
    stop = max(kept.seen["upper"], kept.seen["vwap"])
    assert kept.reason == f"the trend holds; its stop is at {stop:.2f}"


def test_the_note_comes_before_the_order_it_explains():
    session = DAYS[WARM]
    result, _ = run({WARM: trend(100.0, 0.0002)})
    day = [
        r
        for r in result.log
        if isinstance(r, DecisionNote | OrderIntent) and r.time.date() == session.day
    ]
    assert isinstance(day[0], DecisionNote) and isinstance(day[1], OrderIntent)


def test_no_note_before_there_is_history():
    result, _ = run({14: trend(100.0, 0.0002)})
    assert not any(isinstance(record, DecisionNote) for record in result.log)


def test_a_quiet_day_notes_that_it_stays_flat():
    session = DAYS[WARM + 1]
    result, _ = run({WARM + 1: lambda minute: calm(0, minute)})
    *half_hours, last = notes(result, session.day)
    assert len(half_hours) == 12
    assert {n.reason for n in half_hours} == {"inside the noise band: stay flat"}
    assert (last.choice, last.reason) == ("flat", "the day's end: flat")


def test_long_only_notes_why_it_does_not_short():
    session = DAYS[WARM]
    result, _ = run({WARM: trend(100.0, -0.0002)}, MomentumSettings(long_only=True))
    reasons = {n.reason for n in notes(result, session.day)}
    assert "below the band, but long only: stay flat" in reasons


def test_a_turn_is_noted_once_the_rules_turn():
    def path(minute):
        if minute <= 60:
            return 100.0 * (1 + 0.0003 * minute)
        top = 100.0 * (1 + 0.0003 * 60)
        return top * (1 - 0.001 * (minute - 60))

    session = DAYS[WARM]
    result, _ = run({WARM: path})
    reasons = [(n.choice, n.reason) for n in notes(result, session.day)]
    assert ("short", "long stopped below the band: sell short") in reasons


@pytest.mark.parametrize(
    ("before", "wanted", "after", "price", "vwap", "expected"),
    [
        (0, 1, 1, 103.0, 100.0, "above the band: buy"),
        (0, -1, -1, 97.0, 100.0, "below the band: sell short"),
        (0, 0, 0, 100.0, 100.0, "inside the noise band: stay flat"),
        (1, 1, 1, 103.0, 100.0, "the trend holds; its stop is at 102.00"),
        (1, 1, 1, 105.0, 104.0, "the trend holds; its stop is at 104.00"),
        (-1, -1, -1, 97.0, 100.0, "the trend holds; its stop is at 98.00"),
        (-1, -1, -1, 95.0, 96.0, "the trend holds; its stop is at 96.00"),
        (
            1,
            0,
            0,
            101.0,
            100.0,
            "long stopped: price fell below the band's top or VWAP",
        ),
        (
            -1,
            0,
            0,
            99.0,
            100.0,
            "short stopped: price rose above the band's bottom or VWAP",
        ),
        (1, -1, -1, 97.0, 100.0, "long stopped below the band: sell short"),
        (-1, 1, 1, 103.0, 100.0, "short stopped above the band: buy"),
        (0, -1, 0, 97.0, 100.0, "below the band, but long only: stay flat"),
    ],
)
def test_describe(before, wanted, after, price, vwap, expected):
    assert describe(before, wanted, after, price, 98.0, 102.0, vwap) == expected


def test_describe_a_missing_band():
    reason = describe(1, 1, 1, 100.0, NAN, NAN, 99.0)
    assert reason == "no noise band at this minute (a past session lacks it): no change"
