import random
from dataclasses import replace
from datetime import UTC, date, timedelta
from pathlib import Path

import polars as pl
import pytest

from qfin_alphaguard.data.eodhd import MINUTE_SCHEMA
from qfin_alphaguard.events import Bar, DecisionNote, Order
from qfin_alphaguard.guard.config import load_guard_config
from qfin_alphaguard.live.engine import Cancel, SessionStart, Stop, Submit
from qfin_alphaguard.live.journal import Entry
from qfin_alphaguard.live.orders import Accepted, Cancelled, FillReport
from qfin_alphaguard.live.report import (
    GATE,
    bars_of,
    chart_series,
    day_report,
    render_html,
)
from qfin_alphaguard.live.runner import make_engine_for
from qfin_alphaguard.live.warmup import sessions_before
from qfin_alphaguard.sessions import sessions

QQQ_GUARD = load_guard_config(Path(__file__).parents[1] / "guard_qqq.yaml")
DAY = date(2026, 10, 19)
SESSION = sessions(DAY, DAY)[0]
START = SessionStart(DAY, SESSION.open - timedelta(minutes=45), 30_000.0, 30_000.0)
SEEN = timedelta(milliseconds=200)


def walk(session, price, rng, drift=0.0):
    bars = []
    minutes = int((session.close - session.open).total_seconds() // 60)
    for minute in range(minutes):
        step = price * (drift + rng.gauss(0, 0.0008))
        open_, close = price, price + step
        high, low = max(open_, close) + 0.03, min(open_, close) - 0.03
        start = session.open + timedelta(minutes=minute)
        bars.append(
            Bar("QQQ", start, 60, open_, high, low, close, rng.randint(200, 3000))
        )
        price = close
    return bars, price


def make_days(seed=11):
    rng = random.Random(seed)
    price, past = 600.0, []
    for session in sessions_before(DAY):
        bars, price = walk(session, price * (1 + rng.gauss(0, 0.006)), rng)
        past.append((session, bars))
    # The day climbs: the strategy goes long above its noise band.
    today, _ = walk(SESSION, price, rng, drift=0.0004)
    return past, today


PAST, TODAY = make_days()


def simulate(bars, past=PAST, start=START, worse=0.0):
    """A session as the runner writes it, with Alpaca played: each order accepted at
    once and filled in full at the next bar's open, `worse` dollars against it."""
    core = make_engine_for(QQQ_GUARD, past)(start)
    entries = [Entry(start.time, start)]
    waiting: list[Order] = []

    def event(e, at):
        entries.append(Entry(at, e))
        take(core.on_order_event(e, at), at)

    def take(step, at):
        for record in step.records:
            entries.append(Entry(at, record))
        for action in step.actions:
            if isinstance(action, Submit):
                waiting.append(action.order)
                event(Accepted(action.order.client_order_id, at), at)
            elif isinstance(action, Cancel):
                event(Cancelled(action.client_order_id, at), at)

    for bar in bars:
        filled_at = bar.end + SEEN / 2
        due, waiting[:] = list(waiting), []
        for order in due:
            if core.orders[order.client_order_id].is_open:
                price = bar.open + (
                    worse if order.intent.side.value == "buy" else -worse
                )
                event(
                    FillReport(
                        order.client_order_id,
                        filled_at,
                        order.intent.quantity,
                        price,
                    ),
                    filled_at,
                )
        at = bar.end + SEEN
        entries.append(Entry(at, bar))
        take(core.on_bars([bar], at), at)
    return entries, core


ENTRIES, CORE = simulate(TODAY)


def test_a_session_that_is_its_backtest_s_meets_the_gate():
    report = day_report(
        ENTRIES, PAST, QQQ_GUARD, {"IEX's history": TODAY, "SIP": TODAY}
    )
    assert report.replay == "the same" and report.leftover == 0
    choices = {n.choice for n in report.notes}
    assert "long" in choices  # the day traded
    for comparison in report.comparisons:
        assert comparison.share == 1.0 >= GATE and comparison.differences == ()
        assert comparison.orders == len(report.orders)
    assert report.bars.live == report.bars.kept == report.bars.same == len(TODAY)
    assert report.end_value == pytest.approx(CORE.portfolio.value(CORE.last_prices))
    assert report.result_usd == pytest.approx(report.end_value - 30_000.0)


def test_slippage_is_measured_against_the_price_decided_on():
    report = day_report(ENTRIES, PAST, QQQ_GUARD, {"IEX's history": TODAY})
    filled = [o for o in report.orders if o.filled > 0]
    assert filled
    for row in filled:
        sign = 1 if row.side == "buy" else -1
        expected = sign * (row.average_price / row.reference_price - 1) * 10_000
        assert row.slippage_bps == pytest.approx(expected)
    # The first order went at the bar it was decided on; it filled at the next bar's open,
    # the close of the bar decided on: no slippage in this played market.
    assert filled[0].slippage_bps == pytest.approx(0.0, abs=1e-9)


def test_fills_against_the_account_cost_positive_basis_points():
    entries, _ = simulate(TODAY, worse=0.06)  # each fill 6 cents worse than the open
    report = day_report(entries, PAST, QQQ_GUARD, {})
    filled = [o for o in report.orders if o.filled > 0]
    assert {o.side for o in filled} == {"buy", "sell"}
    for row in filled:
        assert row.slippage_bps == pytest.approx(0.06 / row.reference_price * 10_000)


def test_a_backtest_on_other_minutes_shows_where_it_decided_otherwise():
    # The kept minutes stop at 10:30, and one bar differs: the backtest decides at 10:00
    # and 10:30 only, and the live session's later decisions have no match.
    kept = [b for b in TODAY if b.end <= SESSION.open + timedelta(minutes=60)]
    kept[5] = replace(kept[5], volume=kept[5].volume + 1)
    report = day_report(ENTRIES, PAST, QQQ_GUARD, {"IEX's history": kept})
    (comparison,) = report.comparisons
    live_minutes = {n.time.replace(second=0, microsecond=0) for n in report.notes}
    assert comparison.total == len(live_minutes)
    assert comparison.matched == 2 and comparison.share < GATE
    assert all(theirs == "-" for _, _, theirs in comparison.differences)
    assert "MISSES the 95% gate" in report.summary()
    b = report.bars
    assert (b.kept, b.same, b.differing) == (60, 59, 1)
    assert b.live_only == len(TODAY) - 60 and b.kept_only == 0


def test_a_journal_that_does_not_replay_says_so():
    tampered = list(ENTRIES)
    i = next(k for k, e in enumerate(tampered) if isinstance(e.record, DecisionNote))
    note = tampered[i].record
    tampered[i] = Entry(tampered[i].at, replace(note, reason="something else"))
    report = day_report(tampered, PAST, QQQ_GUARD, {})
    assert report.replay.startswith("DIFFERS")


def test_a_stop_and_alerts_are_reported():
    stopped = [*ENTRIES, Entry(SESSION.close, Stop(SESSION.close, "a test"))]
    report = day_report(
        stopped, PAST, QQQ_GUARD, {}, alerts=["2026-10-19T20:00:00 stopping"]
    )
    summary = report.summary()
    assert "STOPPED at 16:00:00 New York time: a test" in summary
    assert "Alert: 2026-10-19T20:00:00 stopping" in summary


def test_the_summary_says_it_in_a_few_lines():
    report = day_report(ENTRIES, PAST, QQQ_GUARD, {"IEX's history": TODAY})
    lines = report.summary().splitlines()
    assert (
        lines[0].startswith("Session 2026-10-19: ") and "on 30,000.00 USD" in lines[0]
    )
    assert lines[1] == "Replayed from its journal: the same"
    # The position as it changed, in New York time: long from 10:00, flat at the close.
    assert lines[2].endswith(": 10:00 long, 15:59 flat")
    assert any(
        line.startswith("Backtest on IEX's history:") and "meets" in line
        for line in lines
    )


def test_a_journal_must_start_with_its_session():
    with pytest.raises(ValueError, match="session's start"):
        day_report(ENTRIES[1:], PAST, QQQ_GUARD, {})


def test_the_page_holds_the_verdicts_the_chart_and_the_tables():
    report = day_report(ENTRIES, PAST, QQQ_GUARD, {"IEX's history": TODAY})
    points = chart_series(report, TODAY, PAST, QQQ_GUARD, SESSION)
    assert len(points) == len(TODAY)
    banded = [p for p in points if p["upper"] is not None]
    assert banded and all(p["lower"] < p["upper"] for p in banded)
    page = render_html(report, points, lambda t: t.astimezone(UTC))
    assert "<title>QQQ session 2026-10-19</title>" in page
    assert "Replayed from its journal: the same" in page
    assert page.count("<tr><td>") >= len(report.notes) + len(report.orders)
    script = page[page.index('id="data">') :]
    assert "</script" not in script[: script.index("</script>")].replace("<\\/", "")


def test_the_page_s_data_cannot_close_its_script():
    report = day_report(ENTRIES, PAST, QQQ_GUARD, {})
    hostile = [
        {
            "t": "</script><script>alert(1)</script>",
            "price": 1.0,
            "vwap": None,
            "lower": None,
            "upper": None,
        }
    ]
    page = render_html(report, hostile, lambda t: t.astimezone(UTC))
    assert page.count("</script>") == 2  # the data's and the chart's own, nothing more
    assert "<\\/script><script>alert(1)<\\/script>" in page


def test_bars_come_from_alpaca_s_frames():
    frame = pl.DataFrame(
        [
            (TODAY[1].start, "QQQ", 1.0, 2.0, 0.5, 1.5, 7),
            (TODAY[0].start, "QQQ", 1.0, 2.0, 0.5, 1.5, 9),
        ],
        schema=MINUTE_SCHEMA,
        orient="row",
    )
    bars = bars_of(frame, "QQQ")
    assert [b.start for b in bars] == [TODAY[0].start, TODAY[1].start]
    assert (bars[0].seconds, bars[0].close, bars[0].volume) == (60, 1.5, 9)
