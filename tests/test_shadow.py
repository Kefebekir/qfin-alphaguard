import math
import statistics
from datetime import UTC, date, datetime, timedelta

import pytest

from qfin_alphaguard.backtest.broker import CostModel, NextBarBroker
from qfin_alphaguard.backtest.engine import BacktestResult, SessionClose, run_backtest
from qfin_alphaguard.backtest.shadow import (
    daily_returns,
    describe,
    shadow_record,
    summarise,
)
from qfin_alphaguard.challengers import ChallengerCard
from qfin_alphaguard.events import Bar
from qfin_alphaguard.guard.check import GuardDecision
from qfin_alphaguard.momentum import MomentumSettings, NoiseAreaMomentum
from qfin_alphaguard.sessions import sessions

CARD = ChallengerCard(
    name="half-size",
    description="The rule at half its size, to check the record.",
    owner="Efe",
    code_commit="b" * 40,
    weights="data/models/none.npz",
    weights_sha256="0" * 64,
    trained_from=date(2026, 1, 2),
    trained_to=date(2026, 2, 27),
    frozen_at=datetime(2026, 2, 28, 12, 0, tzinfo=UTC),
    record_from=date(2026, 3, 2),
    scale=2.0,
    clip=0.05,
    alpha=0.025,
    retired_on=None,
)


def closes(values, first=date(2026, 3, 2)):
    return BacktestResult(
        log=(),
        cash=0.0,
        positions={},
        closes=tuple(
            SessionClose(
                first + timedelta(days=i), datetime(2026, 3, 2, 21, tzinfo=UTC), 0.0, v
            )
            for i, v in enumerate(values)
        ),
    )


def test_a_day_s_return_is_its_close_over_the_one_before():
    r = daily_returns(closes([101.0, 99.99]), cash=100.0)
    assert list(r.values()) == pytest.approx([0.01, -0.01])


def test_an_account_worth_nothing_has_no_return():
    with pytest.raises(ValueError, match="worth"):
        daily_returns(closes([0.0, 1.0]), cash=100.0)


def test_only_days_from_the_record_s_start_that_both_accounts_have_count():
    d = [date(2026, 2, 27), date(2026, 3, 2), date(2026, 3, 3), date(2026, 3, 4)]
    challenger = {d[0]: 0.5, d[1]: 0.01, d[3]: 0.02}
    rule = {d[0]: -0.5, d[1]: 0.0, d[2]: 0.01, d[3]: 0.01}
    record = shadow_record(CARD, challenger, rule)
    assert record.days == (d[1], d[3])  # not before the start; not a day one lacks
    assert record.evidence.outcomes == pytest.approx(
        (0.4, 0.6)
    )  # (2 x 0.01 - 0) / 0.05 ...


def test_a_retired_challenger_s_record_ends_on_its_retirement():
    retired = ChallengerCard(**{**CARD.__dict__, "retired_on": date(2026, 3, 3)})
    days = [date(2026, 3, 2), date(2026, 3, 3), date(2026, 3, 4)]
    record = shadow_record(retired, dict.fromkeys(days, 0.01), dict.fromkeys(days, 0.0))
    assert record.days == tuple(days[:2])


# --- the raw results, shown beside the test ------------------------------------------


def test_a_summary_is_its_days_as_a_year():
    s = summarise([0.01, -0.01, 0.02])
    assert s.days == 3
    assert s.annual_return == pytest.approx(0.02 / 3 * 252)
    assert s.annual_volatility == pytest.approx(
        statistics.stdev([0.01, -0.01, 0.02]) * math.sqrt(252)
    )
    assert s.total_return == pytest.approx(1.01 * 0.99 * 1.02 - 1)
    assert s.sharpe == pytest.approx(s.annual_return / s.annual_volatility)


def test_a_summary_needs_two_days_and_a_flat_account_has_no_sharpe():
    with pytest.raises(ValueError, match="2 days"):
        summarise([0.01])
    assert math.isnan(summarise([0.0, 0.0]).sharpe)


DAYS3 = [date(2026, 3, 2), date(2026, 3, 3), date(2026, 3, 4)]
RULE3 = dict(zip(DAYS3, [0.01, -0.004, 0.006], strict=True))


def test_the_raw_results_stand_beside_those_at_the_rule_s_risk():
    half = {
        d: r / 2 for d, r in RULE3.items()
    }  # the rule at half size; the card's scale is 2
    record = shadow_record(CARD, half, RULE3)
    summaries = record.summaries()
    rule, raw = summaries["the rule"], summaries["the challenger, raw"]
    equal = summaries["the challenger at the rule's risk"]
    assert raw.annual_return == pytest.approx(rule.annual_return / 2)
    assert equal.annual_return == pytest.approx(rule.annual_return)
    assert equal.annual_volatility == pytest.approx(rule.annual_volatility)
    # raw: half the rule's mean day, (0.01 - 0.004 + 0.006) / 3 = 0.004, less the rule's
    assert record.mean_difference(equal_risk=False) == pytest.approx(-0.002)
    assert record.mean_difference(equal_risk=True) == pytest.approx(0.0)


def test_the_description_shows_raw_beside_the_test():
    """The rule at double size: ahead raw, level at the rule's risk, and not promoted."""
    double = ChallengerCard(**{**CARD.__dict__, "scale": 0.5})
    text = describe(shadow_record(double, {d: 2 * r for d, r in RULE3.items()}, RULE3))
    assert "2026-03-02 to 2026-03-04, 3 days" in text
    assert "raw +40.00 bps, at the rule's risk (x 0.5) +0.00 bps" in text
    assert "not promoted" in text
    assert "days beyond the clip: none" in text
    lines = text.splitlines()
    assert [line.split()[1] for line in lines[2:5]] == [
        "rule",
        "challenger,",
        "challenger",
    ]


def test_a_record_too_short_to_describe_says_so():
    record = shadow_record(CARD, {DAYS3[0]: 0.01}, {DAYS3[0]: 0.0})
    assert describe(record) == "1 days: too few to describe"


# --- the rule and a challenger, each in its own account in the engine -------------------

DAYS = sessions(date(2026, 3, 2), date(2026, 4, 30))


class AllowAll:
    def check(self, intent, view):
        return GuardDecision(intent)


def bars(session, path):
    minutes = int((session.close - session.open).total_seconds() // 60)
    last = path(0)
    for minute in range(minutes):
        close = path(minute)
        yield Bar(
            "QQQ",
            session.open + timedelta(minutes=minute),
            60,
            last,
            max(last, close),
            min(last, close),
            close,
            10_000,
        )
        last = close


def feed():
    for number, session in enumerate(DAYS):
        if (
            number >= 15 and number % 3 == 0
        ):  # a trend up every third day once there is history
            yield from bars(session, lambda m: 100.0 * (1 + 0.0002 * m))
        else:
            level = 100.0 * (1 + 0.002 * (-1) ** number)
            yield from bars(
                session, lambda m, level=level: level * (1 + 0.001 * math.sin(m / 30))
            )


def test_a_challenger_that_is_the_rule_at_half_size_earns_no_evidence():
    cash = 30_000.0
    costs = CostModel(half_spread_bps=0.3)
    rule = run_backtest(
        feed(), NoiseAreaMomentum(), NextBarBroker(costs), AllowAll(), cash
    )
    # The same choices at half the size: min(1, 1% / vol) is half of min(2, 2% / vol).
    half = NoiseAreaMomentum(MomentumSettings(cap=1.0, target_vol=0.01))
    challenger = run_backtest(feed(), half, NextBarBroker(costs), AllowAll(), cash)
    r, c = daily_returns(rule, cash), daily_returns(challenger, cash)
    assert sum(1 for v in r.values() if v) >= 5  # the rule traded
    record = shadow_record(CARD, c, r)
    assert len(record.days) == len(DAYS)
    assert max(abs(x) for x in record.evidence.outcomes) < 0.01  # equal at equal risk
    assert not record.evidence.promoted
    assert max(record.evidence.wealth) < 1.05
