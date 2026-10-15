import math
import statistics

import numpy as np
import pytest

from qfin_alphaguard.evidence import (
    BettingTest,
    betting_wealth,
    round_up,
    training_settings,
)


def test_a_day_s_outcome_is_the_clipped_difference_at_equal_risk():
    test = BettingTest(scale=2.0, clip=0.05)
    x = test.outcomes([0.01, 0.10, -0.10], [0.01, 0.0, 0.0])
    # 2 x 0.01 - 0.01 = 0.01 -> 0.2; then clipped at +-5%
    assert x.tolist() == pytest.approx([0.2, 1.0, -1.0])


def test_wealth_by_hand_for_three_days():
    x = np.array([0.2, -0.2, 0.4])
    bets, wealth = betting_wealth(x, max_bet=0.9)
    # day 1: no history, mean 0 -> no bet
    assert bets[0] == 0.0 and wealth[0] == 1.0
    # day 2: mean (0 + 0.2) / 2 = 0.1; squared surprises 0.25 (the prior) + 0.2^2,
    # over 2 -> 0.145; bet 0.1 / (0.145 + 0.01), below the cap of 0.9
    assert bets[1] == pytest.approx(0.1 / (0.145 + 0.1**2))
    assert wealth[1] == pytest.approx(1 + bets[1] * -0.2)
    # day 3: mean (0.2 - 0.2) / 3 = 0 -> no bet; wealth unchanged
    assert bets[2] == 0.0
    assert wealth[2] == pytest.approx(wealth[1])


def test_a_bet_uses_only_the_days_before_it():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, 50)
    bets, _ = betting_wealth(x)
    changed = x.copy()
    changed[30:] = -changed[30:]  # change day 30 and after
    bets_changed, _ = betting_wealth(changed)
    assert np.array_equal(bets[:31], bets_changed[:31])


def test_a_challenger_that_matches_the_rule_gains_no_evidence():
    test = BettingTest(scale=2.0, clip=0.05)
    rule = [0.01, -0.02, 0.003, 0.0, 0.015]
    challenger = [r / 2 for r in rule]  # the rule at half size, at equal risk the same
    evidence = test.run(challenger, rule)
    assert set(evidence.wealth) == {1.0}
    assert not evidence.promoted


def test_a_bet_against_the_challenger_is_never_placed():
    _, wealth = betting_wealth(np.full(100, -0.3))
    assert (wealth == 1.0).all()  # it never bets on a challenger that keeps losing


def test_wealth_can_lose_at_most_max_bet_in_a_day():
    x = np.concatenate([np.full(20, 1.0), [-1.0]])
    bets, wealth = betting_wealth(x, max_bet=0.5)
    assert bets.max() <= 0.5
    assert wealth[-1] / wealth[-2] >= 0.5


def test_promotion_is_the_first_day_wealth_reaches_one_over_alpha():
    test = BettingTest(scale=1.0, clip=0.05, alpha=0.025)
    assert test.threshold == 40
    evidence = test.run([0.04] * 200, [0.0] * 200)  # always 4% better: x = 0.8
    day = evidence.promoted_on
    assert day is not None
    assert evidence.wealth[day] >= 40 > evidence.wealth[day - 1]


@pytest.mark.parametrize(
    "settings",
    [
        {"scale": 0.0, "clip": 0.05},
        {"scale": math.inf, "clip": 0.05},
        {"scale": 1.0, "clip": 0.0},
        {"scale": 1.0, "clip": 0.05, "alpha": 0.0},
        {"scale": 1.0, "clip": 0.05, "alpha": 1.0},
        {"scale": 1.0, "clip": 0.05, "max_bet": 1.0},
    ],
)
def test_settings_are_checked(settings):
    with pytest.raises(ValueError):
        BettingTest(**settings)


def test_the_days_beyond_the_clip_are_listed():
    evidence = BettingTest(scale=1.0, clip=0.05).run(
        [0.01, 0.06, -0.07, 0.05], [0.0] * 4
    )
    assert evidence.clipped == (1, 2)  # 5% itself is counted in full
    assert evidence.outcomes[1:3] == (1.0, -1.0)


def test_records_of_different_lengths_or_with_gaps_are_refused():
    test = BettingTest(scale=1.0, clip=0.05)
    with pytest.raises(ValueError, match="days"):
        test.run([0.01, 0.02], [0.01])
    with pytest.raises(ValueError, match="finite"):
        test.run([0.01, math.nan], [0.01, 0.0])


# --- the test's guarantee, by simulation ---------------------------------------------


@pytest.mark.parametrize("tail", ["normal", "fat"])
def test_a_challenger_no_better_than_the_rule_is_promoted_at_most_alpha_of_the_time(
    tail,
):
    """2,000 records of 1,000 days whose mean difference is 0: at alpha 5%, the share that
    ever reaches 1/alpha stays below 5% (Ville's inequality bounds it; in practice far lower)."""
    rng = np.random.default_rng(1)
    if tail == "normal":
        d = rng.normal(0.0, 0.01, (2000, 1000))
    else:
        d = rng.standard_t(3, (2000, 1000)) * 0.01 / math.sqrt(3)
    x = np.clip(d, -0.05, 0.05) / 0.05
    _, wealth = betting_wealth(x)
    promoted = (wealth.max(axis=1) >= 1 / 0.05).mean()
    assert promoted <= 0.05


def test_a_challenger_worse_than_the_rule_is_promoted_even_less():
    rng = np.random.default_rng(2)
    x = np.clip(rng.normal(-0.0005, 0.01, (2000, 1000)), -0.05, 0.05) / 0.05
    _, wealth = betting_wealth(x)
    assert (wealth.max(axis=1) >= 1 / 0.05).mean() <= 0.01


def test_a_challenger_clearly_better_than_the_rule_is_found():
    """A difference with an annual Sharpe ratio of 3 (mean 0.19 of its daily sd) is promoted
    at alpha 2.5% within four years in nearly every record."""
    rng = np.random.default_rng(3)
    sd = 0.01
    mean = 3 / math.sqrt(252) * sd
    x = np.clip(rng.normal(mean, sd, (500, 1008)), -0.05, 0.05) / 0.05
    _, wealth = betting_wealth(x)
    assert (wealth.max(axis=1) >= 1 / 0.025).mean() >= 0.95


# --- a card's settings, from its training days -----------------------------------------


def test_the_scale_puts_the_challenger_at_the_rule_s_risk():
    rule = [0.01, -0.02, 0.015, -0.005, 0.0]
    challenger = [0.004, -0.006, 0.01, 0.0, -0.002]
    scale, _ = training_settings(challenger, rule)
    assert scale == round(statistics.stdev(rule) / statistics.stdev(challenger), 4)


def test_the_clip_is_the_largest_training_difference_rounded_up():
    rule = [0.01, -0.02, 0.015, -0.005, 0.0]
    challenger = [0.004, -0.006, 0.01, 0.0, -0.002]
    scale, clip = training_settings(challenger, rule)
    largest = max(abs(scale * c - r) for c, r in zip(challenger, rule, strict=True))
    assert largest <= clip < largest + 0.001
    assert BettingTest(scale, clip).run(challenger, rule).clipped == ()


def test_rounding_up_never_lands_below():
    steps = [k / 1000 for k in range(1, 1000)]
    above = [math.nextafter(step, 1) for step in steps]  # just above each step
    # times 1000, 16 of these round down onto the step below them
    assert sum(math.ceil(v * 1000) / 1000 < v for v in above) == 16
    for step, value in zip(steps, above, strict=True):
        assert value <= round_up(value) <= step + 0.001 + 1e-12
        assert round_up(step) == step


@pytest.mark.parametrize(
    ("challenger", "rule", "match"),
    [
        ([0.01], [0.02], "2 days"),
        ([0.0, 0.0, 0.0], [0.01, -0.01, 0.0], "risk"),
        (
            [0.005, -0.01, 0.0],
            [0.01, -0.02, 0.0],
            "nothing to test",
        ),  # the rule at half
        ([0.01, 0.02], [0.01], "days"),
        ([0.01, math.nan], [0.01, 0.0], "finite"),
    ],
)
def test_settings_cannot_come_from_too_little_or_wrong_training(
    challenger, rule, match
):
    with pytest.raises(ValueError, match=match):
        training_settings(challenger, rule)
