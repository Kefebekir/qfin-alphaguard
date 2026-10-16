"""The shadow record on real minutes: the rule against two challengers, each in its own account.

Shows the platform (decision 0020) working end to end on the stored QQQ minutes, behind the
real Guard (guard_qqq.yaml, with max_order_usd lifted as the accounts grow, as in
research/intraday_pilot/guard_needs.py), from 30,000 USD at QQQ's half spread of 0.3 bps.
The rule is momentum.py's NoiseAreaMomentum, deciding every 30 minutes from 10:00. Against it:

1. a variant that applies the same rules every 15 minutes from 09:45
   (MomentumSettings(every_min=15));
2. the rule at double size (cap 4, target volatility 4%), in an account whose size limits are
   doubled too (max_weight and max_gross_exposure 4.2): a thought experiment, since
   guard_qqq.yaml's 2.1 would cut it. It makes the same decisions, so it has no new skill. It
   shows why the test compares at equal risk while the raw results are shown beside it.

For each, the last line says what the same test on raw results (scale 1, its clip from the
same training days) would have said, for comparison only.

Each card is written as if frozen on 30 December 2022, with its settings file standing for the
weights: its scale and clip from 2016-2022 (evidence.training_settings), its record from 3
January 2023 to 9 October 2026. These are not tests. The step 5 research looked at 2023-2026,
so the records spend nothing from the platform's alpha and the cards stay out of challengers/.

Usage: shadow_demo.py ROOT GUARD_QQQ_YAML OUT_DIR
"""

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, replace
from datetime import UTC, date, datetime
from pathlib import Path

import yaml

from qfin_alphaguard.backtest.broker import CostModel, NextBarBroker
from qfin_alphaguard.backtest.engine import run_backtest
from qfin_alphaguard.backtest.shadow import daily_returns, describe, shadow_record
from qfin_alphaguard.challengers import load_card, weights_path
from qfin_alphaguard.data.intraday import minute_feed
from qfin_alphaguard.evidence import BettingTest, round_up, training_settings
from qfin_alphaguard.guard.config import load_guard_config
from qfin_alphaguard.guard.rules import Guard
from qfin_alphaguard.momentum import MomentumSettings, NoiseAreaMomentum

ROOT, LIMITS, OUT = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
CASH = 30_000.0
FIRST, TRAINED_TO, RECORD_FROM, LAST = (
    date(2016, 1, 1),
    date(2022, 12, 30),
    date(2023, 1, 3),
    date(2026, 10, 9),
)
GUARD = replace(load_guard_config(LIMITS), max_order_usd=1e12)
DOUBLED = replace(GUARD, max_weight=4.2, max_gross_exposure=4.2)
COMMIT = subprocess.run(
    ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
).stdout.strip()


def returns(settings, limits):
    feed = minute_feed(
        ROOT, {year: ["QQQ"] for year in range(FIRST.year, LAST.year + 1)}, FIRST, LAST
    )
    broker = NextBarBroker(CostModel(half_spread_bps=0.3))
    result = run_backtest(
        feed, NoiseAreaMomentum(settings), broker, Guard(limits), CASH
    )
    return daily_returns(result, CASH)


def freeze(name, description, settings, challenger, rule):
    """Write the challenger's card as if frozen at the end of 2022, and read it back."""
    training = sorted(d for d in rule.keys() & challenger.keys() if d <= TRAINED_TO)
    scale, clip = training_settings(
        [challenger[d] for d in training], [rule[d] for d in training]
    )
    OUT.mkdir(parents=True, exist_ok=True)
    weights = OUT / f"{name}.json"
    weights.write_text(json.dumps(asdict(settings), sort_keys=True) + "\n")
    card = {
        "name": name,
        "description": description,
        "owner": "Efe",
        "code_commit": COMMIT,
        "weights": weights.as_posix(),
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "trained_from": training[0],
        "trained_to": training[-1],
        "frozen_at": datetime(2022, 12, 30, 23, 0, tzinfo=UTC),
        "record_from": RECORD_FROM,
        "scale": scale,
        "clip": clip,
        "alpha": 0.025,
        "retired_on": None,
    }
    path = OUT / f"{name}.yaml"
    path.write_text(yaml.safe_dump(card, sort_keys=False))
    frozen = load_card(path)
    weights_path(frozen)  # the settings file is the one frozen
    raw_clip = round_up(max(abs(challenger[d] - rule[d]) for d in training))
    return frozen, len(training), raw_clip


rule = returns(MomentumSettings(), GUARD)
challengers = [
    (
        "every-15-minutes",
        "The rule's decisions every 15 minutes from 09:45.",
        MomentumSettings(every_min=15),
        GUARD,
    ),
    (
        "double-size",
        "The rule at double size: the same decisions, no new skill.",
        MomentumSettings(cap=4.0, target_vol=0.04),
        DOUBLED,
    ),
]
for number, (name, description, settings, limits) in enumerate(challengers, 1):
    challenger = returns(settings, limits)
    card, trained, raw_clip = freeze(name, description, settings, challenger, rule)
    print(f"{number}. {name}: {description}")
    print(
        f"card: trained on {card.trained_from} to {card.trained_to} ({trained} days), "
        f"scale {card.scale}, clip {card.clip:.1%}, alpha {card.alpha}"
    )
    record = shadow_record(card, challenger, rule, until=LAST)
    print(describe(record))
    raw = BettingTest(scale=1.0, clip=raw_clip).run(record.challenger, record.rule)
    verdict = (
        f"promoted on {record.days[raw.promoted_on]}"
        if raw.promoted
        else "not promoted"
    )
    print(f"the same test on raw results (clip {raw_clip:.1%}): {verdict}")
    print()
