"""The shadow record on real minutes: the rule against a variant, each in its own account.

Shows the platform (decision 0020) working end to end on the stored QQQ minutes, behind the
real Guard (guard_qqq.yaml, with max_order_usd lifted as the accounts grow, as in
research/intraday_pilot/guard_needs.py), from 30,000 USD at QQQ's half spread of 0.3 bps:

- the rule: momentum.py's NoiseAreaMomentum, deciding every 30 minutes from 10:00;
- a variant as the challenger: the same rules every 15 minutes from 09:45
  (MomentumSettings(every_min=15)), its settings file standing for the weights.

Its card is written as if frozen on 30 December 2022: its scale and clip from 2016-2022
(evidence.training_settings), its record from 3 January 2023 to 9 October 2026. This is not a
test. The step 5 research looked at 2023-2026, so this record spends nothing from the platform's
alpha and its card stays out of challengers/. It shows the parts working together on real data:
two engine runs, their daily returns, the card, its checks, the record and its evidence.

Usage: shadow_demo.py ROOT GUARD_QQQ_YAML OUT_DIR
"""

import hashlib
import json
import math
import subprocess
import sys
from dataclasses import asdict, replace
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import yaml

from qfin_alphaguard.backtest.broker import CostModel, NextBarBroker
from qfin_alphaguard.backtest.engine import run_backtest
from qfin_alphaguard.backtest.shadow import daily_returns, shadow_record
from qfin_alphaguard.challengers import load_card, weights_path
from qfin_alphaguard.data.intraday import minute_feed
from qfin_alphaguard.evidence import training_settings
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
VARIANT = MomentumSettings(every_min=15)


def returns(settings):
    feed = minute_feed(
        ROOT, {year: ["QQQ"] for year in range(FIRST.year, LAST.year + 1)}, FIRST, LAST
    )
    broker = NextBarBroker(CostModel(half_spread_bps=0.3))
    result = run_backtest(feed, NoiseAreaMomentum(settings), broker, Guard(GUARD), CASH)
    return daily_returns(result, CASH)


def year_stats(values):
    v = np.asarray(values)
    return f"{v.mean() * 252:+.1%} a year, Sharpe {v.mean() / v.std(ddof=1) * math.sqrt(252):.2f}"


rule, challenger = returns(MomentumSettings()), returns(VARIANT)
training = sorted(d for d in rule.keys() & challenger.keys() if d <= TRAINED_TO)
scale, clip = training_settings(
    [challenger[d] for d in training], [rule[d] for d in training]
)

OUT.mkdir(parents=True, exist_ok=True)
settings_file = OUT / "variant.json"
settings_file.write_text(json.dumps(asdict(VARIANT), sort_keys=True) + "\n")
card = {
    "name": "every-15-minutes",
    "description": "The rule's decisions every 15 minutes from 09:45 (a demonstration).",
    "owner": "Efe",
    "code_commit": subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip(),
    "weights": settings_file.as_posix(),
    "weights_sha256": hashlib.sha256(settings_file.read_bytes()).hexdigest(),
    "trained_from": training[0],
    "trained_to": training[-1],
    "frozen_at": datetime(2022, 12, 30, 23, 0, tzinfo=UTC),
    "record_from": RECORD_FROM,
    "scale": scale,
    "clip": clip,
    "alpha": 0.025,
    "retired_on": None,
}
card_file = OUT / "every-15-minutes.yaml"
card_file.write_text(yaml.safe_dump(card, sort_keys=False))
frozen = load_card(card_file)
weights_path(frozen)  # the settings file is the one frozen

record = shadow_record(frozen, challenger, rule, until=LAST)
evidence = record.evidence
print(f"training {training[0]} to {training[-1]}, {len(training)} days")
print(f"   rule {year_stats([rule[d] for d in training])}")
print(f"   variant {year_stats([challenger[d] for d in training])}")
print(f"   card: scale {scale}, clip {clip:.1%}, alpha 2.5%: promoted at wealth 40")
print(f"record {record.days[0]} to {record.days[-1]}, {len(record.days)} days")
print(f"   rule {year_stats(record.rule)}")
print(
    f"   variant at the rule's risk {year_stats([scale * c for c in record.challenger])}"
)
diff = np.asarray(record.challenger) * scale - np.asarray(record.rule)
print(
    f"   difference {diff.mean() * 1e4:+.2f} bps a day, t "
    f"{diff.mean() / diff.std(ddof=1) * math.sqrt(len(diff)):+.2f}"
)
print(
    f"   wealth at the end {evidence.wealth[-1]:.3f}, at most {max(evidence.wealth):.3f}; "
    f"promoted: {'on ' + str(record.days[evidence.promoted_on]) if evidence.promoted else 'no'}"
)
print(f"   days beyond the clip: {[str(record.days[i]) for i in evidence.clipped]}")
for year in sorted({d.year for d in record.days}):
    last = max(i for i, d in enumerate(record.days) if d.year == year)
    print(f"   wealth at the end of {year}: {evidence.wealth[last]:.3f}")
