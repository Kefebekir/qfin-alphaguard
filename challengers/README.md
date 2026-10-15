# Challengers

Each file here is the card of one frozen challenger, `<name>.yaml`,
committed before its record's first day
([decision 0020](../docs/decisions/0020-the-shadow-record.md)). There are
none yet.

A card is never edited after it is committed, except to set `retired_on`.
It is never deleted: its share of the chance of error stays spent.

## A card

```yaml
name: day-sizer                      # lower-case words joined by '-'; the file is day-sizer.yaml
description: Sizes the rule's day from what is known at 10:00.
owner: Efe
code_commit: 0123456789abcdef0123456789abcdef01234567   # the full commit of the code that runs it
weights: data/models/day-sizer.npz   # under the git-ignored data/
weights_sha256: <64 hexadecimal characters: sha256sum of the weights file>
trained_from: 2016-01-26
trained_to: 2026-10-09
frozen_at: 2026-10-16 12:00:00+00:00 # with its time zone
record_from: 2026-10-19              # after trained_to, and after the day it was frozen
scale: 1.1                           # from evidence.training_settings on the training days
clip: 0.031                          # the same
alpha: 0.0125                        # its share of the total, 0.025
retired_on: null                     # a date once its record ends
```

`challengers.load_cards()` reads every card here. It refuses a card that
breaks a rule, and a set of cards whose `alpha`s add to more than 0.025.
`challengers.weights_path(card)` refuses weights whose bytes are not the
ones frozen.

## How a challenger is judged

- **Its record.** From `record_from`, the challenger and the rule run on the
  same minutes, each in its own account. Both have the same simulated broker,
  costs and `guard_qqq.yaml` (`backtest/shadow.py`). A day's result is the
  account's closing value over the one before.
- **The evidence** (`evidence.py`). Each day:
  - the difference at equal risk, `scale × challenger − rule`, clipped and
    divided by `clip`;
  - a bet on it, fixed from the days before, that the challenger does better;
  - the wealth from that bet, starting at 1.

  If the challenger is no better than the rule, the wealth reaches
  1 / `alpha` with a chance of at most `alpha`, however often it is read.
- **Promoted** when the wealth reaches 1 / `alpha`. A promotion is evidence
  for Efe's decision record, never a switch.
- **What the record shows** (`shadow.describe`):
  - the challenger's raw results beside those at the rule's risk. A model
    that only takes more risk is ahead raw and level at equal risk;
  - the days beyond the clip: results the challenger never showed in
    training.

## Choosing the share

The `alpha`s of every card ever, retired ones too, add to at most 0.025. A
larger share finds a real edge sooner. Below, the median time to promotion
for an edge whose daily difference with the rule has the given annual
Sharpe ratio (`research/step5/results/12_evidence_power.txt`):

| Sharpe of the difference | `alpha` 0.025 | 0.0125 | 0.005 |
| --- | --- | --- | --- |
| 1 | 8.3 years | 10.1 years | over 12 years |
| 2 | 2.0 years | 2.3 years | 2.8 years |
| 3 | 1.1 years | 1.3 years | 1.5 years |

## Freezing a challenger

1. Train it on days up to `trained_to` only.
2. Run it and the rule through the engine over the training days, each in its
   own account (`run_backtest`, then `shadow.daily_returns`). Take `scale` and
   `clip` from `evidence.training_settings(challenger_days, rule_days)`.
3. Save the weights under `data/models/`, and take their hash with
   `sha256sum`.
4. Commit the code that runs it, and write the card with that commit.
   Commit the card before `record_from`.
