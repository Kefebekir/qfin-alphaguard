# 0020: The shadow record: frozen challengers, each in its own account, judged by a bet

- **Status:** accepted
- **Date:** 2026-10-15
- **Refines:** decision 0018, part 3. Its test, a t-statistic above 2 after
  counting every challenger tried, becomes a betting test whose chance of
  error is shared out among the challengers in advance.

## Context

- **Decision 0018** lets machine learning trade only after it beats the rule
  forward, and Gate M2 asks that every challenger be frozen, with its test
  written down, before its record starts.
- **The step 5 research found nothing that beats the rule** out of sample
  (`research/step5/LEDGER.md`): eight tests written down before they ran,
  all failed. So the platform's job is to judge challengers honestly, not to
  find them, and to say early when one is no better.
- **A t-test read every evening is not a test.** A t-test is valid when it
  is read once, at a horizon fixed in advance. The record will be read every
  evening, in the day book. In 2,000 simulated records of 12 years with no
  edge, a t-statistic read every evening from the 20th day reached 1.96 (a
  one-sided 2.5% test) in **26%** of them (`research/step5/evidence_power.py`,
  `results/12_evidence_power.txt`).
- **Testing by betting** is built for this (Shafer and Vovk; Waudby-Smith and
  Ramdas 2024, JRSS B). One bets each day, against the rule, that the
  challenger will do better, with a stake fixed from the days before. If it
  is no better, the winnings never grow on average, and the chance that they
  ever reach 1/α is at most α, however often they are read and whenever the
  record stops (Ville's inequality).
- **Efe chose** that each challenger trade in its own virtual account, with
  the rule's costs and Guard.

## Decision

**1. A challenger is a card, `challengers/<name>.yaml`** (`challengers.py`).
- It holds what the challenger is and who owns it, the commit of the code
  that runs it, its weights file under the git-ignored `data/` with the
  SHA-256 of its bytes, the days it was trained on, when it was frozen, the
  first day of its record, its test's settings and a retirement day.
- The first day of the record is after the last training day and after the
  day it was frozen: only days it could not have seen count.
- Weights whose hash differs from the card's are refused, so a model cannot
  change during its record.
- Every setting is required and an unknown one is an error, as in
  `guard.yaml`. The card is committed before its record starts. Afterwards it
  is never edited, except to set `retired_on`, and never deleted.

**2. The record: two accounts on the same minutes** (`backtest/shadow.py`).
- The challenger and the rule each run through the engine in their own
  account, on the same minutes, with the same simulated broker, costs and
  `guard_qqq.yaml`. The difference between their days is then the
  difference between their decisions.
- A day's result is its closing value over the one before. Only days on
  which both accounts closed count.
- First on backtest replays; in Phase 3 on live bars, beside the paper
  account. Both sides of the record stay simulated, so that the comparison
  is fair. The paper account's fills are compared with the rule's simulated
  account separately: that is Phase 3's check that the system trades as its
  backtest says.

**3. The evidence: a bet on each day's difference at equal risk**
(`evidence.py`).
- Each day: d = scale × challenger − rule, x = clip(d, ±clip) / clip, and the
  wealth W = W × (1 + bet × x), from W = 1.
- **scale** puts the challenger at the rule's risk: the rule's standard
  deviation over the challenger's on its training days. Without it, a
  challenger that only held more would beat the rule in a rising market.
- **The bet** is aGRAPA: mean / (variance + mean²) of the x seen so far, with
  one prior day of mean 0 and variance 0.25, never below 0 and never above
  0.5. It uses only the days before.
- **Promoted** on the first day W reaches 1/α. The null hypothesis is that
  every day, given what came before, the challenger at equal risk is
  expected to do no better than the rule (after the clip).

**4. The chance of error is shared out in advance.**
- **The total is 2.5%:** the chance of promoting any challenger that does
  not beat the rule, over every card ever (`TOTAL_ALPHA`).
- **Each card states its share,** its `alpha`, before its record starts, and
  is promoted when W reaches 1 / its share. The shares of every card, retired
  ones too, add to at most 2.5%; `load_cards` refuses more. A retired card's
  share stays spent.
- **Why not count the challengers instead** (threshold = challengers so far /
  α, the first draft): a challenger alone at first would be tested at 1/α,
  and later ones at 2/α, 3/α and so on. The proven bound then becomes
  α(1 + 1/2 + … + 1/N), 2.28α for five. In simulation the excess was small
  (1.16% against 0.95% for five challengers a year apart), but the guarantee
  was gone. Adding a challenger would also have raised every other
  challenger's threshold, so a promotion could disappear. With fixed shares,
  each test is written down in full before its record starts, as Gate M2
  asks.
- **The research's ideas do not count against the shares.** The record's
  days come after the challenger was frozen, so no choice made on earlier
  data could have seen them. Choosing which challenger to freeze, from any
  number of ideas tried on 2016–2026, cannot raise its chance of a false
  promotion. They count in expectations: most ideas fail.

**5. The clip comes from the training days.**
- **The guarantee is for the clipped difference,** and a fixed clip is wrong
  either way:
  - **Too narrow hides a tail.** An insurance seller earns 0.04% a day and
    loses 9.96% on one day in 250 on average, so it is no better than the
    rule. At a clip of 5% it is promoted in **56%** of 12-year records; at
    10% in 0.3%.
  - **Too wide costs time.** With a daily difference of 0.25% and an edge
    of Sharpe ratio 2, a clip of 5% takes 5.2 years, at the median, to
    promote it; a clip from training takes 1.8.
- **So `training_settings` sets the card's scale and clip** from its
  training days. The clip is the largest difference there, rounded up to
  0.1%, so no training day is clipped.
- **A record day beyond the clip** is a loss or a gain the challenger never
  showed in training, and `Evidence.clipped` lists it.

**6. A promotion is evidence, not a switch.** A promoted challenger trades
only by Efe's decision record, after he has seen its record: its days
against the rule's, its worst days and the days beyond its clip. Guard's
limits stay the same whatever decides (decision 0019).

## Evidence

`research/step5/evidence_power.py` (`results/12_evidence_power.txt`). Its
setup: daily differences with a standard deviation of 1%, a clip of 5%, and
records of 12 years. It gives the same output on every run.

| Records with no edge | Share ever promoted |
| --- | --- |
| this test, α 2.5% | 1.3% (Ville's bound: 2.5%) |
| this test, α 5% | 2.6% (bound 5%) |
| a t-test read every evening, t ≥ 1.96 | 26% |
| five challengers a year apart, shares of 0.5% each | 0.95% of runs promote any (bound 2.5%) |

How long a real edge takes. The annual Sharpe ratio of the daily
difference; the share promoted within 2 / 4 / 8 years, and the median:

| Sharpe of the difference | Share of 2.5% | Share of 1.25% | Share of 0.5% |
| --- | --- | --- | --- |
| 0.5 | 3 / 6 / 10%, over 12 years | 1 / 3 / 6%, over 12 years | 0 / 1 / 4%, over 12 years |
| 1 | 10 / 24 / 48%, 8.3 years | 6 / 18 / 38%, 10.1 years | 2 / 11 / 29%, over 12 years |
| 2 | 51 / 88 / 100%, 2.0 years | 39 / 83 / 99%, 2.3 years | 23 / 75 / 99%, 2.8 years |
| 3 | 92 / 100 / 100%, 1.1 years | 86 / 100 / 100%, 1.3 years | 75 / 100 / 100%, 1.5 years |

**On real minutes** (`research/step5/shadow_demo.py`, `results/13_shadow_demo.txt`). It
replays the rule and a variant that decides every 15 minutes, each in its own
account behind `guard_qqq.yaml`. The variant's card is written as if frozen
at the end of 2022: scale 0.9299 and clip 2.8% from 2016–2022.

- From 3 January 2023 to 9 October 2026, the rule earned 18.7% a year
  (Sharpe 1.56) and the variant 13.9% at the rule's risk (1.14), 1.87 bps a
  day less.
- The bet never grew: W ended at 0.982, and the variant was not promoted.
- One day was beyond the clip: 7 April 2025, after the tariff announcement.
- It is a demonstration, not a test. The research had looked at 2023–2026,
  so its card stays out of `challengers/` and spends no share.

The tests break each part of the platform one at a time. Of 29 such changes,
28 fail a test. The 29th turns `ceil` into `floor` plus the guard after it,
which is the same function.

## Consequences

- **Only large edges can be proven soon.**
  - **Sharpe 2 or more:** an edge whose daily difference has an annual
    Sharpe ratio of 2 is found in about two years.
  - **Sharpe 1:** in eight or more.
  - **A typical improvement may never be proven.** Take a challenger that
    raises the rule's Sharpe ratio from 1.3 to 1.6 at a correlation of 0.9
    with it. Its difference has a Sharpe ratio of about 0.7: not provable
    within a decade.

  The record still shows early, and cheaply, a challenger that is worse.
- **The share is Efe's choice for each card.** All of 2.5% for one
  challenger finds an edge soonest; five shares of 0.5% leave room for
  others and cost about 40% more time.
- **The first challenger is Efe's own model,** trained by him. A guide to
  training and freezing it comes next. `challengers/README.md` describes
  the card.
- **Before Phase 3:**
  - the simulated broker needs a half spread per instrument (an open item
    from step 2);
  - the live engine runs the challengers' accounts on the same bars as the
    rule's. Only the rule's orders go to Alpaca's paper account.
