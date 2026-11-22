# 0024: KUTBIT's architecture, after a review on the data

- **Status:** accepted; the skill test's daily correlation amended on 17 November 2026, after
  its placebo's first run and before any real run (see the amendment at the end); sections 1
  and 5 changed by decision 0025, which makes the hedge a decision; the skill test run on 22
  November 2026 (see its result at the end)
- **Date:** 2026-11-17 (agreed with Efe on 16 November)
- **Builds on:** decision 0023, which it changes where this record says so. Everything else in
  0023 stands: KUTBIT is separate from the QQQ rule, has its own Guard file and accounts, trades
  within the day only, and is judged by the shadow record.

## Context

Before any model was trained, decision 0023's design was checked against the stored minutes of
2016-2021, the development years. 2022-2026 were not read.

- **One position at 10x cannot survive a single minute's jump.** Over 30 minutes, at 10x:

  | Account | Its move's std | Worst 1 in 1,000 | Worst seen |
  |---|---|---|---|
  | One stock at 10x | 4.43% | −27.1% | −141.8% |
  | 10 stocks at 1x, all long | 2.63% | −15.7% | −35.9% |
  | 10 stocks at 1x, all long, QQQ hedged | 1.43% | −8.1% | −21.4% |

  Of 28.4 million single-stock minutes, 16 moved more than 5% and one 41% (CHK, 8 February 2016).
- **Much of a stock's move is the market's.** At a beta known before the open (median 0.69),
  QQQ is 29% of the variance of a stock's 30-minute moves. With it taken out, the mean
  correlation between two stocks falls from 0.28 to 0.06, so ten hedged positions are close to
  ten separate bets.
- **The clock dominates raw moves.** A move in the first five minutes is about four times a
  midday one. Scaled by the day's volatility alone, a model would spend itself learning the clock.
- **A stock's own drift does not persist.** The correlation of stocks' mean open-to-close
  return from one year to the next was +0.51, +0.81, −0.42, −0.06 and +0.06. A free identity
  vector per stock would memorise it.
- **The known patterns are small.** Across stocks, the same half hour of the 5 sessions before
  predicts the next, apart from the market, at a correlation of +0.009 (t 4.9); the half hour
  just before, at −0.017 (t −8.6), partly bid-ask bounce. On 2 bps a round trip, a trade on the
  strongest signals needs about 0.04.
- **Alpaca's free live feed shows a different chart.** On 2021, IEX alone held 3.8% of stocks'
  volume and had a trade in 83% of their minutes. Inputs built from it correlated with those of
  the full tape at 0.75 for a minute's move and 0.43 for its volume; summed over 5, 15 and 30
  minutes the moves agreed at 0.92, 0.97 and 0.98.
- **0023's margin line is out of date.** FINRA replaced the pattern day trader rule, and its
  25,000 USD minimum, with an intraday margin standard from 4 June 2026 (Regulatory Notice
  26-10). Alpaca's documentation gives up to 4x intraday buying power from 2,000 USD of equity.
  When Alpaca applies the new rule, and whether its paper accounts follow it, is not stated.
- **Efe's wish, restated on 16 November.** A bot that, for each stock and at each minute,
  forecasts where the position goes; keeps risk as low as it can be; earns as much as it can at
  very high leverage; and keeps learning each stock as it goes, improving on its own earlier
  gains. It is written from scratch: no open-source model, no pretrained weights.

## Decision

**1. The leverage is spread, not stacked.** (How much is hedged is changed by decision 0025.)
- The account's gross leverage is Efe's cap: 4x on paper, 10x in the virtual account.
- It is shared among several positions at once, each with a cap of its own, and each position's
  market part is hedged with QQQ. Longs and shorts offset part of the hedge.
- A position's size is continuous, from nothing up to its cap, and follows the strength of its
  forecast: leverage is the output of the edge, as the Kelly criterion gives it, never more than
  the caps.

**2. What the forecaster learns: the hedged return in usual moves.**
- For each horizon of 1, 5, 15, 30 and 60 minutes, `z`: the return of a long position less beta
  times QQQ's over the same bars, over the usual size of a move in the minutes held
  (`labels.targets`, #108).
- The worst and best moments along the way stay the instrument's own, since a stop sits on it.

**3. What it sees: inputs v2** (#107). Each minute's move over the usual move at that minute
(`swing`); the move apart from QQQ; the beta; and `habit`, what the instrument did apart from
QQQ over the same 30 minutes of the 5 sessions before. A sector ETF's part comes later, when the
repository has each stock's sector.

**4. The forecast is an edge and a scale; the shape comes from real moves.**
- The network gives, for each horizon, the expected `z` (the edge) and how large today's moves
  are against the usual (the scale).
- The shape of what may follow, its tails and its worst moments, comes from a bank of real past
  paths in the same units, shifted and stretched by the forecast. `z`'s kurtosis is 7 at 30
  minutes and 20 at one minute, so no formula's shape will do.

**5. The decision is computed, every minute, open positions included.** (Its measure is
changed by decision 0025: the certainty equivalent at Efe's risk appetite.)
- For each candidate size of each position, including keeping, cutting or closing one already
  held, the account's expected log growth after costs is computed over those paths, under Efe's
  stop, caps and costs. The best is taken; if none adds growth, KUTBIT waits.
- A change of Efe's settings changes the decision at once, without retraining.
- Reinforcement learning stays a variant, for managing an open position, beside this rule.

**6. It learns at three speeds, and cannot learn itself worse.**
- **Every minute:** each instrument's calibration and a small head of its own are updated from
  every forecast whose outcome is known. The head is pulled toward the shared model, so a few
  minutes cannot throw it.
- **Every night:** each instrument's beta, usual moves and volumes, and its costs, learned from
  KUTBIT's own fills.
- **Every month:** the shared network's weights. Development uses the same monthly schedule as
  live; 0023's yearly retraining in development against nightly live is replaced.
- **Versions are mixed by their recent forecast loss.** A new month's network joins the earlier
  ones, and each version's weight follows how well it has forecast lately, so a worse version
  loses its weight by itself.
- The per-instrument head replaces 0023's free identity vector; a free identity vector is
  only a variant.

**7. It is built from scratch.** No pretrained model, no open-source model or strategy, no
ready-made model classes. PyTorch is used as a tool, for tensors and derivatives only. Efe writes
the network's layers, its loss and its training loop. LightGBM is dropped.

**8. A ladder before the network.** Each rung must beat the one below on the same years before
the next is built:
1. the known patterns, with no parameters: `habit`, and the half hour just before reversed;
2. a linear model of the same inputs (the skill test below);
3. Efe's network.

**9. The skill test, its rules written before it runs.**
- **Model:** a ridge regression of each horizon's `z` on everything the network will see: the
  120-minute window's 8 channels and the 11 scalars, 971 inputs.
- **Instruments:** every stock and ETF of each year's list, but QQQ, which is the hedge.
- **Years:** fitted on 2016-2018 for each of five penalties; the penalty with the highest
  correlation on 2019 is kept; refitted on 2016-2019; measured once on 2020-2021. 2022-2026 are
  not read.
- **Rows:** every minute of every usable session but its last. Inputs and `z` are clipped at
  ±10 and ±5 for the fit.
- **Measures, on 2020-2021, for each horizon:**
  - the correlation of forecast and `z`, each day's, averaged over the days, with its t (taken
    about the means of all the rows, not each day's own, since the amendment below);
  - the same within each minute, across instruments, which takes out what is common to all;
  - the hedged return, in bps, of trading in the forecast's direction on its strongest 10% and
    1%, by thresholds set on 2019, against 2 bps a round trip.
- **The placebo, which must show nothing, or the harness is wrong:** every session's prices
  replaced by a random walk with each minute's usual move, run through the same code. No
  forecaster may reach |t| 3 at any horizon in any measure. It runs first; the real run waits
  for it.
- **No placebo of shuffled targets here.** A model fitted on real targets dealt to another day
  or another instrument still carries what is real in them, such as each stock's pattern
  through the day or one instrument leading another, so it need not show nothing even when
  the harness is right; on planted data it did not. Only the random walk is empty by
  construction. 0023's shuffled labels stay for the variants, as a bar to beat.
- **What it decides.** It is a ruler, not a variant: it does not count among the ten, and it is
  never traded. Efe's network must beat its correlation on the same years. If the strongest 1%
  of its forecasts already pay the cost at some horizon, the inputs hold a tradable edge; if no
  horizon comes near, that is written down as it is, and Efe decides whether the network is
  still worth building.
- Efe runs it.

**10. The live feed.** The free IEX feed does not show the minute chart the model is trained
on. KUTBIT goes to paper with the full consolidated feed, Alpaca's paid plan (listed at 99 USD a
month). Nothing is bought before then; at that point the price and Efe's balance are checked first.

## Consequences

- Done: the inputs, the labels and the data set (#106), inputs v2 (#107), the target (#108).
- Next, in order: the skill test; the bank of real paths; the simulator, whose broker, margin
  and stop rules Efe writes; the decision rule, Efe's; Efe's network.
- Still to check: each instrument's cost, measured from the consolidated tape's quotes by
  `research/kutbit/costs.py`, whose rules its docstring gives; and the paper account's margin
  once Efe opens the second one.
- Real money stays outside this decision, as in 0023.

## Amendment of 17 November 2026: the daily correlation

- **What happened.** The skill test's placebo ran first, as section 9 requires. It failed on
  one measure only: the last 30 minutes reversed, in each day's correlation, at t +5.8, +5.8,
  +6.6, +7.8 and +8.7 for 1, 5, 15, 30 and 60 minutes. The ridge and `habit` passed every
  measure, and the reversal every other one. The real run did not start; no real price of
  2020-2021 was read.
- **Why.** The measure took each day's correlation about that day's own means. Every
  instrument of a day shares QQQ's path. Over a day a path's ups and downs add up to the day's
  move, so once the day's mean is taken out, a rise above the mean before a minute leaves less
  than the mean after it: the past and the future move against each other, even in a random
  walk. An instrument's own path averages out over the day's instruments; the shared one
  does not.
- **The check, on random walks of 2017-2019** (2020-2021 not read): the old measure gave the
  reversal t +5.6 to +10.0; with QQQ held flat, so that no path was shared, t +1.5 to +2.2.
  About the means of all the rows the same correlation gave |t| at most 1.0 in both cases,
  and the correlation across instruments at most 0.8.
- **The change.** Each day's correlation is taken about the means of all the measured rows;
  it is still averaged over the days, with its t. The correlation across instruments, the
  strongest forecasts' returns and the placebo rule are unchanged. A test plants a path shared
  by a day's instruments and checks that the measure shows nothing where the old one did.
- **Why it is fair.** The change was decided on random walks alone, before the real run, so
  it cannot have been chosen to suit a result. On the real prices the old measure would have
  tilted every forecast built on the same day's past: toward a reversal, against momentum.
  Efe runs the placebo again; the real run still waits for it.

## Result of 22 November 2026: a real forecast, too small to pay 2 bps on all instruments

Efe ran both on 22 November.

- **The placebo passed** on its second run. None of its 180 t values reached 3; the largest
  was +2.59 (`habit`, 1 minute, ETFs, each day's correlation). The reversal's daily t, +5.8 to
  +8.7 before the amendment, was +0.9 to +2.2.
- **The real run** took 7 minutes. It fitted on 24,458,227 rows of 2016-2019 and was measured
  once on 12,627,628 rows of 2020-2021. The penalty kept was 0.1 at every horizon but 30
  minutes, where it was 0.01. On all instruments:

  | | 1 min | 5 min | 15 min | 30 min | 60 min |
  | --- | --- | --- | --- | --- | --- |
  | Ridge, each day's correlation (t) | +0.006 (7.7) | +0.009 (6.6) | +0.012 (5.5) | +0.014 (4.7) | +0.016 (3.6) |
  | Ridge, within each minute across instruments (t) | +0.010 (17.1) | +0.013 (13.1) | +0.013 (9.0) | +0.013 (6.6) | +0.012 (4.3) |
  | `habit`, each day's correlation (t) | +0.002 (2.7) | +0.003 (2.1) | +0.004 (1.9) | +0.005 (1.6) | +0.001 (0.3) |
  | The last 30 minutes reversed, each day's correlation (t) | +0.003 (3.9) | +0.007 (4.0) | +0.009 (3.2) | +0.008 (2.3) | +0.008 (2.2) |
  | Ridge's strongest 1%, bps a signal before costs (t) | +0.34 (5.4) | +1.21 (3.8) | +1.89 (2.4) | +1.89 (1.5) | +1.89 (1.1) |
  | The same after 2 bps | −1.66 | −0.79 | −0.11 | −0.11 | −0.11 |

- **What it shows.**
  - The inputs hold a forecast. The ridge's correlation is above zero at every horizon, at a
    t of 3.6 to 17.1, and above both known patterns at every horizon in both correlations:
    the ladder's second rung beats its first.
  - The forecast is small, a correlation of about 0.01, and out of sample it was about twice
    too large: the slope of `z` on it was 0.36 to 0.55.
  - The strongest 1% of its forecasts, about 360 signals a day, earn 1.89 bps a signal before
    costs at 15, 30 and 60 minutes, 0.11 bps short of 2 bps a round trip. They pay it only in
    parts: the stocks at 15 minutes (+0.02 after costs, on 285 signals a day), and the ETFs at
    30 and 60 minutes (+0.14 and +1.28, on about 70 signals a day, at a t of only 1.6 and 1.7
    before costs). The strongest 10% fall 1.5-1.9 bps short at every horizon.
  - A signal is one minute's forecast, and one trade may span several. The 2 bps leave out the
    hedge's own cost. At the 5 bps that the first agent's gym charges at an extreme minute,
    none of these pays.
- **What it decides** (section 9). On all instruments the strongest 1% come near the cost
  without paying it; they pay it only in the parts above. Whether a forecast this size can pay
  depends on each instrument's real cost, which is still to check. Efe decides whether the
  network is worth building. If it is built, it must beat the ridge's correlation on the same
  years.
