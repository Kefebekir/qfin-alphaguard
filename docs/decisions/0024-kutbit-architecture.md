# 0024: KUTBIT's architecture, after a review on the data

- **Status:** accepted
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

**1. The leverage is spread, not stacked.**
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

**5. The decision is computed, every minute, open positions included.**
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
  - the correlation of forecast and `z`, each day's, averaged over the days, with its t;
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
- Still to check: each instrument's cost, and the paper account's margin once Efe opens the
  second one.
- Real money stays outside this decision, as in 0023.
