# 0027: KUTBIT's network, the ladder's third rung

- **Status:** accepted
- **Date:** 2026-11-23 (agreed with Efe on 23 November)
- **Builds on:** decision 0023 (variants, placebos, who writes what), 0024 (the inputs, the
  target, the ladder, the skill test and its results, the costs) and 0026 (variant 1 and its
  result). It changes the order of 0024's next steps.
- **Variant:** 2 of the 10 that 0023 allows.

## Context

- **The ladder so far** (0024, section 8). Its first two rungs ran on 22 November. The ridge
  forecasts `z` at every horizon, above both known patterns, at a t of 3.6 to 17.1. But its
  correlation is about 0.01, and its strongest 1% earn 1.89 bps a signal before costs at
  15-60 minutes.
- **What a round trip costs,** measured on 23 November (0024's result): 2.80 bps for the
  median stock, 1.17 for the median ETF and 0.62 for QQQ. The ridge's strongest 1% are below
  the median stock's cost in every part of the day, and above that of about a third of the
  instruments, mostly ETFs and the largest stocks.
- **Variant 1 failed** (0026). On 2019 no bar made money after 5 bps. After 2 bps the most
  any made was +0.23 bps an event, betting against events without a headline at the open,
  where a round trip costs about 8 bps (0026's page of 2019). Cheaper costs alone would not
  rescue an agent of extreme minutes.
- **What Efe's goal asks.** To multiply the account by 4 in about 900 trades, each trade must
  add about 15 bps to it after costs (ln 4 / 900). At 4x that is about 4 bps a position, or
  5-6 bps before the cheapest instruments' costs: about three times the ridge's strongest 1%.
- **Efe's choice, 23 November.** His network next, before the bank of real paths and the
  simulator. They are worth building only for a forecast that pays its costs.

## Decision

**1. What it is.** Efe's network: the ladder's third rung, and variant 2 of 0023's ten. It
forecasts the skill test's target from the skill test's inputs, on the same rows and years,
so that only the model differs from the ridge.

**2. Its rows, inputs and target are the skill test's.**
- Every minute but the last of every usable session of every instrument but QQQ, on days
  when QQQ's session is usable too.
- The 971 inputs: the 120-minute window's 8 channels and the 11 scalars, clipped at ±10 and
  standardised on the rows it learns from.
- For each horizon of 1, 5, 15, 30 and 60 minutes, `z`, clipped at ±5.
- A test checks, row by row, that the network is given what the ridge was given.

**3. The network: design A.**
- A linear path from the 971 inputs to the five horizons. Beside it, two hidden layers of
  width W with ReLU, whose output is added to the linear path's.
- The linear path and the hidden path's last layer start at zero, so the network starts by
  forecasting nothing, and its hidden layers add only what they find. The hidden layers
  start at He's scale.
- Its loss is the mean square error of the five horizons' clipped `z`.
- It learns with Adam, written by hand: a rate of 0.001, β of 0.9 and 0.999, and a weight
  decay λ on every weight but the biases. Batches are of 4,096 rows, in a new random order
  each epoch.
- Five seeds learn it; its forecast is their mean.
- It is written from scratch (0024, section 7): PyTorch for tensors and derivatives only, no
  `torch.nn`, no `torch.optim`.

**4. The setting, chosen on 2019.**
- Four settings, in this order: W 128 with λ 0.1; W 128 with λ 0.01; W 512 with λ 0.1; W 512
  with λ 0.01.
- Each setting's five seeds learn on 2016-2018 for 20 epochs. After each epoch, their mean
  forecasts 2019.
- The setting and the epoch kept are those whose forecasts of 2019 score best: each day's
  correlation with `z`, about the means of all the rows, averaged over the days and over 15,
  30 and 60 minutes. A tie keeps the earlier.
- With the setting kept, the five seeds learn again on 2016-2019, for the epochs kept.
- The thresholds of its strongest 10% and 1% are set on its forecasts of 2019, as the
  ridge's were.

**5. The measures, once, on 2020-2021.** For the network, the ridge, the network on shuffled
labels and the two known patterns, at each horizon, on all instruments, the stocks and the
ETFs:
- each day's correlation with `z`, about the means of all the rows, averaged over the days,
  with its t;
- the same within each minute, across instruments;
- the hedged return of trading in the forecast's direction on its strongest 10% and 1%, in
  bps a signal: before costs; after 2 bps; after each signal's own cost (section 7); and
  after twice that.

**6. The ridge beside it.** The harness fits the ridge again as the skill test did, with the
same penalties chosen on 2019, and forecasts the same rows. Its measures must equal those of
0024's result of 22 November; if they do not, the run stops. The two are then compared day by
day.

**7. A signal's own cost** (0024's result of 23 November). The mean round trip of its
instrument in its part of the day (the first 30 minutes, the day between, the last 30
minutes), over the cost study's moments of 2016-2021; plus the instrument's beta times QQQ's
in that part, for the hedge. An instrument without moments takes the median of its kind,
stock or ETF, in that part. A spread is there to see when an order goes out, so this reads no
future return.

**8. The gates.**
1. **The placebo** runs first, through the same code: every session's prices replaced by the
   skill test's random walk, the choice made on its 2019 and the measures taken on its
   2020-2021. No forecaster may reach |t| 3 at any horizon in any measure. The real run waits
   for it.
2. **The ridge.** The network's day-by-day margin over the ridge in each day's correlation,
   averaged over 15, 30 and 60 minutes, must have a t of at least 2.
3. **Shuffled labels** (0023, section 6). The same network, with the setting and the epochs
   kept, learns on 2016-2019 with each minute's `z` shuffled between the instruments of that
   minute. The real network's margin over it, measured the same way, must have a t of at
   least 2.
4. **Its cost.** At one horizon of 15, 30 and 60 minutes at least, its strongest 1% must earn
   more than their own cost, with a t, clustered by day, of at least 2.

- Failing gate 2 or 3, variant 2 failed, and it is written down as it is.
- Passing gates 2 and 3 but not 4, the network is a better ruler than the ridge, but no
  forecast pays yet. The bank of real paths and the simulator wait, and Efe decides.
- Passing all four, the bank of real paths and the simulator come next.

**9. The runs and their records.**
- Efe runs them, in this order: a check of speed that shows no score; the placebo; the choice
  on 2019; the measurement, once.
- Each record keeps the SHA-256 of `net.py`, the harness and the data code. The choice waits
  for a placebo that passed on today's code. The measurement waits for that placebo and a
  choice made on today's code, and it runs only once.
- A page shows each run.

**10. Who writes what.**
- Efe: `research/kutbit/net.py`, step by step from a guide: the network's layers, its loss,
  Adam, its learning loop and its seeds. He runs every run.
- Claude: the network's data and its batches on the GPU; the harness, the shuffled labels,
  the placebo and the measures; the tests, the guide and the pages.

**11. Left for the forecaster that trades.** The scale of today's moves (0024, section 4),
each instrument's own head and the monthly learning (0024, section 6), and the calibration
(0023, section 6). This first test asks one question: does a network see more in the same
inputs than the ridge does?

## Consequences

- 0024's order changes. Efe's network comes first; the bank of real paths, the simulator and
  the decision rule come only after gate 4.
- Pull requests, in order: this record; the network's data, with its check of speed; the
  harness, with a stand-in network; the guide and the tests of `net.py`. Then Efe writes
  `net.py` and runs it.
- Decision 0023 allows eight variants after this one.
- No promise is made. On data this noisy a network often beats a well-tuned linear model by
  little, or not at all. Even passing every gate, Efe's goal asks about three times the
  ridge's edge.
