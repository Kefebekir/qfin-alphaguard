# 0027: KUTBIT's network, the ladder's third rung

- **Status:** accepted; measured once on 23 November 2026 and failed (see the result at the
  end)
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

## Result of 23 November 2026: failed

Efe ran the placebo, the choice and the measurement on 23 November; the three records keep
the same code's fingerprint. A correlation is each day's, about the means of all the rows,
averaged over the days; a t is in brackets.

- **The placebo passed.** On the random walks no forecaster reached |t| 3. The largest were
  1.42 for the network, 2.06 for the ridge, 1.96 for the network on shuffled labels, and 1.84
  and 2.58 for the two known patterns; the ridge's and the patterns' are those of 0024's
  placebo. Every gate failed there, as it should: the margin over the ridge was −0.0029
  (t −0.55), over the network on shuffled labels −0.0032 (t −0.98), and no horizon paid its
  own cost. The setting kept on the walks' 2019, W 128 with λ 0.1 after 19 epochs, scored
  +0.0061 there, and −0.0005 to +0.0016 on their 2020-2021 (t −0.32 to +0.52).
- **The choice on 2019 kept W 128 with λ 0.1, after 10 epochs,** at +0.0155.
  - All 80 scores were above zero, from +0.0036 to +0.0155, where the walks' highest was
    +0.0061.
  - The narrow network beat the wide one, and the strong decay the weak: the four settings'
    20 epochs averaged +0.0127, +0.0109, +0.0097 and +0.0067. W 512 with λ 0.1 was still
    rising at epochs 19 and 20 (+0.0141, +0.0124).
  - From one epoch to the next a score moved by about 0.002, so the epoch kept owes something
    to luck.
- **Measured once on 2020-2021, it failed gate 2.** Each day's correlation with `z`:

  | | 1 min | 5 min | 15 min | 30 min | 60 min |
  | --- | --- | --- | --- | --- | --- |
  | The network | +0.0037 (5.9) | +0.0078 (6.7) | +0.0098 (5.5) | +0.0114 (5.2) | +0.0134 (3.8) |
  | The ridge | +0.0061 (7.7) | +0.0092 (6.6) | +0.0122 (5.5) | +0.0140 (4.7) | +0.0160 (3.6) |
  | The network on shuffled labels | −0.0006 (−0.9) | −0.0004 (−0.5) | +0.0020 (1.5) | +0.0009 (0.5) | +0.0063 (2.2) |

  1. The placebo: passed.
  2. Its margin over the ridge, 15-60 minutes: −0.0025, t −0.85, where 2 was asked: failed.
  3. Its margin over the network on shuffled labels: +0.0085, t +3.90: passed.
  4. Its strongest 1% after their own cost: −1.34 bps at 15 minutes (t −3.0), −0.37 at 30
     (t −0.5) and +0.91 at 60 (t +0.7): failed.

  As section 8 says, variant 2 is written down as failed and is not used.
- **What its page showed, for the next variant to start from.**
  - It forecasts, but less than the ridge: at every horizon; across instruments within each
    minute (+0.0054 to +0.0108, against the ridge's +0.0104 to +0.0134); in stocks at every
    horizon; and in ETFs at 15-60 minutes. A curve saw no more in these inputs than the
    ridge's straight line did.
  - What it learned is each instrument's own. On labels shuffled within each minute, the same
    network learned little (−0.0006 to +0.0063), and nothing across instruments within a
    minute: what a minute's instruments share forecasts little of `z`.
  - Its forecasts are noisier than the ridge's. The slope of `z` on them is 0.10-0.18, against
    the ridge's 0.36-0.55: they move more than `z` follows, for less correlation.
  - Its strongest signals at 30-60 minutes earned more before costs than the ridge's: +2.33
    and +3.70 bps (t 3.2 and 3.0), against +1.89 and +1.89 (t 1.5 and 1.1). Its thresholds,
    set on 2019, let through 2.1-2.4% of 2020-2021's rows, against the ridge's 1.4%. About a
    third of them were ETFs, against a fifth of the ridge's, so they cost less: 2.70-2.79 bps
    a round trip, against 3.30-3.38. After their own cost they made +0.91 bps at 60 minutes,
    with a t of 0.7. Seen after the measurement, this is a hint for a next variant, not a
    result.
  - The runs took longer than the check of speed said: the placebo 2 h 29 min, the choice
    1 h 41 min and the measurement 30 minutes, at W 128 for 10 epochs. The check timed a
    stand-in network's passes forward and back, but not the hand-written Adam step and the
    standardising, about 1 ms more a batch.
- **Next.** Decision 0023 allows eight more variants. The bank of real paths and the
  simulator still wait for a forecast that pays its own costs. Efe decides what variant 3 is.
