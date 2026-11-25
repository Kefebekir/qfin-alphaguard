# 0028: KUTBIT's third variant: the sector's move, given to the ridge

- **Status:** accepted
- **Date:** 2026-11-25 (agreed with Efe on 25 November)
- **Builds on:** decision 0023 (variants and placebos), 0024 (the inputs, the target, the skill
  test and its ridge, the costs) and 0027 (variant 2 and its result).
- **Variant:** 3 of the 10 that 0023 allows.

## Context

- **Variant 2 failed** (0027's result of 23 November). A network on the skill test's 971
  inputs forecast at every horizon, but less than the ridge: its day-by-day margin was −0.0025,
  at a t of −0.85. The limit looks like the information in the inputs, not the model.
- **What the ridge leans on.** Fitted on 2016-2018 only, each input's share of its forecast's
  covariance with `z`:
  - at 1 minute, 60% is the instrument's own move apart from QQQ, reversed. Of that, 40% is the
    last minute alone, and part of it is probably the bounce between bid and ask.
  - At 15 minutes, that reversal is 35%, the instrument's volatility 22% and QQQ's moves 16%,
    mostly those of 30 to 120 minutes before.
  - At 60 minutes, volatility is 33%, the time of day 22% and where the price stands in the day
    17%.
  - Headline counts carry about 1% at every horizon.
- **Each instrument sees only itself and QQQ.** QQQ's moves, the only other instrument's, carry
  5-16% of the ridge's forecast. Nine of the sixteen ETFs are sector funds (XLB, XLE, XLF, XLI,
  XLK, XLP, XLU, XLV and XLY), and most stocks belong to one of them, but no instrument sees
  their moves.
- **A paper's idea.** Balestriero, LeCun and Varaku (arXiv 2611.24778, November 2026) answer
  what would have happened otherwise by reusing a residual: what a hidden disturbance did, seen
  in what happened. KUTBIT, a price-taker, needs no such model for its own trades, since every
  action's outcome is known from the prices. The spirit carries over to forecasting: a shock to
  a sector may show in one of its instruments before the others.
- **Efe's choice, 25 November:** the sector's move apart from QQQ as variant 3, measured with
  the ridge first.

## Decision

**1. What it is.** The skill test's ridge, given one more channel: the recent move apart from
QQQ of the instrument's sector. It asks whether a sector's move forecasts the next move of an
instrument in it, beyond what the instrument's own chart and QQQ's say. The rows, the target
and everything else are the skill test's, so that only the new channel differs.

**2. The match, made on 2016-2018 only.**
- Every instrument but QQQ and the nine sector ETFs is matched with one sector ETF: the one
  whose `apart`, its move less beta times QQQ's, correlates most with its own, minute by
  minute, over the minutes of 2016-2018 when both traded. `r` is that correlation.
- An instrument with fewer than 100 sessions in 2016-2018 has no match.
- Found on 2016-2018, 69 instruments matched:
  - XLE: CVX 0.75, XOM 0.71, SLB 0.68, HAL 0.62, KMI 0.53 and CHK_old 0.38; AABA 0.07 and FTR
    0.06 only weakly;
  - XLF: JPM 0.68, C 0.63, BAC 0.60, WFC 0.59, BRK-B 0.56 and GS 0.51;
  - XLV: JNJ 0.53, PFE 0.53, MRK 0.52, UNH 0.47 and seven more at 0.33-0.44;
  - XLP: PG 0.60, KO 0.54, WMT 0.39, COST 0.38, CVS 0.34, VZ 0.27 and T 0.23;
  - XLI: SPY 0.62, DIA 0.60, BA 0.46, UNP 0.44, GE 0.39, IWM 0.37 and four below 0.3;
  - XLY: HD 0.45, DIS 0.34 and ten more at 0.10-0.30;
  - XLK: twelve stocks, most of them in technology, at 0.10-0.24, since QQQ already holds most
    of that sector;
  - XLB: DWDP 0.74; XLU: GLD 0.07 and TLT 0.07.
- Thirty-five instruments enter the universe after 2018 and have no match: 10.1% of 2019's
  rows, 10.6% of 2020's and 13.5% of 2021's. Their channel is zero, and they count for no
  sector.

**3. The channel, `sector`.** At minute t, from the minute before:
- for a matched instrument, r times its sector ETF's `apart` at minute t − 1;
- for a sector ETF, the mean of r times `apart` at minute t − 1 over its matched instruments
  that traded then;
- zero where that minute had no trade, in a session's first minute, and for an instrument
  without a match.

Like every channel it is clipped at ±10 and seen over the last 120 minutes, so the inputs grow
from 971 to 1,091.
- **Why scaled by r.** A strong match passes on most of its sector's move and a weak one almost
  none, without a threshold to choose.
- **Why a minute old.** A minute's close is its last trade, and a stock's last trade can come
  before its sector ETF's. A lead of the ETF that is only that gap would look like a forecast
  and pay nothing. A minute's delay keeps it out.

**4. The ridge, as the skill test's** (0024, section 9).
- It is standardised on the rows it learns from.
- For each horizon, a penalty of 0.001, 0.01, 0.1, 1 or 10 on the inputs' correlation matrix,
  chosen on 2019 by the correlation of the forecasts of the fit on 2016-2018.
- It is fitted again on 2016-2019 with the penalties kept.
- The thresholds of its strongest 10% and 1% are set on that fit's forecasts of 2019.

**5. The measures, once, on 2020-2021,** as 0027's section 5: for the new ridge, the skill
test's ridge, the new ridge on shuffled labels, the new ridge with the wrong sector, and the two
known patterns.

**6. The skill test's ridge beside it.** The harness fits it again, and its measures must equal
those of 0024's result of 22 November; if they do not, the run stops.

**7. A signal's own cost,** as 0027's section 7.

**8. The gates.**
1. **The placebo** runs first, through the same code: every session's prices replaced by the
   skill test's random walk, with the match and its r kept from the real prices. The choice is
   made on its 2019 and the measures taken on its 2020-2021. No forecaster may reach |t| 3 at
   any horizon in any measure. The real run waits for it.
2. **The ridge.** The new ridge's day-by-day margin over the skill test's ridge in each day's
   correlation, averaged over 15, 30 and 60 minutes, must have a t of at least 2.
3. **Shuffled labels** (0023, section 6). The same ridge, with the penalties kept, is fitted on
   2016-2019 with each minute's `z` shuffled between the instruments of that minute. The new
   ridge's margin over it, measured the same way, must have a t of at least 2.
4. **Its cost.** At one horizon of 15, 30 and 60 minutes at least, its strongest 1% must earn
   more than their own cost, with a t, clustered by day, of at least 2.

- **Shown, not a gate: the wrong sector.** The same ridge, each instrument given instead the
  sector ETF least related to it on 2016-2018, the one whose correlation with it is nearest zero,
  still scaled by its own r. A sector ETF is given the instruments of the sector least related
  to it. If the right sector does no better than the wrong one, the gain is not the sector's.
- Failing gate 2 or 3, variant 3 failed, and it is written down as it is.
- Passing gates 2 and 3 but not 4, the ridge is better with its sector, but no forecast pays
  yet. Efe decides; 0027's network on the 1,091 inputs could be a next variant.
- Passing all four, the bank of real paths and the simulator come next.

**9. The runs and their records.**
- Efe runs them, in this order: the channel's build; the placebo; the measurement, once.
- They run on the CPU, in minutes.
- Each record keeps the SHA-256 of the code it ran on. The measurement waits for a placebo that
  passed on today's code, and it runs only once.
- A page shows each run.

**10. Who writes what.**
- Claude: the match, the channel and its check, the harness, the controls, the measures, the
  tests and the page.
- Efe: every run, and what comes after.

## Consequences

- Variant 3 of 10; seven are left after it.
- Pull requests, in order: this record; the match and the channel, built and checked; the
  harness. Then Efe runs them.
- No promise is made. A sector and its stocks move together within a minute, but a lead of one
  over another at this scale is what traders take first, and it may be small or gone.
- Even passing every gate, Efe's goal asks about three times the ridge's edge (0027's context).
