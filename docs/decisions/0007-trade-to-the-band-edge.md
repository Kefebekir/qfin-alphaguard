# 0007: Trade a gap only as far as the band's edge

- **Status:** accepted
- **Date:** 2026-10-09
- **Changes:** decision 0006, which traded a gap beyond the band to the target

## Context

Decision 0006 leaves a stock alone while its gap is within its own band
(`band_k` days of its daily volatility times the larger of its weights) and
trades it to the target once the gap is larger. With costs proportional to
the trade, the best policy keeps such a no-trade region but, outside it,
trades only to the region's boundary, not to its centre (Davis and Norman,
1990; Leland, 2000): the last part of the gap is within the stock's own noise
and costs as much to trade as the rest.

Before step 8, Efe asked whether any formula of step 7 should change. Three
were tested on real data from January 2017 to October 2026 (9.7 years), with
the plans of step 7a. The simulation follows the band study of 0006:
positions drift with prices, trades under 1,000 USD wait, dropped stocks are
sold completely, buys are paid from cash, and each trade costs 3.2 bps.

| k = 10 | 30,000 USD: target → edge | 100,000 USD: target → edge |
| --- | --- | --- |
| Trades a year | 139 → 115 | 269 → 185 |
| Cost, bps a year | 24.7 → 19.9 | 13.3 → 8.6 |
| Realised volatility | 13.31% → 12.73% | 13.31% → 12.91% |
| Return after costs | 10.39% → 11.12% | 10.05% → 10.26% |
| Sharpe ratio | 0.78 → **0.87** | 0.76 → 0.79 |
| 95% interval of the Sharpe change | +0.03 to +0.16 | −0.02 to +0.09 |
| Invested, on average | 96.9% → 94.5% | 98.2% → 96.5% |

Intervals come from a block bootstrap of daily returns (20-day blocks). Part
of the lower volatility comes from holding 2 points more cash, since a new
stock starts at its band's edge, below its target; the Sharpe ratio does not
depend on that.

Run again with the band's own code (`trades_at_open`: whole shares, real
cash, the previous close as the price) over the same years, the edge gave
0.76 → 0.84 at 30,000 USD (95% interval +0.01 to +0.13) and 0.79 → 0.82 at
100,000 USD (−0.02 to +0.07), with 22% and 26% fewer trades. With a commission-free broker's costs (1.45 bps a trade, no
minimum, every stock the optimiser wants), the edge also did better: 0.79
→ 0.82 at both capital levels.

How wide the band is matters less than where a trade stops. Trading to the
target with k = 5, 10 or 20 changed the Sharpe ratio by at most 0.05; at
100,000 USD, k = 20 to the target (0.81) did slightly better than k = 10 to
the edge.

Also tested, and not adopted:

- **Covariance estimators.** Sample covariance, Ledoit-Wolf towards constant
  correlation, nonlinear shrinkage (Ledoit and Wolf, 2020), EWMA with a
  63-day half-life and a single-index model were compared by the realised
  volatility of the portfolios they build, out of sample. All came within
  0.25 points of Ledoit-Wolf's 13.70% for the top-10 plan. EWMA was lowest
  (13.45%) but traded 50% more; the single-index model traded 60% less but
  was riskier. Ledoit-Wolf stays.
- **A buffer for the top 10.** Keeping a held stock while it ranks within
  the top 12 halved the trades at 30,000 USD (139 → 71 a year, 24.7 → 11.6
  bps) without changing the risk, but the Sharpe ratio did not rise (0.74
  against 0.78, within noise). It stays a step 10 experiment.

## Decision

A gap beyond the band is traded only as far as the band's edge: the target
minus the band for a stock below its target, plus the band for one above it
(`rebalance._change`). Efe chose this on the numbers above; the code is his.

## Consequences

- Positions rest at their band's edge, not at the target, and the tracking
  error to the plan rises (1.10% → 1.64% at 30,000 USD). The plan's daily
  moves are mostly noise (0006), and following them less closely cost no
  return here.
- A new stock starts at its band's edge, 80–90% of its target for a daily
  volatility of 1–2%.
- `band_k` is still chosen in the step 10 backtest, now with the edge rule.
- The evidence uses daily closing prices and one cost per trade, and 14
  variants were compared, so one could look better by chance. The step 10
  backtest, on 1-minute bars, has to confirm it.
