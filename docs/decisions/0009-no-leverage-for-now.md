# 0009: No leverage for now, and a gate for later

- **Status:** accepted
- **Date:** 2026-10-10

## Context

Efe asked whether leverage would earn more, perhaps with several models at
different leverage levels. It was measured on the commission-free plan of
decision 0008: 500 USD, the band trading to its edge, borrowing at 6% a year.

**Constant leverage:**

| Leverage | 2017–2021 | 2022–2026 |
| --- | --- | --- |
| 1× | 14.0% a year, worst drawdown −24% | 6.7% a year, −19% |
| 1.5× | 17.1%, −35% | 6.4%, −29% |
| 2× | 19.8%, −44% | 5.7%, −38% |
| 3× | 23.2%, −61% | 3.4%, −54% |

Leverage pays only while the strategy earns well above the borrowing rate. In
2022–2026 it earned 6.7% against 6%, and every level of leverage earned less
than none. The Sharpe ratio falls with leverage in both periods.

**Several accounts at different leverage hold the same stocks.** Four
accounts at 1×, 2×, 3× and 4× did what one account at 2.5× did: a Sharpe
ratio of 0.51 against 0.55, and a worst drawdown of −54% against −53%.
Leverage scales a forecast; it does not need a model of its own.

**Volatility targeting** sets `exposure = min(cap, target / forecast
volatility)`, so the account holds less when markets are rough and more when
they are calm. The test covered:

- three forecasts: the plan's Ledoit-Wolf covariance, an EWMA covariance, and
  the EWMA of the plan's own returns (both EWMAs with a 21-day half-life);
- targets of 10, 12 and 15%;
- caps of 1×, 1.5× and 2×.

Settings were chosen on 2017–2021 and judged on 2022–2026:

- **Forecasts.** The plan's 252-day covariance forecasts the next month's
  volatility poorly (correlation 0.16); the EWMA forecasts do better (0.37 and
  0.39).
- **Without leverage.** The best setting of 2017–2021 (EWMA of own returns,
  10%) changed 2022–2026 as follows:
  - worst drawdown −19% → −14%;
  - Sharpe ratio 0.62 → 0.66;
  - growth 6.7% → 5.8% a year, because less was invested.
- **With leverage allowed** (a cap of 1.5× or 2×), no setting grew more than
  0.2 points a year faster than holding 1× in 2022–2026, and every one had a
  larger drawdown than the same setting without leverage.

## Decision

- **No leverage.** `max_gross_exposure: 1.0` in `guard.yaml` stays. A margin
  account needs 2,000 USD anyway (decision 0008).
- **Volatility targeting is not adopted yet.** It cuts drawdowns, but also
  growth, and its Sharpe gain is small and unproven. The step 10 backtest
  reports it as a variant, and Phase 2's volatility forecasts may make it
  worth more.
- **Leverage is reconsidered**, by hand, only when all of these hold:
  - the account holds at least 2,000 USD;
  - six months of live results fall inside the expected range (Phase 3b);
  - the live return is at least 5 points a year above the borrowing rate.

  Even then it goes no further than 1.5×, through volatility targeting, with
  a drawdown stop in Guard.

## Consequences

- Efe likes leverage. This record is a gate, not a ban: the numbers that
  would open it are written above.
- The step 10 report shows the leverage and volatility-targeting variants net
  of borrowing costs, next to the plan without them.
