# 0006: Rebalance a stock only when its gap beats its own volatility

- **Status:** accepted
- **Date:** 2026-10-09

## Context

The design moved from weekly rebalancing to a nightly plan executed intraday.
That raised the question whether the portfolio formulas must change too. The
system makes three decisions on three time scales:

| Decision | When | From | Formula |
| --- | --- | --- | --- |
| What to hold | nightly | daily returns | Ledoit-Wolf covariance, minimum variance (`plan.py`) |
| Whether a gap is worth trading | once a day, at the open | the plan and the positions | the band (this record) |
| When to trade it | every minute | 1-minute bars | the timing rule (Phase 1 step 8, Phase 2) |

Only the third works intraday, and its formulas already are. The plan stays
daily: on 2025's data the target weight of a stock held on two days in a row
moves by a median of 0.07–0.08 percentage points, mostly noise, while a round
trip costs 6–7 bps (Phase 1 step 6).

To choose the band, 2025 was simulated with the plan of Phase 1 step 7a, on
real data: positions drift with prices, the band decides at the open which
gaps to close, no trade is smaller than 1,000 USD, buys are paid from cash, and
each trade costs 3.2 bps of its value.

| Band | 30,000 USD, 10 stocks | 100,000 USD, 16–29 stocks |
| --- | --- | --- |
| Fixed 2 percentage points | 102 / 18.7 / 1.20% | 54 / 3.6 / 1.64% |
| 25% of the target weight | 100 / 18.6 / 1.27% | 113 / 5.7 / 1.15% |
| Own volatility, k = 5 | 102 / 18.7 / 1.20% | 139 / 6.0 / 0.87% |
| Own volatility, k = 10 | 102 / 18.7 / 1.20% | 121 / 5.6 / 0.94% |
| Own volatility, k = 20 | 89 / 17.0 / 1.64% | 89 / 4.8 / 1.30% |
| Only the 1,000 USD minimum | 102 / 18.7 / 1.20% | 141 / 6.0 / 0.87% |

Each cell: trades in the year / cost in bps of capital / tracking error
against the plan, a year.

- **10,000 USD:** the plan holds 3 stocks at Guard's 25% each, and every band
  gives the same result: 10 trades, 8.0 bps, 0.76%.
- **30,000 USD:** the band hardly matters. The 1,000 USD minimum is 3.3 points
  of the portfolio, more than most daily gaps in a 10% position, so most trades
  come from stocks entering and leaving the top 10, 42 times in 2025: with
  k = 10, 82 of the 102 trades and 16.4 of the 18.7 bps.
- **100,000 USD:** the band matters: from 54 to 141 trades and 3.6 to 6.0 bps
  a year. The narrower the band, the closer the portfolio stays to the plan.

## Decision

Efe chose a band of each stock's own: trade a stock when

    |target weight - current weight| > k * daily volatility * max(target, current)

so a stock's gap counts only when it is larger than k days of that stock's own
price noise (over 2024, NVDA moved 3.3% a day and JNJ 0.95%). k is 10 by
default and is chosen in the full backtest (Phase 1 step 10) against 5, 20 and
the 25% band, by net Sharpe ratio. The 1,000 USD minimum trade applies on top.

## Consequences

- The plan carries each stock's daily volatility (`DailyPlan.volatility`): the
  standard deviation of its daily returns over the last year.
- The band is Efe's code (Phase 1 step 7b). Two more rules were settled with
  it; Efe left the choice to Claude:
  - A stock the plan drops is sold completely, even below 1,000 USD. At
    100,000 USD and k = 10, keeping such remainders would leave a median of 25
    stocks held instead of 15, and the tracking error would be 0.94% instead
    of 0.72%, for 5.6 instead of 6.8 bps a year.
  - Buys are paid from cash and the day's sales. When those fall short, every
    buy is cut by the same factor; nothing is borrowed, as Guard's
    `max_gross_exposure: 1.0` also requires.
- At 30,000 USD and below, swaps cost more than any band saves. Keeping a held
  stock unless its replacement is clearly better changes the plan, not the
  band; it is measured in the step 10 backtest.
- When Phase 2 adds forecasts, targets will move more; then costs may belong
  inside the optimiser rather than in a band.
