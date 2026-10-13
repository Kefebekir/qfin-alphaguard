# 0013: When the plan trades: three orders from 10:30, and days worth 20 USD

- **Status:** accepted
- **Date:** 2026-10-13
- **Corrects:** decision 0012's consequence for the nightly plan's fees

## Context

Before step 8 builds the child orders, Efe asked whether orders every 10 or
15 minutes would earn more than three, and why Guard allows only 3 orders a
stock a day. Measured on the nightly plan's real trades from 2017 to October
2026: 18,521 trades at 100,000 USD, decided at 09:35 New York time and
executed on the stored 1-minute bars. The spreads come from Alpaca's SIP
quotes.

- **The spread by time of day.** Alpaca fills a fractional order at the
  quote, with no price improvement, so half the quoted spread is what it
  pays. On 60 random sessions of 2024–2026 and 20 of the universe's stocks,
  half the spread averaged 3.66 bps at 09:35, 1.79 at 10:30 and 1.10 at
  15:30. At 09:35 it is 1.88 bps wider than at 10:30 (95% interval 1.75 to
  2.02). The cost model's 1 bp holds from mid-morning on.
- **How often to send.** Cost against the 09:35 price, with the spread at
  each order's time:

  | Schedule | Cost | Orders a stock a day |
  | --- | --- | --- |
  | All at 09:35 | 3.85 bps | 1 |
  | 3 orders from 09:35 | 1.23 bps | 3 |
  | 3 orders from 10:30 | 0.47 bps | 3 |
  | Every 15 minutes | 0.33 bps | 24 |
  | Every 10 minutes | 0.34 bps | 36 |

  The number of orders moves the price by fractions of a bp, within noise;
  the time of the first order is what matters.
- **The limit of 3 orders.**
  - With perfect foresight, the best price among the day's 5-minute points
    would save 75 bps a trade, and one order would take it. More orders only
    average the price when nothing is known.
  - With decision 0010's reversal signal (rank IC 0.02), 31 orders a stock a
    day beat 3 equal orders by 1.26 bps (95% interval 0.08 to 2.44): about
    0.40 USD a year at 500 USD. 3 timed orders gained nothing.
  - Trading the signal itself loses after costs (decision 0010).
- **How many days the plan trades.** With the 1 USD minimum of decision
  0012, the plan at 500 USD trades on 214 days a year, with a median trade
  of 2 USD. Alpaca rounds each fee's day total up to the cent, so its fees
  come to 4 USD a year, 0.80% of the account. Decision 0012 called this
  negligible for the nightly plan, which was wrong. Letting a day's trades
  wait until they are worth 20 USD together:

  | Account | Days with trades a year | Fees a year | Return a year against every day |
  | --- | --- | --- | --- |
  | 500 USD | 214 → 72 | 0.80% → 0.29% | +0.34 points (t = 4.8) |
  | 1,000 USD | 232 → 115 | 0.44% → 0.23% | +0.20 points |
  | 2,000 USD | 242 → 157 | 0.23% → 0.16% | +0.06 points |
  | 5,000 USD | 245 → 207 | 0.10% → 0.08% | +0.02 points |

  - **Robustness.** At 500 USD the gain holds in both halves: +0.35 points
    in 2017–2021 and +0.32 in 2022–2026. Volatility stays at 13.1%, and the
    share of the portfolio that differs from the plan goes from 5.1% to 5.8%.
  - **Trading once a week** cut the fees as much (0.20%), but its gain was
    within noise (+0.25 points, t = 1.0). It would also leave four of five
    daily forecasts unused once Phase 2's model forecasts the next day.
  - **A higher minimum trade** of 25 USD left 12% of the account in cash
    and lost 1.3 points a year.

## Decision

- **Child orders:** at most 3, equally spaced from 10:30 (`start_after_open_min`
  60) to 30 minutes before the close (15:30). What a child leaves unfilled
  goes with the next one.
- **A day's trades wait together** until they are worth at least 20 USD
  (`min_day_usd`). So Alpaca's smallest day of fees, 2 cents, stays at most
  0.1% of what is traded. `min_trade_usd` stays 1 USD.
- **Guard's 3 orders a stock a day stay.** Efe expects to raise the limit
  once a model shows a measured gain from more orders; the roadmap keeps the
  question open in Phase 2.

## Consequences

- **The code.** `ExecutionSettings` starts orders at 60 minutes and gains
  `min_day_usd`; the band applies it, and step 8 builds the child orders.
- **The QQQ strategy** trades on most days; its fees at small capital
  (decision 0012) are not changed by this.
- **Found on the way:** a held stock whose prices end stays in the books
  for ever. TWTR was taken private for cash, and VIAC's bars stop after it
  became PARA. The backtest must turn such a stock into cash at its last
  price (step 10).
