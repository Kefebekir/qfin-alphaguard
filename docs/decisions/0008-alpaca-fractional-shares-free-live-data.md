# 0008: Trade on Alpaca with fractional shares, with free live data

- **Status:** accepted
- **Date:** 2026-10-10
- **Replaces:** decision 0002 (Interactive Brokers)

## Context

Efe plans to start trading real money with about 500 USD. Decision 0002 chose
IBKR, and its minimum commission of 0.35 USD an order shaped the design: no
trade under 1,000 USD, and at most `capital / 3,000 USD` stocks held, so that
each can be built in three such orders.

Simulated on real data from January 2017 to October 2026, with the band
trading to its edge (decision 0007). IBKR is modelled as designed: 3.2 bps a
trade and the 1,000 USD minimum. The commission-free broker has fractional
shares, takes trades from 1 USD and costs 1.45 bps a trade: the 1.3 bps
against the decision price measured in step 6, plus SEC and FINRA fees on
sales.

| Account | IBKR as designed | Commission-free, fractional shares |
| --- | --- | --- |
| 2,000 USD | nothing bought: a 25% position is 500 USD, under the minimum | 20 stocks, 98% invested, Sharpe 0.82 |
| 5,000 USD | 1 stock, 23% invested, Sharpe 0.28 | the same |
| 10,000 USD | 3 stocks, 67% invested, Sharpe 0.56 | the same |
| 30,000 USD | 10 stocks, Sharpe 0.87 | the same |
| 100,000 USD | 18 stocks, Sharpe 0.79 | the same |

With fills half as bad again (2.2 bps a trade), the commission-free Sharpe
ratio stays 0.82.

Alpaca, checked in October 2026:

- **Trading:** no commission on US stocks. Fractional shares come with market
  and limit day orders, from 1 USD.
- **Accounts:** paper accounts are free and open at once. A margin account
  needs 2,000 USD (FINRA). UK residents can open an account, but not an ISA.
- **How it earns:** payment for order flow; orders go to market makers.
  FINRA faulted its disclosures of these payments in March 2025.
- **Our paper account:** opened and checked, read-only, on 9 October 2026. It
  is a margin account with 4× buying power and short selling enabled.

Market data:

- **History.** EODHD's 1-minute bars are the consolidated tape. Alpaca's SIP
  bars match them minute by minute: a median close gap of 0.00 bps and a
  volume ratio of 1.000, over about 2,700 minutes on each of three days in
  2018, 2025 and 2026. Alpaca's free SIP history, from 2016, has the minutes
  EODHD lacks:
  - TSLA from July 2023 to May 2024, and NVDA in June and July 2024;
  - BKNG before 2018, AABA under YHOO, CHK_old under CHK, and DWDP.

  Spot checks found every code but VIAC in 2022. If every day holds, the
  stock-days without minutes fall from 1.7% to about 0.1%.
- **Live.** Both affordable feeds come from a single exchange. EODHD's
  WebSocket, included in the current plan, is Cboe EDGX only; Alpaca's free
  feed is IEX only. Alpaca's consolidated live feed costs 99 USD a month,
  more than twice a 500 USD account each year. Both single-exchange feeds were
  measured against SIP on 9 October 2026, over the last two hours of the
  session and 50 stocks:

| | EDGX (EODHD) | IEX (Alpaca, free) |
| --- | --- | --- |
| Minutes with a bar | 93% | 88% |
| Close gap, median (95th percentile) | 0.70 (3.6) bps | 0.70 (4.4) bps |
| Volume, share of SIP | 2.4% | 5.0% |
| Return correlation over 1 / 5 / 15 / 30 minutes | 0.90 / 0.97 / 0.99 / 0.99 | 0.83 / 0.94 / 0.97 / 0.98 |

## Decision

- **Broker.** Alpaca: paper trading now, real money from about 500 USD.
  Decision 0002 is replaced.
- **History.** Historical minutes stay with EODHD, and Alpaca's SIP history
  fills its holes.
- **Live prices.** They come from a free single-exchange feed. The intraday
  model's features use horizons of 5 minutes or more, where these feeds
  follow the consolidated tape closely. Paper trading measures the difference
  again over many days, and the paid feed waits until the account makes it
  worth its cost.

## Consequences

- **Fractional shares.** The event types, the simulated broker and the band
  all assume whole shares; their design for fractions comes in its own
  record.
- **Costs and minimum trade.** `min_trade_usd` and the cost model follow the
  broker: no commission and a minimum trade of a few dollars at Alpaca. The
  cap on stocks held (`positions_for`) then no longer binds.
- **Margin.** Guard's `max_gross_exposure: 1.0` and `allow_short: false` keep
  the system from using the paper account's margin and short selling
  (decision 0009).
- **Tax.** Gains are taxable, since there is no ISA. At the starting capital
  they stay within the UK's yearly allowances.
- **Fixed fees.** At small capital they matter more than anything else: 30
  USD a month is 72% a year of 500 USD. The first real-money period tests
  whether the live system behaves like the backtest; nobody expects it to pay
  for its data.
