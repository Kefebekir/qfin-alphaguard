# 0015: The backtest: what it runs, what it reports, what it found

- **Status:** accepted
- **Date:** 2026-10-13

## Context

Gate M1 asks for one command that produces the same report from the same
inputs every time, with results after costs compared with baselines. Step 10
builds it, `qfin backtest`. Running the nightly plan over the stored data
from 2016 raised four questions the earlier steps had left open.

- **Odd corporate actions.** `data/corporate.py` stops on an adjustment that
  is neither a split nor a dividend. EODHD's daily bars hold 33 of them:
  - 24 are NVDA's rounding from January 2015 to March 2016, when its adjusted
    price was under 1 USD;
  - 9 are mergers, spin-offs and special dividends of eight other stocks,
    such as CHTR's and JCI's mergers in 2016.

  None falls on a day the backtest can hold the stock.
- **Held stocks whose bars end.** TWTR was bought for cash on 27 October
  2022; VIAC's minutes end on 16 February 2022, when its code became PARA, and
  DWDP's on 31 May 2019, when it became DD. A position in them could be
  neither sold nor valued again (decision 0013).
- **The S&P 500 baseline.** The goal is to beat an S&P 500 UCITS ETF, but
  the store held only index members.
- **Stocks without minutes.** The daily bars of VIAC and DWDP go on after
  their minutes end, so the plan can hold a stock the backtest cannot trade.

## Decision

**What runs.**
- The days run from the first session with a year of daily returns before
  it, 5 January 2016, to the last with both daily bars and minutes.
- One plan a day, from the daily bars before it. Above a few dozen dollars
  the capital does not change a plan, so every capital shares them.
- The stored minutes of each year's universe, and of the stocks that left
  it until the end of January; EqualSpaced, Guard (decision 0014), and the
  simulated broker at Alpaca's costs.
- Splits and dividends only of the stocks the backtest can hold, on the days
  it can hold them. The odd adjustments above therefore never stop it, and an
  odd one on a day it can hold a stock still would.
- A **Delisting** on the first session a stock has no minutes after having
  had them. A stock still held then is paid out at its last price. A dividend
  of the same day is paid first.
- SPY's daily bars are stored with the members' by `qfin ingest`, never as a
  member.

**What it reports**, for each capital (500, 5,000 and 30,000 USD by default):

| Measure | How |
| --- | --- |
| Return a year | compounded, 252 sessions a year |
| Volatility | the daily returns' standard deviation, times √252 |
| Sharpe ratio | their mean over that deviation, times √252; no risk-free rate |
| Maximum drawdown | the deepest fall from an earlier high |
| Winning days, worst day | the share of sessions with a gain; the worst session |
| Turnover | the value of every fill over the account's average value, a year |
| Costs | the fees as charged, and the cost model's half spread of 1 bp on every fill's value, an estimate; together, a year, over the average account |
| Guard | its events by rule and action |
| Data | delistings, and the stock-days the plan held a stock without minutes |

**Two baselines**, over the same sessions, before costs, with dividends:
- **The same stocks at equal weight:** each year's trading universe in equal
  parts, bought at the close before the year's first session and held
  through the year. A stock whose bars end is held as cash at its last close.
- **SPY:** bought at the close before the first session and held.

**The same inputs give the same report.** It holds no time or place of its
making, and each run's fingerprint is a hash of its whole log.

## Results

From 5 January 2016 to 8 October 2026, 2,706 sessions. The report is in
`data/backtests/`, outside the repository, like the data it comes from.

| | A year | Volatility | Sharpe | Max drawdown | Worst day |
| --- | --- | --- | --- | --- | --- |
| The plan, 500 USD | +10.6% | 12.8% | 0.85 | −23.4% | −7.7% |
| The plan, 5,000 USD | +10.7% | 12.8% | 0.86 | −23.3% | −7.7% |
| The plan, 30,000 USD | +10.8% | 12.8% | 0.86 | −23.3% | −7.7% |
| The same stocks, equal weight | +17.5% | 19.3% | 0.93 | −31.7% | −12.3% |
| SPY | +15.2% | 17.7% | 0.89 | −33.7% | −10.9% |

- **The plan loses to both baselines.** It takes about a quarter less risk
  than SPY and falls a third less at its worst, but it earns 4.5 points a
  year less. Its Sharpe ratio, 0.85 to 0.86, is below SPY's 0.89 and equal
  weight's 0.93, so even at the same risk it would earn less. Holding the
  least-variance mix of these stocks has no edge over simply holding them.
- **Costs are not the reason.** They come to 0.20% a year at 500 USD, 0.08%
  at 5,000 and 0.05% at 30,000: fees as charged, plus the spread estimate.
  Turnover is 3.4 to 3.8 times the account a year, buys and sells together.
- **Guard.** Over the ten years, the daily loss limit dropped 513 to 828
  orders, and the cash rule cut 179 to 616 buys and dropped 25 to 40.
  Without Guard, the plan earned 10.63% and 10.74% a year at 500 and
  5,000 USD, against 10.63% and 10.73% with it, and the account closed
  days with negative cash, down to −3.00 USD: Guard costs almost nothing.
- **Data.** There were 74 delistings, 2 of them of stocks still held. The plan
  held a stock without minutes on 72 of its 54,912 stock-days.
- **The same inputs, the same report.** The run was made twice, before and
  after SPY's bars were added to the store, which changes nothing the plan
  trades: the three fingerprints are the same.

## Consequences

- **Gate M1's two conditions hold.** One command gives the same report from
  the same inputs, and it compares the results after costs with SPY and with
  the same stocks at equal weight. Efe decides whether the gate is passed.
  Phase 1 items still open:
  - the map of renamed tickers;
  - a faster nightly download, and each day's minutes in the nightly job;
  - the buffer of decision 0007 and decision 0013's schedule, to measure on
    this backtest;
  - the report's variants with volatility targeting and leverage;
  - Alpaca's 1 USD minimum in the simulated broker;
  - what to do with a broken bar such as LUMN's of 12 October 2026, which
    stopped the nightly ingest.
- **A full run** of 2016–2026 at three capitals takes about 15 minutes and
  2 GB of memory on Efe's machine.
- **SPY reaches the nightly AWS job's file** with the next image pushed to ECR.
