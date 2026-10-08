# 0005: Research on the S&P 500 as it was each day; trade its most liquid stocks

- **Status:** accepted
- **Date:** 2026-10-08
- **Replaces:** the "8–10 stocks" part of 0004. Its other points stand: US
  single stocks, long only, no leverage.

## Context

0004 set the universe at 8–10 stocks without a reason for the number, and the
30 tickers in `Config` were picked with today's knowledge. Measured on daily
data from 2 January 2015 to 7 October 2026:

| Equal-weight buy-and-hold | Return a year |
| --- | --- |
| The 30 tickers in `Config` | 29.4% |
| The same without NVDA (up about 490 times) | 13.7% |
| The 30 Dow stocks of 2 January 2015 | 12.4% |
| SPY, for scale | 13.8% |

Picking stocks with hindsight adds about 17 points a year to a backtest, far
more than any choice about how many stocks to hold.

More stocks lower volatility only a little (equal weight, 10 → 30 stocks:
17.7% → 16.8% a year), but they give the Phase 2 models more data, and a
forecast with real skill earns more over more independent bets.

Free data cannot support an honest history. Of the 771 tickers in the S&P 500
at some point from 2015 to 2026, 276 left the index; Yahoo has prices for
99.7% of the member-days of current members but only 40.4% of those of
departed ones.

How many positions the account can hold depends on capital: IBKR charges at
least about 0.35 USD an order and its API trades whole shares only.

## Decision

1. **Research and ML universe:** the S&P 500 members on each day, from the
   point-in-time history of [fja05680/sp500](https://github.com/fja05680/sp500)
   (MIT), copied into the package at a fixed commit.
2. **Prices:** EODHD end-of-day data, which keeps stocks that have left the
   exchange. Checked on the free plan before subscribing.
3. **Trading universe:** every 1 January, the N members with the highest
   average daily dollar volume over the previous 12 months, fixed for the
   year. N is 30 to start.
4. **Positions held:** set by capital, in the daily plan (Phase 1, step 7).
   Capital is not fixed yet, so backtests report results at several levels.
5. **Broker:** IBKR stays (0002) for now. Fractional-share brokers such as
   Trading 212 go into the cost model and are decided, with backtest numbers,
   before Phase 3.

## Consequences

- The nightly job downloads about 770 tickers instead of 30 and needs an
  EODHD API key: in the git-ignored `.env` locally, as a secret on AWS.
- EODHD data is licensed for personal use, so raw prices never go into this
  public repository.
- The membership history ends at its latest snapshot. It is updated by hand,
  and index changes after that date are unknown until then.
- The history is kept by one person. It matches its own full snapshot file on
  all 2,720 dates, but it is not an official S&P record.
