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
   year. N is 30 to start (50 since 9 October 2026; see the update below).
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

## Update, 9 October 2026: N = 50, and how the ranking works

Efe set N to 50 after checking everything it touches:

| Area | Check | Result |
| --- | --- | --- |
| Liquidity | dollar volume of the 50th stock, 2016–2026 | 512 million to 1.3 billion USD a day; our orders are negligible |
| Turnover | names that change each January | 6–11 of 50 |
| Live data, IBKR | at least 100 market data lines | 50 plus held positions fit |
| Live data, EODHD | WebSocket for 50 tickers | full at 50; ask EODHD for more if Trading 212 is chosen |
| IBKR pacing | 60 new real-time bar requests per 10 minutes | 50 at start-up fits; after a reconnect, resubscribe gradually |
| Intraday history | 1-minute bars since 2015 | about 58 million rows: download once, then add each day |
| Guard | `max_orders_per_min: 10` | the engine spreads orders near the deadline; depends on positions held, not on N |

Settled while building the ranking (Phase 1, step 2c):

- **Dollar volume at each day's traded price and share count:**
  close / adjustment × volume / split_factor. Adjusted prices would let later
  dividends decide earlier rankings: on 2 January 2015, dividends paid since
  had scaled prices down by a median 16% and up to 37%. The fix changes 2–6
  of the 50 names a year.
- **At least 126 trading days in the previous year**, so a new listing's busy
  first weeks cannot buy it a place.
- **One share class per company:** of two stocks whose daily returns differ by
  less than 0.2% on a typical day, only the more traded is kept. GOOG and
  GOOGL differ by 0.06–0.15%; the closest different companies (KO and PEP,
  MA and V) by 0.30% or more. A correlation threshold failed here: one bad
  GOOG price in July 2021 pulled that year's correlation to 0.978. On
  2016–2026 the rule drops GOOG every year and nothing else.
- **Members at the end of the previous year:** the universe is chosen from
  the members on the previous year's last session, so it is known the evening
  before the year's first session, when its first daily plan is made (Phase 1,
  step 7a). On 2016–2026 it gives the same 50 names as the members of the
  year's first session.
