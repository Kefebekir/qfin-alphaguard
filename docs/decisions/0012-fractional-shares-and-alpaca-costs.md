# 0012: Fractional shares and Alpaca's costs in the backtest

- **Status:** accepted
- **Date:** 2026-10-12

## Context

Decision 0008 moved trading to Alpaca because of its fractional shares: at
500 USD, a 5% position is 25 USD, less than one share of most stocks in the
universe. The code still assumes whole shares (the event types, the books,
the simulated broker and the band) and IBKR's costs: 0.0035 USD a share, at
least 0.35 USD an order, and no trade under 1,000 USD.

Alpaca's rules, checked in October 2026:

- **Fractional orders:** market, limit, stop and stop-limit orders, for the
  day only. Quantities take up to 9 decimals, and a buy is at least 1 USD.
  Fractions cannot be sold short: every fractional sale is from shares held.
- **Splits:** since August 2021 Alpaca keeps the fractions a split leaves,
  instead of paying them in cash. A reverse split may still be settled in
  cash for what does not divide.
- **Fees** (fee schedule of 1 October 2026): no commission. On sales, the SEC
  fee is 0.0000206 times the value sold (20.60 USD a million since 4 April
  2026). FINRA's activity fee is 0 at Alpaca. The CAT fee is 0.000003 USD a
  share, bought or sold. Each fee is totalled per account and day, on the
  exact fractional quantities, and the total rounded up to the cent.
- **Paper accounts** apply no splits or dividends.

## Decision

- **Quantities are floats with at most 9 decimals**, Alpaca's precision.
  - The event types reject more decimals, so whatever computes a quantity
    rounds it on purpose (`round_shares` in events.py).
  - The books round every change, so 0.1 + 0.2 shares make 0.3, and a
    position sold to nothing disappears.
  - Not Decimal: prices and cash are floats, mixing the two raises errors,
    and numpy and polars do not use it. Not whole billionths of a share:
    exact, but every quantity becomes a huge number to convert at each edge.
- **Splits keep fractions:** a position is multiplied by the ratio. Reverse
  splits that Alpaca settles in cash are rare among the most liquid stocks,
  and are not modelled.
- **Costs in the simulated broker:**
  - No commission.
  - The SEC fee on sales and the CAT fee on every fill, each day's total of
    each rounded up to the cent. A fill pays what it adds to the rounded
    total, so a day's fills add up to what Alpaca charges.
  - The half spread stays at 1 bp; step 6 measured 1.3 bps against the
    decision price. No currency conversion: the account holds dollars, and
    converting a deposit happens once.
- **Minimum trade 1 USD,** Alpaca's smallest buy. `positions_for` no longer
  limits the stocks held.
- **Partial fills:** an order still takes at most `participation` of a bar's
  volume, but in fractions, not rounded down to whole shares.
- **Short sales stay whole shares,** since Alpaca does not short fractions.
  The plan never shorts (Guard's `allow_short: false`); the QQQ strategy
  sizes its shorts in whole shares (Phase 1b).

## Consequences

- **Two pull requests:**
  1. Claude: the event types, the books and the engine take fractional
     quantities. Whole shares still work, so nothing else changes yet.
  2. Efe: the simulated broker's costs and fills, and the band's trades in
     fractions, against tests Claude writes first. The 1 USD minimum comes
     with them.
- **Fees at small capital:** a day with a sale costs at least 0.01 USD of SEC
  fee, and a day with a fill 0.01 USD of CAT fee. For the nightly plan, which
  trades on few days, this is negligible. For the QQQ strategy, which trades
  on most days, it is up to 0.02 USD a day and 5 USD a year: 1% of 500 USD.
- **Paper trading** shows no splits or dividends, so its comparison with the
  backtest leaves them out.
