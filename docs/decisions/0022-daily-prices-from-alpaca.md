# 0022: Alpaca's daily bars continue EODHD's history

- **Status:** accepted
- **Date:** 2026-11-13
- **Builds on:** decision 0008 (Alpaca and its free plan) and 0016 (a broken bar is
  repaired); replaces EODHD in `qfin ingest`.

## Context

- **The nightly prices stopped.** EODHD's subscription ended on 11 November 2026. The
  nightly job of 12 and 13 November stopped a minute after starting, with EODHD's HTTP 401.
  The last good file was written on 11 November at 07:08 UTC, with bars to 10 November.
  Nothing alerted: the gap was found when N5's and N6's records stopped at 9 November.
- **What needs the prices.** N5's and N6's forward records score each day from them, and
  `qfin plan` and `qfin backtest` read them. The live engine does not: it takes its bars
  from Alpaca.
- **The choices.**
  - Renew EODHD. Efe chose not to.
  - Download everything again from Alpaca. That would change every past return the
    research read. Alpaca's history also starts in 2016 and does not keep EODHD's earlier
    companies under a reused ticker.
  - Keep EODHD's history as it is and continue it with Alpaca's bars. **This one.**

## Decision

**1. EODHD's last file is frozen.**
- It is kept as `data/raw/prices_eodhd.parquet`, locally and at the same key in S3, and is
  never written again.
- It is a copy of `data/raw/prices.parquet` as of 11 November 2026 (S3 version
  `N43zABzvDN1xFmTI5rcb69UyoGEQEGdS`, SHA-256 `5a2ba0cc269c5c4819ff…`).
- It holds 1,959,195 bars from 2 January 2015 to 10 November 2026.

**2. Each night, `qfin ingest` continues it with Alpaca** (`ingest.continue_prices`).
- **Which tickers continue:**
  - every ticker with a bar on the frozen file's last day, except EODHD's earlier companies
    (`_old`), which no longer trade;
  - every index member since then;
  - a new member with nothing stored, which comes with its whole history from Alpaca.
- **Prices:** daily bars as traded, from the consolidated tape. A session counts once it has
  been over for 15 minutes, when Alpaca's free plan serves it.
- **Splits, dividends and spin-offs** come from Alpaca's corporate actions and are applied as
  EODHD applied the earlier ones:
  - a split of ratio r divides the earlier prices by r and multiplies their volume and
    `split_factor` by r;
  - a dividend D multiplies the earlier prices by 1 − D / P, where P is the traded close of
    the session before the ex-date;
  - a spin-off is stored as a split, as EODHD stored GE's of April 2024. Its ratio is the
    fall in Alpaca's spin-off adjustment across the ex-date. Spin-offs of warrants, which
    Alpaca does not adjust for, are left out.
- **Why not Alpaca's adjusted prices:** Alpaca rounds them to the cent. On a 10 USD stock
  that makes the dividend factor jitter by up to 0.05% a day. `corporate.py` would read each
  jitter above 0.01% as a dividend, and the backtest would pay out dividends that never were.

**3. Stored returns do not change.** A ticker's bars to 10 November are scaled by one constant,
the product of its events since. Every return to 10 November stays EODHD's.

**4. On AWS** (applied by hand, with Efe's approval):
- the nightly task reads Alpaca's two keys instead of EODHD's;
- its role may read the frozen file;
- the command line does not change.

## Evidence

`research/phase3/prices_from_alpaca.py`, results in `results/05_prices_from_alpaca.txt`:

**1. The past, rebuilt.** EODHD's history was cut at the end of 2024, as it stood then, and
continued by Alpaca to 10 November 2026, as the nightly job will continue it. Set beside
EODHD's own bars, on 233,297 member-days:
- the traded close agrees within 1 bp on 99.94% of them;
- the daily return agrees within 1 bp on 99.84%, and within 10 bps on 99.98%; the mean
  absolute gap is 0.034 bps;
- membership agrees on every member-day.

The largest gaps:
- **Spin-offs that EODHD left unadjusted, or adjusted as a small dividend.** Aptiv on 1 April
  2026 shows −10.58% in EODHD and +5.56% here; Lennar, Fortive and Western Digital are
  similar. Alpaca's adjustment is closer to the value spun off.
- **TransDigm's 90 USD special dividend of 2 September 2025,** which Alpaca lists twice:
  +6.55% here against EODHD's +0.19%.

**2. The switch.**
- Every return to 10 November is unchanged; the largest change is 2.2e-16.
- Every split and dividend to that day is unchanged.
- The result passes validation, with 500 members with a bar on each new session.
- On each of the 12 days N5 and N6 have scored, both records see the same members and the
  same next-day returns as before.

## Consequences

- **The nightly job needs neither EODHD nor money.** It takes a minute instead of twenty.
- **Doubled dividends are possible.** Alpaca now and then lists a dividend twice, as with
  TransDigm. Ford's two 0.15 USD dividends of February 2025, a regular one and a special
  one, look the same and are real. Both are kept as listed; a doubled dividend shows as one
  wrong return on its ex-date.
- **Foreign companies' dividends** differ from EODHD's by a few basis points (Accenture,
  Smurfit Westrock).
- **Renamed members and reused tickers.** A member renamed after 10 November arrives as a new
  ticker with Alpaca's history. Before the rename, its corporate actions under the new
  symbol may belong to another security. A ticker that a new company reuses, with the
  earlier company's bars stored under it, would continue those bars.
- **`qfin intraday` and `qfin etf-minutes` still ask EODHD,** so they no longer work. The
  stored minutes stay. They are needed only for the nightly plan's backtest and QQQ's check
  against daily closes.
- **A failed nightly job still raises no alert.** That remains an open item.
- **Renewing EODHD would not undo this.** The frozen file would be continued the same way.
