# 0004: Trade a small universe of liquid US single stocks

- **Status:** accepted
- **Date:** 2026-10-07

## Context

The first plan used US ETFs such as SPY and QQQ. A UK retail investor cannot
buy US-domiciled ETFs: an overseas fund can only be marketed to UK retail
investors as a recognised scheme, and no US funds are recognised in the UK
(FCA PS25/20). London-listed UCITS ETFs trade 08:00–16:30 UK time, during
lectures.

## Decision

Trade 8–10 liquid US single stocks during the US session (14:30–21:00 UK
time for most of the year). Long only, no leverage.

## Consequences

- Single stocks carry more idiosyncratic risk than ETFs, so the optimiser's
  weight cap and Guard's `max_weight` matter more.
- Picking today's large companies creates survivorship bias in backtests. The
  universe rule is written down and the bias is stated in every report.
- An S&P 500 UCITS ETF remains a benchmark, because it is what a UK investor
  could hold instead.
