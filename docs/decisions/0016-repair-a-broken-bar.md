# 0016: A broken bar is repaired, not a reason to stop the nightly job

- **Status:** accepted
- **Date:** 2026-10-13

## Context

On 13 October 2026 the nightly AWS job stopped. Its log ends with "validation
failed; nothing written". One of 1.9 million daily bars broke the rule
`low <= open, close <= high`: EODHD's bar of LUMN for 12 October. A local
`qfin ingest` stopped on the same bar. By evening EODHD had not corrected it,
and its LUMN bar for 13 October was broken the same way. Left alone, every
nightly run would stop until EODHD fixed both bars.

Alpaca's SIP minutes show which fields are wrong:

| LUMN, 12 October 2026 | Open | High | Low | Close |
| --- | --- | --- | --- | --- |
| EODHD | 5.25 | 5.05 | 5.05 | 5.05 |
| SIP minutes (Alpaca) | 5.25 | 5.28 | 4.99 | 5.055 |

The open and close are right. The high and low are wrong: they are the same
number, as they are again on 13 October (4.985).

Three ways to handle such a bar:

- **Stop, as before.** One bar of one stock holds up every stock's data,
  night after night.
- **Drop the bar.** The stock is then a day short, and the plan needs 252
  full days of returns: a stock of the trading universe would leave the
  plans for a year.
- **Repair it.** Widen the high and low to cover the open and close. The
  returns, which come from the closes, stay right. Of the daily bars, only the
  high and low change, and nothing in the system uses them for a decision:
  fills come from the 1-minute bars.

## Decision

- **`qfin ingest` repairs before it validates.** A bar whose open or close
  lies outside its low and high gets them widened to cover the open and
  close (`repair_broken_bars`). The job prints a warning that names each
  repaired bar, and the log keeps it.
- **Validation still treats such a bar as an error,** so a broken bar that
  reaches it by any other way still stops the write.
- **The full history is downloaded every night,** so a bar EODHD corrects
  later replaces the repaired one by itself.

## Consequences

- The nightly job runs again from the next image pushed to ECR.
- The repaired high and low lie inside the true range (5.25 and 5.05 against
  5.28 and 4.99 for LUMN). Daily highs and lows are not used for decisions,
  so this changes no plan, trade or result.
- A wrong open or close cannot be told from the bar alone. The `large_moves`
  check warns when one makes a day's move implausible, as before.
