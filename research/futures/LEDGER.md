# Ledger: the QQQ rule on other markets' futures

Every test of the rule beyond QQQ, in order. A pre-registered test was written down, with its
hypothesis, data, costs and threshold, and approved by Efe before any of its results was
computed; its file in `preregistrations/` is unchanged since (SHA-256 below, of the file as
committed).

## Pre-registered tests

| # | Written (UTC) | File | SHA-256 | Hypothesis | Pass if | Result |
|---|---|---|---|---|---|---|
| F1 | 2026-10-17 13:18:46 | `F1_nikkei_rule.txt` | `bf23a37ef7ebd0805b655c8de0f0123d8c225c4b6bdbdbb2d2a00bb2aa6f6fb5` | The QQQ rule, unchanged, earns on CME Nikkei 225 futures (NIY) over the Tokyo session after one tick a fill | one-sided t ≥ 2.0 on 3 January 2017 – 9 October 2026 | pending |

## Notes

| # | Written (UTC) | File | SHA-256 | Note |
|---|---|---|---|---|
| 1 | 2026-10-17 13:19:23 | `F1_note_1.txt` | `a9f8aaddf7d412185e118f919892e287e7826059e6e54cb3fb8e351091da57c3` | The rule is momentum.py's next_position: VWAP is its stop, not part of its entry; F1's summary line simplified it |

## Data bought

| Date | Data | Cost |
|---|---|---|
| 2026-10-17 | CME NIY 1-minute bars, NIY.v.0, 2016-01-04 to 2026-10-09 (`download_niy.py`) | 12.09 USD of Databento credit |
