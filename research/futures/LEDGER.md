# Ledger: the QQQ rule on other markets' futures

Every test of the rule beyond QQQ, in order. A pre-registered test was written down, with its
hypothesis, data, costs and threshold, and approved by Efe before any of its results was
computed; its file in `preregistrations/` is unchanged since (SHA-256 below, of the file as
committed).

## Pre-registered tests

| # | Written (UTC) | File | SHA-256 | Hypothesis | Pass if | Result |
|---|---|---|---|---|---|---|
| F1 | 2026-10-17 13:18:46 | `F1_nikkei_rule.txt` | `bf23a37ef7ebd0805b655c8de0f0123d8c225c4b6bdbdbb2d2a00bb2aa6f6fb5` | The QQQ rule, unchanged, earns on CME Nikkei 225 futures (NIY) over the Tokyo session after one tick a fill | one-sided t ≥ 2.0 on 3 January 2017 – 9 October 2026 | **Failed.** −5.4% a year after one tick a fill (Sharpe −0.59, t −1.82), down in 9 of 10 years; at half a tick, 0.0% (Sharpe 0.01) |

## Notes

| # | Written (UTC) | File | SHA-256 | Note |
|---|---|---|---|---|
| 1 | 2026-10-17 13:19:23 | `F1_note_1.txt` | `a9f8aaddf7d412185e118f919892e287e7826059e6e54cb3fb8e351091da57c3` | The rule is momentum.py's next_position: VWAP is its stop, not part of its entry; F1's summary line simplified it |

## Data bought

| Date | Data | Cost |
|---|---|---|
| 2026-10-17 | CME NIY 1-minute bars, NIY.v.0, 2016-01-04 to 2026-10-09 (`download_niy.py`) | 12.09 USD of Databento credit |

## F1's test (run once, 17 October 2026, commit a8fe31d)

`results/02_F1_test.json` (the lock is `data/futures/F1_result.json`; a second run is
refused). The implementation check first reproduced qqq_momentum.py on QQQ exactly
(`results/00_check_qqq.txt`); 2016, the implementation year, is in
`results/01_implementation_2016.txt`.

| 2017 – 9 October 2026 | A year | Sharpe | t |
|---|---|---|---|
| One tick a fill (the test) | −5.4% | −0.59 | −1.82 |
| Half a tick | +0.1% | +0.01 | +0.02 |

- **The rule finds almost nothing on Nikkei.** Before costs it earns about 2.2 bps a day,
  some 5.5% a year, and one tick a fill costs about the same. With either cost it is no
  better than zero, and with one tick it is down in nine of the ten years.
- **The breadth was real; the edge was not.** Its daily returns are uncorrelated with the
  QQQ rule's (−0.02), but together at equal risk the two have a Sharpe ratio of 0.62,
  against 1.44 for the QQQ rule alone. Adding a market without an edge dilutes one with it.
- **What it says.** As in step 5's breadth research (no edge outside QQQ, SPY and XLK), the
  rule's edge looks particular to the US index, where the hedging of index options (0DTE
  above all) and leveraged ETFs is largest. CME's Nikkei contract, thin in Tokyo hours, may
  also not be where Japan's own flows show: Osaka's is the main market. The FTSE 100
  (55.82 USD) is not bought: F1 made it worth buying only on a pass.
