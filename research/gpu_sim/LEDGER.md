# Ledger: the GPU gym's networks

Every pre-registered test of a network trained in the GPU gym, and every setting tried on its
development years. A pre-registered test was written down and approved by Efe before any of
its results on real data was computed; its file is unchanged since (SHA-256 below).

## Pre-registered tests

| # | Written (UTC) | File | SHA-256 | Hypothesis | Pass if | Result |
|---|---|---|---|---|---|---|
| G1 | 2026-10-17 15:12:02 | `G1_es_network.txt` | `f079a348eb8451206f337120a8b2677bb5d309bdc682745a55abd8d9cf0c8227` | A small network trained by evolution strategies on 16 ETFs' earlier years beats the QQQ rule at equal risk on QQQ | one-sided t ≥ 2.0 on 2022 – 9 October 2026 | pending |

## Settings tried on the development years (2018-2021)

At most 10. Each row is written before the next setting is run. The one with the highest
development mean a day goes to the holdout (fixed in the pre-registration). None yet.
