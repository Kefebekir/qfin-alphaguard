# Ledger: the GPU gym's networks

Every pre-registered test of a network trained in the GPU gym, and every setting tried on its
development years. A pre-registered test was written down and approved by Efe before any of
its results on real data was computed; its file is unchanged since (SHA-256 below).

## Pre-registered tests

| # | Written (UTC) | File | SHA-256 | Hypothesis | Pass if | Result |
|---|---|---|---|---|---|---|
| G1 | 2026-10-17 15:12:02 | `G1_es_network.txt` | `f079a348eb8451206f337120a8b2677bb5d309bdc682745a55abd8d9cf0c8227` | A small network trained by evolution strategies on 16 ETFs' earlier years beats the QQQ rule at equal risk on QQQ | one-sided t ≥ 2.0 on 2022 – 9 October 2026 | **Failed.** −7.88 bps a day against the rule, t −2.92: worse in all five years. The network −0.9% a year (Sharpe −0.08), the rule +19.0% (1.56) |

## Settings tried on the development years (2018-2021)

At most 10. Each row is written before the next setting is run. The one with the highest
development mean a day goes to the holdout (fixed in the pre-registration).

| # | Run (UTC) | Network | Evolution | Network less rule, a day | t | 2018 / 2019 / 2020 / 2021, bps a day | Training Sharpe at the end | Note |
|---|---|---|---|---|---|---|---|---|
| 1 | 2026-10-17 | hidden 16, sees the rule | 300 generations, population 256, sigma 0.05, lr 0.02, seed 0 | −2.51 bps | −0.52 | −21.79 / −3.87 / +11.84 / +3.66 | 2.08 / 1.78 / 1.44 / 2.06 | Worse than the rule with two years to learn from (2018), better with four and five (2020, 2021). A second run gave the same numbers to six decimals |
| 2 | 2026-10-17 | hidden 16, blind to the rule | 300 generations, population 256, sigma 0.05, lr 0.02, seed 0 | −0.59 bps | −0.12 | −24.55 / +0.37 / +16.88 / +4.78 | 2.04 / 1.66 / 1.49 / 1.97 | 2018's loss is QQQ's own fall: trained on 2016-2017 only, it is long through the day (beta to QQQ's open-to-close move +1.19, correlation 0.83). 2020 and 2021 are not: beta +0.02 and +0.04, and +21.9 and +9.0 bps a day beyond it |
| 3 | 2026-10-17 | hidden 16, blind to the rule | 300 generations, population 256, sigma 0.05, lr 0.02, seed 1 | −1.83 bps | −0.38 | −23.28 / −5.95 / +15.42 / +6.32 | 2.09 / 1.77 / 1.49 / 1.97 | Setting 2 with another seed: 2018, 2020 and 2021 repeat; 2019 moves from +0.37 to −5.95. The gap between settings 2 and 3 is the seed's alone |
| 4 | 2026-10-17 | hidden 16, blind to the rule | 600 generations, population 256, sigma 0.05, lr 0.02, seed 0 | +1.85 bps | +0.37 | −14.03 / −1.42 / +19.48 / +3.23 | 2.40 / 2.20 / 1.96 / 2.34 | Setting 2 trained twice as long: the best so far. Beyond its beta to QQQ's day: +8.6 / −0.6 / +22.1 / +7.4 bps a day |
| 5 | 2026-10-17 | hidden 32, blind to the rule | 300 generations, population 256, sigma 0.05, lr 0.02, seed 0 | −3.17 bps | −0.75 | −27.21 / −1.97 / +11.41 / +4.92 | 1.83 / 1.80 / 1.51 / 2.06 | Setting 2 twice as wide: worse. Run before setting 4's row was written (both were run back to back); recorded here in their order |

**Chosen for the holdout by the fixed rule (the highest development mean): setting 4,** a
network of 16 hidden units blind to the rule, trained for 600 generations (+1.85 bps a day
against the rule on 2018-2021). Settings 4 and 5 were both run before either was recorded;
the rule that picks the holdout's setting was fixed before any of them ran.

## G1's holdout (run once, 17 October 2026, commit a8ef197)

`results/02_G1_holdout.json` (the lock is `data/gpu_sim/holdout_G1.json`; a second run is
refused). Setting 4, retrained from scratch for each year on all 16 ETFs' years before it:

| | 2022 | 2023 | 2024 | 2025 | 2026 | All |
|---|---|---|---|---|---|---|
| Network less rule, bps a day | −8.48 | −11.62 | −9.32 | −1.23 | −8.97 | −7.88 (t −2.92) |
| Network's Sharpe ratio | −0.14 | 0.04 | 0.34 | 0.22 | −1.00 | −0.08 |
| Rule's Sharpe ratio | 1.52 | 2.63 | 2.23 | 0.39 | 1.18 | 1.56 |

**G1 failed, clearly: the network is worse than the rule in every holdout year.** It does
not lean on QQQ's own move (its beta to QQQ's day is −0.04 to +0.23), so this is not a bad
market for a long-biased model: it simply has no edge. What it found in 2020 and 2021 on
development, with two seeds alike, was particular to those years. Its training Sharpe
stayed high (1.5-2.2 at the end of each year's training), so the gap between what it
learned on 16 ETFs' past and what holds on QQQ's future is the overfitting the research
warned of, now at GPU scale.
