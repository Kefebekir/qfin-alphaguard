# Experiment ledger: Phase 2, step 5 research (15 October 2026)

Every idea tested on our data, in the order it was tested. A **pre-registered** test was written
down, with its direction, data and threshold, before its result was computed; its text is in
`preregistrations/`, unchanged since (the SHA-256 below is of the file as committed), and its
output is in `results/`. An **exploration** was looked at without such a note: it can suggest a
hypothesis, never confirm one. Times are UTC.

Ideas tried count against every later claim: with this many tests, a single t of 2 means
little. The shadow record (step 5's platform) counts its challengers on top of these.

## Pre-registered tests

| # | Written | File (SHA-256, first 16) | Hypothesis | Data | Pass if | Result | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | 11:19:06 (correction 11:19:24) | `01_policy_networks.txt` (`40f41840f020e15e`) | The gate network fitted on QQQ, SPY, IWM and DIA, the only one of 26 learned policies to clear the selection test on 2019-2022 (t 3.35), beats the rule on 2023-2026 | QQQ 2023-2026, walk-forward | t >= 2.0 | -2.81% a year against the rule, t -1.14; no learned variant beat the rule | failed |
| 2 | 16:01:11 | `02_gamma_proxies.txt` (`3d7ea6099e2ef85d`) | Sizing the rule up when yesterday's variance ratio and realised/implied volatility say dealers are short gamma (H1), and flat on FOMC and monthly expiry days (H2), beats it at equal risk | QQQ 2023-2026 | t >= 2.24 | H1 t -2.51, H2 t -1.97: the relation of 2016-2022 reversed | failed |
| 3 | 16:42:42 | `03_overnight.txt` (`ccf02715125632d3`) | O1: the overnight return falls with the session's open-to-close return; O2: higher overnight into CPI and payrolls mornings | QQQ, SPY 2016-2026 | O1 t <= -2.24, O2 t >= 2.24 | O1 QQQ t -2.19 (SPY -3.58); O2 t 0.36 / 0.61 | failed (QQQ) |
| 4 | 17:04:03 | `04_overnight_validation.txt` (`2820f8f80c5d582b`) | The last half hour reverses overnight (t -9 on 2016-2026), on years before 2016 | SPY, QQQ minutes 2014-2015; daily 1993-2015 | t <= -2.64 | V1/V2 t about 0; V3 positive (QQQ +3.41). The 2016-2026 effect is almost all 2020 | failed |
| 5 | 16:14:02 (amended 16:47:25, 17:07:05) | `05_real_gamma.txt` (`bc58221b1b865e49`) | The rule earns less as real dealer gamma (GEX from options open interest) rises | QQQ rule, QQQ and SPY options 2017-2022 | t <= -2.24 | G1 t -1.58, G2 t -1.85 | failed |
| 6 | 18:06:07 (note 18:12:11) | `06_real_gamma_validation.txt` (`110cc9ccb9c115c7`) | The rule earns less on days whose QQQ GEX rank is in its top third (seen in exploration on 2017-2022, as Baltussen et al. 2021 find) | QQQ 2023-2026 | t <= -2.0 | +4.24 against +9.37 bps a day, t -1.07; lower in each of the 4 years; flat on those days loses (Sharpe 1.55 -> 1.43) | failed |
| 7 | 18:17:06 | `07_batch.txt` (`6d43f7901db71a9d`) | Six ideas from the literature: same-half-hour continuation (B1), last 15 minutes (B2), Treasury auction cycle on TLT (B3), New York morning sunshine (B4), HMM regimes (B5), Bitcoin weekly momentum (B6) | as stated in the file | t beyond 2.64 | B1 t -4.10 (opposite sign); B2 -0.34; B3 0.63; B4 0.58; B5 -0.08; B6 1.92 | all failed |
| 8 | 18:21:58 | `08_rsi.txt` (`8f98411d9aa1be05`) | Replication of the rule's source paper (Zarattini, Aziz, Barbon 2024): the strategy earns less as SPY's 5-day RSI rises | QQQ rule 2016-2026 | t <= -2.0 | t -1.88; after the paper (2024-06 on) t -0.69 | failed |

Not in a separate file, but fixed in the script's docstring before it ran: the online expert
allocation (`hedge_test.py`, Hedge and Fixed-Share over 33 experts, 2019-2026): -4.70% a year
against the rule, t -3.67. Failed.

## Explorations (not pre-registered)

| Script | Question | What it showed |
|---|---|---|
| `policy_nets.py`, `policy_gate_check.py` (selection run) | Can learned policies beat the rule on 2019-2022? | One of 26 (gate, index ETFs) cleared t 3.10; its gain was almost all 2022 |
| `flow_explore.py` | Where does the rule earn: time of day, event days, volatility regime, leveraged-ETF demand? | Consistent with dealer hedging and news (FOMC days worse, payrolls days better, stress better), each weak |
| `breadth_open.py` | Does breadth at 09:45 mark trend days? | Not monotone: no |
| `timing_luck.py` | A grid shifted by 15 minutes? | Worse (Sharpe 1.18 -> 0.93); the :00/:30 grid matters |
| `gamma_proxies.py` | Price-based gamma proxies on 2016-2022 | Strong and monotone in 2016-2022; reversed in 2023-2026 (test 2) |
| `calendar_effects.py` | Overnight vs intraday, turn of the month, pre-FOMC drift | Overnight ~60% of QQQ's return, uncorrelated with the rule; pre-FOMC drift gone 2016-2019, back 2020-2026 |
| `overnight_review.py` | The rule plus an overnight QQQ position, on independent (EODHD) data | Sharpe up in 2017-2019 and 2023-2026, down in 2020-2022: it adds market risk, not an edge |
| `tsmom.py` | Time-series momentum on the 16 ETFs | Sharpe 0.33-0.55 (long/flat 0.74-0.83); a small diversifier |
| `breadth_stocks_etfs.py` | The rule on 97 stocks and 15 ETFs | No edge before costs outside QQQ, SPY and XLK |
| `gamma_top_third.py` | After test 5 failed: the top third of GEX | Rule earns ~0 there (t -2.07 / -2.97), which led to test 6 |
| `overnight_by_year.py` | The last half hour's reversal, year by year | Driven by 2020 |
