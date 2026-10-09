# Roadmap

Every phase ends with something that runs and an up-to-date README. A phase
starts only after the previous gate is passed. Dates assume about ten hours a
week alongside university.

## Where we are

**Phase 1, step 6 is next:** the simulated broker and its cost model, written by
Efe; Claude prepares the skeleton, the tests and the explanations.

| Phase 1 step | Core logic by | State |
| --- | --- | --- |
| 1. Daily OHLCV bars, split- and dividend-adjusted, with validation | Claude | done, #7 |
| 2a. Universe rule (decision 0005) and S&P 500 membership by date | Efe decided, Claude wrote | done, #8 |
| 2b. Daily bars from EODHD for every member since 2015, stocks that left included | Claude | done, #9 |
| 2c. Trading universe: the 50 most liquid members, chosen each 1 January | Claude; Efe set N = 50 | done, #10 |
| 3. Exchange calendar: sessions, holidays and half days, all in UTC | Claude | done, #11 |
| 4a. 1-minute bars since 2016 for each year's trading universe, cut to the regular session | Claude | done, #12 |
| 4b. The stored minutes as `Bar` events, in time order | Claude | done, #13 |
| 5. Event loop: time-ordered feed and `Strategy` interface; decide after bar t, fill no earlier than t+1 | Claude, reviewed by Efe | done, #14 |
| 5b. Splits and dividends: share counts follow splits, dividends paid in cash | Claude | done, #15 |
| 6. Simulated broker and cost model: next-bar fills, commission, FX, spread, slippage | Efe | next |
| 7. Daily plan: `qfin plan` with Ledoit-Wolf and CVXPY; no trade below `band_pct` | Claude; Efe writes the band rule | to do |
| 8. Timing without ML: equal-spaced child orders | Efe | to do |
| 9. Guard rules in the backtest | Efe | to do |
| 10. Report against buy-and-hold and equal weight: `qfin backtest` | Claude | to do |

Phase 0 is done and Gate M0 is met (#1–#6): cleanup, README, event types,
`CLAUDE.md`, `guard.yaml` and its loader, docs, CLI subcommands.

Outside the code: open the IBKR live account early. The paper account is tied
to it and approval can take time.

## Goal and evidence

The goal is net profit after costs and tax, beating an equal-weight
buy-and-hold of the same stocks and an S&P 500 UCITS ETF at similar risk.

Showing that a result is not luck needs a t-statistic of about 2. With an
annual Sharpe ratio SR over T years, t ≈ SR·√T, so T ≈ (2 / SR)²: about four
years of live trading for a Sharpe of 1. Evidence therefore comes in layers:

1. **Long out-of-sample walk-forward backtests**, costs included. This is the
   main evidence.
2. **Paper trading** shows the live system can earn what the backtest earns;
   wrong fills, delays and data errors quietly destroy backtest profits.
3. **Small real money** confirms live results stay inside the backtest's
   expected range (the 5–95% band of same-length backtest periods), and
   capital grows by rule:
   - start with a small part of the planned capital;
   - every three months, step up if the net result is inside the expected
     range and Guard reported no incidents;
   - stop and investigate if the drawdown exceeds 1.5 times the largest
     backtest drawdown, or results fall below the range.

The answer may be "no edge". The stop rules then end trading without a large
loss, and the project keeps its engineering value.

## Phases

### Phase 0: Scope and skeleton (8–18 October 2026)

Fix the decisions and build the skeleton (steps above).

- Time scale: a nightly plan plus intraday timing on 1-minute bars.
- Universe: 8–10 liquid US stocks (see `decisions/0004`).

**Gate M0:** CI green; README describes the new design; `events.py` and
`guard.yaml` tested.

### Phase 1: Event-driven backtester (19 October – 6 December 2026)

Run the nightly plan and intraday timing end to end on history, without ML.

- [x] Data: EODHD daily bars for every S&P 500 member since 2015, and 1-minute
      bars since 2016 for each year's trading universe (decision 0005).
- [x] Store OHLCV, not only closes: fills at the next bar's open need it.
- [ ] Universe rule written down, and its survivorship bias stated in the report.
- [x] Exchange calendar with holidays and half days; all timestamps UTC.
- [ ] Look-ahead rule: decide after bar t closes, fill no earlier than bar t+1.
- [x] Split- and dividend-adjusted prices.
- [ ] Map renamed tickers (RE → EG, PEAK → DOC, IR → TT, BHGE → BKR,
      WYND → TNL, CDAY → DAY, ARNC → HWM) so their earlier index years have
      prices; with a few acquired companies EODHD lacks, about 0.67% of
      member-days have no prices today. The same map finds 1-minute bars for
      stocks whose earlier ticker the membership file does not show (BKNG was
      PCLN until 2018, AABA was YHOO until 2017).
- [ ] Speed up the nightly download, 17 minutes for 767 codes: parallel
      requests or EODHD's bulk end-of-day endpoint.
- [ ] Add each day's 1-minute bars in the nightly job once a backtest runs in
      the cloud. Until then `qfin intraday` refreshes them locally in about 3
      minutes; on S3 they should be one file per day, because versioning would
      otherwise keep a copy of every stock's whole-year file each night.
- [ ] Backtests read the stored Parquet file, never a fresh download: two
      yfinance downloads of the same history can differ in the fifth decimal.
- [x] Event loop: the same `Strategy` class runs on history and, in Phase 3, live.
- [ ] Selling a stock that leaves the trading universe in January: its 1-minute
      bars for the new year are not downloaded yet. Download January for the
      previous year's universe too, as part of the daily plan (step 7).
- [ ] Daily plan from the CVXPY optimiser with Ledoit-Wolf covariance; no
      trade below `band_pct`.
- [ ] Timing without ML: equal-spaced child orders.
- [ ] Choose `max_children` and `min_gap_min` in the backtest, net of
      commissions (for example 1, 2, 3, 5 or 10 child orders; 0 to 60 minutes
      apart), with walk-forward tests, instead of keeping the Phase 0 guesses.
      Guard's limits stay hand-set safety limits, not tuned for profit.
- [ ] Cost model: commission and FX for the chosen broker, plus spread and slippage.
- [ ] Guard rules applied in the backtest too.
- [ ] Report: return, Sharpe, maximum drawdown, turnover, cost share, against
      buy-and-hold and equal-weight baselines.

**Gate M1:** one command produces the same report from the same inputs every
time; results after costs compared with the baselines.

### Phase 2: Alpha AI v1 (7 December 2026 – 31 January 2027)

Prove whether ML adds anything after costs. Slower during January exams.

- [ ] Two targets: next-day return and volatility (for the optimiser); the
      direction of the next 30 minutes (for timing).
- [ ] Purged walk-forward: drop training samples whose target horizon overlaps
      the test period, with an embargo before it.
- [ ] Ridge and logistic regression first, then LightGBM; deep models only if
      they beat both.
- [ ] Feature parity test: batch and incremental calculations agree.
- [ ] Model registry (version, training date, data range) and a
      champion/challenger rule.
- [ ] Health check: if the score distribution drifts, fall back to timing without ML.

**Gate M2:** two reports. Nightly model: Sharpe after costs against the plan
without ML. Intraday model: execution cost against equal-spaced timing. "ML
adds nothing" is a valid result; that part then runs without ML.

### Phase 3: Paper trading (February – May 2027)

Prove the live system behaves like the backtest. Eight weeks cannot prove
profit; this phase asks whether the system works correctly.

- [ ] Always-on server, IB Gateway with IBC for automatic login and restarts,
      alerts, and the `live_trading` lock.
- [ ] IBKR paper account and `ib_async`.
- [ ] Live data: 5-second bars into 1-minute bars (at most 60 new bar requests
      per 10 minutes).
- [ ] Order state machine: new → sent → acknowledged → partial → filled /
      cancelled / rejected; a unique client order id per order.
- [ ] Guard live: every rule, halt and kill switch.
- [ ] Reconciliation at open and close: broker positions equal engine positions.
- [ ] Chaos tests: lost connection, stale data, rejected orders; the system
      stops safely.
- [ ] Daily report: PnL, slippage, deviation from backtest.
- [ ] At least 8 weeks of uninterrupted running (April–May).

**Gate M3:** replaying the same days reproduces at least 95% of paper
decisions; measured slippage stays inside the cost model; no manual fix needed
for 8 weeks.

### Phase 3b: Small real money and scaling (June 2027 onwards)

- [ ] Broker and account type (ISA or general, cash or margin).
- [ ] Starting capital: a small part of the plan, an amount Efe can afford to lose.
- [ ] `live_trading: true` only in this phase, set by hand.
- [ ] Monthly report: net return, difference from benchmarks, expected range,
      slippage, Guard events.
- [ ] Scaling and stop rules (above).

**Gate M3b (end of November 2027):** six months of live results inside the
expected range, no stop rule triggered; capital steps up one level.

### Phase 4: C++ engine (June – September 2027)

Move the live path to C++ and measure it. Moves to September–December if a
summer internship takes priority.

- [ ] C++20, CMake, GoogleTest; IBKR TWS API C++ client.
- [ ] C++ copies of the features and Guard, reading the same `guard.yaml`.
- [ ] ML inference through ONNX Runtime.
- [ ] Golden test: a recorded trading day gives identical decisions in Python and C++.
- [ ] Benchmark bar-to-decision latency: median, p99, p99.9.

**Gate M4:** the C++ engine replaces the Python engine on the paper account,
with identical decisions and a measured latency table.

### Phase 5: FPGA (foundations November 2026 – March 2027; main work October 2027 – March 2028)

- [ ] 5a Foundations, 1–2 hours a week: counter, FSM, UART, PC ↔ FPGA data.
- [ ] 5b Risk gate: order size and position checks, approve or reject over UART.
- [ ] 5c Streaming features: returns, EMA, variance; fixed-point error analysis.
- [ ] 5d Ethernet/UDP receive, ITCH parser, top of book.
- [ ] 5e Benchmark: the same ITCH file through C++ and the FPGA; latency,
      throughput, jitter, resource use.
- [ ] A cocotb testbench for every module, bit-exact against the C++ model.

**Gate M5:** measurement table, resource report, README with the design and results.

### Phase 6: Optional (April – June 2028)

- [ ] `research/quantum`: classical solver against QAOA on the same problem and time budget.
- [ ] Intraday tilt experiment, with separate, small capital.
- [ ] Custom PCB.

## Tests that prove the system works

| Test | Proves | Phase |
| --- | --- | --- |
| Point-in-time | no feature sees the future | 1 |
| Determinism | same data and settings give the same backtest output (hash) | 1 |
| Rule violation | every Guard rule fires on a case that breaks it | 1 |
| Feature parity | batch and incremental calculations agree | 2 |
| Walk-forward | ML is measured only on unseen data | 2 |
| Chaos | lost connection, stale data, rejects and partial fills stop the system safely | 3 |
| Cost calibration | backtest slippage matches what paper trading measures | 3 |
| Replay | a recorded live day gives the same decisions again | 3 |
| Golden model (C++) | the C++ engine decides like the Python engine | 4 |
| Golden model (FPGA) | FPGA output matches C++ bit for bit | 5 |

**Benchmark rules.** Same input file and order; no timing on the warm-up run;
at least 10⁶ messages; report median, p99, p99.9 and a histogram. On the CPU,
pin the process to one core, use a monotonic clock, and record compiler
settings and hardware. On the FPGA, count cycles on the board and report the
PC ↔ board transfer time separately.

## Risks

| Risk | Sign | Mitigation |
| --- | --- | --- |
| Scope creep | new layers before a phase ends | no phase starts before its gate; see Out of scope |
| No edge | after costs, results do not beat the benchmarks | no real money; stop rules |
| Overfitting | great backtest, poor paper results | purged walk-forward, simple models, no-ML baseline |
| Underestimated costs | gross profit, net loss | cost model from Phase 1; orders of at least 1,000 USD |
| Account and regulatory rules | intraday and settlement limits | US stocks; no same-day reversals; check IBKR rules before Phase 3b |
| Broker API limits | rate limits, market orders only on some brokers | start on IBKR; test any other broker on its demo first |
| Data quality | missing bars, split errors | validation layer, stale-data rule |
| Optimistic paper fills | good on paper, worse live | conservative cost model; re-measure slippage in the first live month |
| Broker connection | Gateway drops, subscriptions to renew | IBC, halt, reconciliation at open and close |
| University workload | pauses in January and May–June | passive work then: monitoring, reading, FPGA basics |
| FPGA learning curve | weeks stuck in simulation | small modules, testbench first, start 5a early |

## Out of scope

- High-frequency trading, co-location and direct market access.
- Leverage, short selling and options.
- Same-day opposite-direction trades (intraday tilt); possible as a Phase 6 experiment.
- An LLM layer.
- Quantum optimisation in the live path; only offline research.
- A custom PCB before Phase 6.
