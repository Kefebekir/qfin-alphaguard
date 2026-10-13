# Roadmap

Every phase ends with something that runs and an up-to-date README. A phase
starts only after the previous gate is passed. Dates assume about ten hours a
week alongside university.

## Where we are

**Phase 1, step 10 is next:** `qfin backtest`, the report against
buy-and-hold and equal weight. Step 9 is done: Guard checks every order
against `guard.yaml`, and the strategy keeps within its limits (decision
0014, #28). The band's trades go out in equal-spaced child orders, three from
10:30 to 15:30, once a day's trades are worth 20 USD together (decision 0013,
#26 and #27). The backtest trades fractional shares at Alpaca's costs
(decisions 0008 and 0012). Efe chose Alpaca to start real money with about
500 USD, and no leverage for the nightly plan for now (decision 0009). A
second strategy, intraday momentum on the Nasdaq-100 (decision 0011), has its
own steps in Phase 1b; the first is done: QQQ's 1-minute bars are in the
store (#22).

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
| 6. Simulated broker and cost model: next-bar fills, commission, FX, spread, slippage | Efe | done, #16 |
| 7a. Daily plan: `qfin plan` with Ledoit-Wolf and CVXPY, as many stocks as the capital can hold | Claude | done, #17 |
| 7b. The band: which gaps to trade at the open, in whole shares (decision 0006) | Efe | done, #18 |
| 7c. 1-minute bars for stocks that leave the universe, so they can be sold | Claude | done, #19 |
| 7d. The band trades a gap only as far as its edge (decision 0007) | Efe | done, #20 |
| 7e. Fractional shares and the broker's costs: event types, simulated broker, band (decisions 0008 and 0012) | Claude designs; Efe's rules | done, #24 and #25 |
| 7f. Fill each source's minute holes from the other: the stocks' EODHD minutes from Alpaca's SIP history (decision 0008), QQQ's Alpaca minutes from EODHD's | Claude | done, #23 |
| 8. Timing without ML: equal-spaced child orders (decision 0013) | Efe | done, #27 |
| 9. Guard rules in the backtest (decision 0014) | Claude; Efe's limits in `guard.yaml` | done, #28 |
| 10. Report against buy-and-hold and equal weight: `qfin backtest` | Claude | to do |

Phase 0 is done and Gate M0 is met (#1–#6): cleanup, README, event types,
`CLAUDE.md`, `guard.yaml` and its loader, docs, CLI subcommands.

Outside the code: the Alpaca paper account is open (9 October 2026); its keys
are in the git-ignored `.env`.

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
- [x] Selling a stock that leaves the trading universe at the new year: its
      January minutes are downloaded too, 6–10 stocks a year (step 7c).
- [ ] Stock-days without minutes. Step 7f filled EODHD's holes from Alpaca's
      SIP history (#23); neither provider has VIAC's minutes after 16
      February 2022, when it became PARA, nor DWDP's after 31 May 2019, when
      it became DD (next item). The backtest must not trade a stock on a day
      without minutes, and the report says how often the plan wanted to (step
      10). Step 8's timing waits for the price of every stock in the plan or
      held: one never priced holds up the day's trades to the deadline, and
      one priced before is ordered at a stale price and never fills.
- [ ] A held stock whose bars stop during its year: renamed (DWDP → DD in
      2019, FB → META and VIAC → PARA in 2022) or bought for cash (TWTR, 54.20
      USD on 27 October 2022). The backtest must carry the shares over to the
      new code or pay the cash out, or it keeps a position it can neither sell
      nor value.
- [x] Daily plan from the CVXPY optimiser with Ledoit-Wolf covariance, holding
      as many stocks as the capital can hold.
- [x] The band: trade a stock only when its gap is larger than its own
      volatility allows (decision 0006, step 7b), and then only as far as the
      band's edge (decision 0007, step 7d).
- [ ] Fewer swaps at small capital: at 30,000 USD the plan holds its top 10,
      and stocks entering and leaving it, 42 times in 2025, cost 16 of the 19
      bps a year whatever the band (decision 0006). A buffer (keep a held
      stock while it ranks within the top 12) halved the trades over
      2017–2026 without raising the Sharpe ratio (decision 0007); measure it
      again in the step 10 backtest.
- [x] Timing without ML: equal-spaced child orders, three from 10:30 to
      15:30; a day's trades wait until they are worth 20 USD together
      (decision 0013, step 8).
- [ ] Confirm decision 0013's schedule in the step 10 backtest, net of
      costs, with walk-forward tests. Its measurements: the number of
      orders hardly matters, the half spread at 09:35 is twice that of
      10:30, and at 500 USD days of at least 20 USD cut Alpaca's fees from
      0.80% to 0.29% a year. Guard's limits stay hand-set safety limits,
      not tuned for profit.
- [x] Cost model: the broker's commission and fees, plus spread and slippage;
      Alpaca's fees since step 7e (decision 0012). On 2025's real minutes a
      marketable ~2,000 USD order costs about 1.3 bps against the decision
      price; IBKR would have added 1.9 bps of commission.
- [x] Guard rules applied in the backtest too, `max_gross_exposure` included
      when each order is sent (decision 0014, step 9). Buys count at their
      limits and an open sale frees no cash, so the account never borrows;
      without Guard it closed a day of 2025 with −1.15 USD of cash at 30,000
      USD.
- [ ] The simulated broker fills a fractional buy under 1 USD, which Alpaca
      refuses. Only a buy Guard cuts can be that small: 4 in 2025, at 30,000
      and 5,000 USD (decision 0014).
- [ ] A held stock whose prices end, as TWTR's did when it was taken private
      for cash and VIAC's after it became PARA, is turned into cash at its
      last price; otherwise it stays in the books for ever (decision 0013).
- [ ] Report: return, Sharpe, maximum drawdown, turnover, cost share, against
      buy-and-hold and equal-weight baselines, and the days: daily profit and
      loss, the share of winning days, the worst day. Capital levels: 500,
      5,000 and 30,000 USD. Variants: volatility targeting and leverage, net of
      borrowing costs (decision 0009).

**Gate M1:** one command produces the same report from the same inputs every
time; results after costs compared with the baselines.

### Phase 1b: Intraday momentum on the Nasdaq-100 (decision 0011)

A second strategy beside the nightly plan: it follows the day's trend on the
Nasdaq-100 and is flat at every close.

- [x] QQQ's 1-minute bars from Alpaca's SIP history into the store, checked
      against EODHD's: `qfin etf-minutes QQQ` (#22).
- [ ] The strategy as its own `Strategy` in our engine: the Noise Area, VWAP,
      decisions at HH:00 and HH:30, a size of min(cap, 2% / 14-day
      volatility), flat at the close. Efe writes the trading rules; Claude
      prepares the skeleton and the spec.
- [ ] Backtest through our engine and simulated broker. It must reproduce
      the research: 2016–2026, cap 2×, 18.3% a year, Sharpe ratio 1.47,
      with the minute holes filled (decision 0011's update).
- [ ] Guard limits for an intraday strategy, set by hand: round trips and
      reversals allowed for it, a leverage cap of 2×, the daily loss limit.
- [ ] Paper trading of QQQ on Alpaca's paper account for two to three
      months, compared with the backtest every day.
- [ ] A broker and instrument for real money from the UK: QQQ at Alpaca if
      its UK accounts allow it, otherwise a Nasdaq-100 CFD or spread bet with
      an API and small sizes.

**Gate M1b:** the engine's backtest matches the research, and the paper
results stay inside the backtest's range.

### Phase 2: Alpha AI v1 (7 December 2026 – 31 January 2027)

Prove whether ML adds anything after costs. Slower during January exams.

- [ ] Two targets: next-day return and volatility (for the optimiser); the
      direction of the next 30 minutes, only to time the plan's trades: it
      never trades its own signal (decision 0010).
- [ ] Purged walk-forward: drop training samples whose target horizon overlaps
      the test period, with an embargo before it.
- [ ] Ridge and logistic regression first, then LightGBM; deep models only if
      they beat both.
- [ ] Feature parity test: batch and incremental calculations agree.
- [ ] Model registry (version, training date, data range) and a
      champion/challenger rule.
- [ ] Prequential evaluation: every model predicts first and learns after, on
      all stocks together; retrained weekly, later daily, and promoted only by
      the champion/challenger rule. No model updates itself after each signal
      or for each stock (decision 0010).
- [ ] Health check: if the score distribution drifts, fall back to timing without ML.
- [ ] Revisit Guard's limit of 3 orders a stock a day. Efe expects to raise
      it once a timing model shows a measured gain from more orders; with
      decision 0010's signal, 31 orders beat 3 by 1.26 bps a trade
      (decision 0013). The change is Efe's, by hand in `guard.yaml`, with
      a decision record.

**Gate M2:** two reports. Nightly model: Sharpe after costs against the plan
without ML. Intraday model: execution cost against equal-spaced timing. "ML
adds nothing" is a valid result; that part then runs without ML.

### Phase 3: Paper trading (February – May 2027)

Prove the live system behaves like the backtest. Eight weeks cannot prove
profit; this phase asks whether the system works correctly.

- [ ] Always-on server for the engine, alerts, and the `live_trading` lock.
- [ ] Alpaca paper account and its API: orders by REST, fills by WebSocket.
- [ ] Live data: trades from a free single-exchange feed (EODHD's EDGX
      WebSocket or Alpaca's IEX) built into 1-minute bars, and compared with
      the consolidated tape every day (decision 0008).
- [ ] Order state machine: new → sent → acknowledged → partial → filled /
      cancelled / rejected; a unique client order id per order.
- [ ] Guard live: every rule, halt and kill switch. The View's price times
      come from the feed and its last close value from the broker's equity;
      Efe decides whether the daily loss limit halts the engine until he
      restarts it (decision 0014).
- [ ] Reconciliation at open and close: broker positions equal engine positions.
- [ ] Chaos tests: lost connection, stale data, rejected orders; the system
      stops safely.
- [ ] Daily report: PnL, slippage, deviation from backtest, share of winning days.
- [ ] At least 8 weeks of uninterrupted running (April–May).

**Gate M3:** replaying the same days reproduces at least 95% of paper
decisions; measured slippage stays inside the cost model; no manual fix needed
for 8 weeks.

### Phase 3b: Small real money and scaling (June 2027 onwards)

- [ ] Account type at Alpaca: cash or margin; leverage only through the gate
      of decision 0009.
- [ ] Starting capital: about 500 USD, an amount Efe can afford to lose.
- [ ] `live_trading: true` only in this phase, set by hand.
- [ ] Monthly report: net return, difference from benchmarks, expected range,
      slippage, Guard events.
- [ ] Scaling and stop rules (above).

**Gate M3b (end of November 2027):** six months of live results inside the
expected range, no stop rule triggered; capital steps up one level.

### Phase 4: C++ engine (June – September 2027)

Move the live path to C++ and measure it. Moves to September–December if a
summer internship takes priority.

- [ ] C++20, CMake, GoogleTest; a C++ client for the broker's REST and WebSocket API.
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
| Underestimated costs | gross profit, net loss | cost model from Phase 1; fills measured in paper trading; monthly fees counted against small capital |
| Account and regulatory rules | intraday and settlement limits | US stocks; no same-day reversals; check Alpaca's rules before Phase 3b |
| Broker API limits | rate limits; fractional orders are day orders | everything on Alpaca's paper account first |
| Data quality | missing bars, split errors | validation layer, stale-data rule |
| Optimistic paper fills | good on paper, worse live | conservative cost model; re-measure slippage in the first live month |
| Broker connection | API or feed outages | halt on stale data, reconciliation at open and close |
| University workload | pauses in January and May–June | passive work then: monitoring, reading, FPGA basics |
| FPGA learning curve | weeks stuck in simulation | small modules, testbench first, start 5a early |

## Out of scope

- High-frequency trading, co-location and direct market access.
- Leverage, short selling and options.
- Same-day opposite-direction trades (intraday tilt); possible as a Phase 6 experiment.
- An LLM layer.
- Quantum optimisation in the live path; only offline research.
- A custom PCB before Phase 6.
