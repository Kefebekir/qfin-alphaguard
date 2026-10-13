# Roadmap

Every phase ends with something that runs and an up-to-date README. A phase
starts only after the previous gate is passed. Dates assume about ten hours a
week alongside university.

## Where we are

**Phase 1 is done and Gate M1 is passed; Phase 2 starts (decision 0017).**
- **In Phase 2 the ML picks the nightly plan's stocks.** It ranks every
  S&P 500 member by its expected return over the coming month, from price,
  volume and fundamental features, point in time.
- **Step 1 is next:** the data, meaning renamed tickers and SEC EDGAR
  fundamentals.
- **The pilot.** On price and volume alone, monthly gradient-boosted trees
  picking 10 of the year's 50 most liquid stocks beat holding all 50 by
  9.3% a year after costs, in 8 of 9 years (2018–2026). That is not yet
  proof: the deflated Sharpe ratio is 0.70 of the 0.95 needed
  (`docs/research/machine-learning-for-the-plan.md`).
- **Phase 1's result.** `qfin backtest` replays the nightly plan on the
  stored minutes since 2016 (decision 0015, #29). The minimum-variance plan
  loses to both baselines, at the same risk too:
  - the plan, 10.6–10.8% a year with 12.8% volatility;
  - SPY, 15.2% with 17.7%;
  - equal weight, 17.5% with 19.3%.

  Guard checks every order (decision 0014), and the trades go out in
  equal-spaced child orders (decision 0013).

Efe chose Alpaca to start real money with about 500 USD, and no leverage for
the nightly plan for now (decision 0009). A second strategy, intraday
momentum on the Nasdaq-100 (decision 0011), has its own steps in Phase 1b;
the first is done: QQQ's 1-minute bars are in the store (#22).

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
| 10. Report against buy-and-hold and equal weight: `qfin backtest` (decision 0015) | Claude | done, #29 |

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
- [x] Universe rule written down (decision 0005), and its survivorship bias
      stated in the report (step 10).
- [x] Exchange calendar with holidays and half days; all timestamps UTC.
- [x] Look-ahead rule: decide after bar t closes, fill no earlier than bar t+1
      (step 5: the engine stops a run that breaks it).
- [x] Split- and dividend-adjusted prices.
- [ ] → Phase 2, step 1 (decision 0017). Map renamed tickers (RE → EG, PEAK → DOC, IR → TT, BHGE → BKR,
      WYND → TNL, CDAY → DAY, ARNC → HWM) so their earlier index years have
      prices; with a few acquired companies EODHD lacks, about 0.67% of
      member-days have no prices today. The same map finds 1-minute bars for
      stocks whose earlier ticker the membership file does not show (BKNG was
      PCLN until 2018, AABA was YHOO until 2017).
- [ ] → Phase 3 (decision 0017). Speed up the nightly download, 17 minutes for 767 codes: parallel
      requests or EODHD's bulk end-of-day endpoint.
- [ ] → Phase 3 (decision 0017). Add each day's 1-minute bars in the nightly job once a backtest runs in
      the cloud. Until then `qfin intraday` refreshes them locally in about 3
      minutes; on S3 they should be one file per day, because versioning would
      otherwise keep a copy of every stock's whole-year file each night.
- [x] Backtests read the stored Parquet file, never a fresh download: two
      yfinance downloads of the same history can differ in the fifth decimal
      (`qfin backtest`, step 10).
- [x] Event loop: the same `Strategy` class runs on history and, in Phase 3, live.
- [x] Selling a stock that leaves the trading universe at the new year: its
      January minutes are downloaded too, 6–10 stocks a year (step 7c).
- [x] Stock-days without minutes. Step 7f filled EODHD's holes from Alpaca's
      SIP history (#23); neither provider has VIAC's minutes after 16
      February 2022, when it became PARA, nor DWDP's after 31 May 2019, when
      it became DD. The backtest trades a stock only on its own bars (decision
      0014), and the report counts the stock-days the plan held a stock
      without minutes: 72 of 54,912 over 2016–2026 (decision 0015).
- [x] A held stock whose bars stop: renamed (DWDP → DD in 2019, VIAC → PARA
      in 2022) or bought for cash (TWTR, 54.20 USD on 27 October 2022). On
      the first session without its minutes, a Delisting pays it out at its
      last price (decisions 0013 and 0015); over 2016–2026, 2 held stocks were.
- [x] Daily plan from the CVXPY optimiser with Ledoit-Wolf covariance, holding
      as many stocks as the capital can hold.
- [x] The band: trade a stock only when its gap is larger than its own
      volatility allows (decision 0006, step 7b), and then only as far as the
      band's edge (decision 0007, step 7d).
- [ ] → Phase 2, step 5 (decision 0017). Fewer swaps at small capital: at 30,000 USD the plan holds its top 10,
      and stocks entering and leaving it, 42 times in 2025, cost 16 of the 19
      bps a year whatever the band (decision 0006). A buffer (keep a held
      stock while it ranks within the top 12) halved the trades over
      2017–2026 without raising the Sharpe ratio (decision 0007); measure it
      again in the step 10 backtest.
- [x] Timing without ML: equal-spaced child orders, three from 10:30 to
      15:30; a day's trades wait until they are worth 20 USD together
      (decision 0013, step 8).
- [ ] → Phase 2, step 5 (decision 0017). Confirm decision 0013's schedule in the step 10 backtest, net of
      costs, with walk-forward tests, on the backtest of step 10. Its measurements: the number of
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
- [ ] → Phase 2, step 5 (decision 0017). The simulated broker fills a fractional buy under 1 USD, which Alpaca
      refuses. Only a buy Guard cuts can be that small: 4 in 2025, at 30,000
      and 5,000 USD (decision 0014).
- [x] Report: return, Sharpe, maximum drawdown, turnover, cost share, against
      buy-and-hold (SPY) and equal-weight baselines, and the days: the share
      of winning days, the worst day. Capital levels: 500, 5,000 and 30,000
      USD (`qfin backtest`, decision 0015).
- [ ] → Phase 2, step 6 (decision 0017). Report variants: volatility targeting and leverage, net of borrowing
      costs (decision 0009).
- [x] One broken bar stopped the nightly ingest. On 13 October 2026 EODHD's
      bar of LUMN for the 12th opened at 5.25, above its high of 5.05, and
      validation failed. Such a bar's high and low are now widened to its open
      and close, with a warning (decision 0016).

**Gate M1:** one command produces the same report from the same inputs every
time; results after costs compared with the baselines. Both hold with
`qfin backtest` (decision 0015). **Passed on 13 October 2026**; the items
still open above move to Phase 2 and 3 (decision 0017).

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

### Phase 2: Alpha AI v1 (13 October 2026 – 31 January 2027)

The ML picks the stocks the nightly plan holds, and must prove it adds
something after costs (decision 0017; the research is in
[docs/research](research/machine-learning-for-the-plan.md)). Slower during
January exams.

1. [ ] **Data, point in time.**
   - Map renamed tickers (from Phase 1).
   - Fundamentals from SEC EDGAR's XBRL company facts, each figure from its
     `filed` date; Efe chooses the contact e-mail SEC asks for.
   - Industries from SIC codes, and earnings dates.
2. [ ] **A feature store** of price, volume and fundamental features for
   every member:
   - the same batch and incremental values (feature parity);
   - a test that no feature uses data from after its date.
3. [ ] **The test harness.**
   - Labels: the rank of the coming 21- and 5-session returns, from the
     close after the decision.
   - Purged walk-forward with an embargo, and purged cross-validation for
     choices.
   - A registry of every variant tried; the deflated Sharpe ratio and the
     probability of backtest overfitting.
4. [ ] **Models.**
   - Ridge and LightGBM (needs `libgomp1`), averaged by rank; a small
     neural-network ensemble as challenger.
   - A model registry (version, training dates, data range) and a
     champion/challenger rule.
   - Monthly retraining.
5. [ ] **The portfolio from the forecasts.** Compare, in `qfin backtest`, at
   500, 5,000 and 30,000 USD and net of costs:
   - the top stocks at equal weight;
   - a tilt on the minimum-variance plan;
   - mean-variance with a trading-cost penalty.

   Also here: the buffer (decision 0007), decision 0013's schedule, and
   Alpaca's 1 USD minimum in the simulated broker (from Phase 1).
6. [ ] **Risk.**
   - HAR volatility forecasts from the 1-minute realised variance.
   - Volatility targeting and leverage as variants, net of borrowing costs
     (decision 0009).
7. [ ] **A health check:** if the forecasts' distribution drifts, the plan
   falls back to the Phase 1 plan.
8. [ ] **Forward record.** From the first frozen model on, each night's
   predictions are committed to the public repository before the open.
9. [ ] **Optional, last:** an intraday timing model (decisions 0001 and 0010).
   - Revisit Guard's limit of 3 orders a stock a day. Efe expects to raise
     it once a timing model shows a measured gain from more orders; with
     decision 0010's signal, 31 orders beat 3 by 1.26 bps a trade
     (decision 0013).
   - The change is Efe's, by hand in `guard.yaml`, with a decision record.

**Gate M2:** the ML plan's active return over the same stocks at equal
weight must pass three tests:
- a deflated Sharpe ratio of at least 0.95, counting every variant tried;
- positive after 10 bps of costs in most years;
- and the plan beats SPY at similar risk.

"ML adds nothing" is a valid result; Phase 3 then paper trades the Phase 1
plan.

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
