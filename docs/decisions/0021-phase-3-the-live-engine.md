# 0021: Phase 3, the live engine: the free IEX feed, AWS, 30,000 USD on paper

- **Status:** accepted
- **Date:** 2026-10-17
- **Builds on:** decision 0008 (Alpaca and its free feed), 0019 (Guard for an intraday
  strategy) and 0020 (the shadow record); Phase 3 of the roadmap.

## Context

- **The research is done for now.** Phase 2's tests found one robust edge, the QQQ rule
  (Sharpe 1.56 on G1's holdout, 2022 – October 2026), and nothing that beats it:
  - learned policies (step 5);
  - the news reader (N1);
  - the rule on Nikkei futures (F1);
  - a network trained by evolution strategies on 16 ETFs (G1).
- **Paper trading is the next proof.** It must show that the live system trades as its
  backtest says.
- **The data question.** Alpaca's free plan streams IEX's trades only, about 1–3% of QQQ's
  volume, while every backtest used the consolidated tape (SIP). Decision 0008 planned to
  trade on the free feed and compare it with the tape every evening.
- **NautilusTrader's design**, read for this phase, shows what a live engine needs beyond a
  backtest:
  - an order state machine;
  - reconciliation that fails closed;
  - an event log that a restart resumes from.

## Decision

**1. The rule decides on Alpaca's free IEX feed.**
- `research/phase3/iex_vs_sip.py` replays 2021 – 9 October 2026, the years Alpaca keeps IEX's
  minutes. The rule decides on IEX's minutes and fills at SIP's prices.
- **Same choices:** it holds the same position as on SIP at 98.4% of 17,136 decisions, and
  every decision of the day is the same on 85.1% of days.
- **Same results:** 17.3% a year (Sharpe 1.44) against 16.4% (1.36). The difference,
  +0.36 bps a day, is not significant (t +0.78), and daily results correlate at 0.973.
- The 99 USD a month SIP plan is not needed. The evening report still compares each day's
  decisions with a replay on the tape.

**2. The engine runs on a small AWS server,** always on through the US session, beside the
nightly job's account. Efe's laptop is not a server.

**3. The paper account starts with 30,000 USD,** not Alpaca's default 100,000. It is the
account `guard_qqq.yaml`'s `max_order_usd` (70,000, the rule's 2x and room to move) was set
for, and the size the IEX research above assumes. `live_trading` stays false.

**4. The live engine reuses everything the backtest runs:**
- the strategy (`momentum.py`), Guard and `guard_qqq.yaml`;
- the events and the decision log.

New are the parts a backtest does not need:
- **Live data:** IEX minute bars from Alpaca's WebSocket. A late or missing bar is stale
  data, which Guard already refuses to trade on.
- **The order state machine:** each order goes new → submitted → accepted → partly filled →
  filled, cancelled, rejected or expired, with a unique client order id. These are the
  roadmap's states, named as Alpaca reports them, with expired added: a day order Alpaca
  ends at the close.
  - Efe writes its transition rules; Claude writes the skeleton and the tests.
  - A cancel counts as open until the broker confirms it (decision 0019).
- **The Alpaca adapter:** orders by REST, their updates by WebSocket. A command whose outcome
  is unknown, after a lost connection, is never sent again blindly: it is resolved by asking
  for the order's state.
- **Reconciliation, at the start and through the day:**
  - positions and open orders at Alpaca are compared with the engine's;
  - a difference stops the strategy from starting;
  - a missing report is never read as flat.
- **An event log:** every event is written to disk, and a restart resumes from it.
- **The evening report:**
  - the day's fills against a replay of the same day;
  - the decisions on IEX against those on the tape;
  - slippage and the shadow record's challengers.

## Consequences

- Phase 3's checklist in the roadmap stands, and starts in October 2026 instead of February
  2027. Its live-data item is answered: Alpaca's IEX, not EODHD's EDGX.
- The C++ core, the FPGA Guard and the microstructure lab of the second architecture come
  after paper trading works. This engine's Python core is the reference they must match.
- Real money stays off until Efe turns it on, after Gate M3: at least eight weeks of paper
  trading that a replay reproduces and that stays inside the backtest's range.
