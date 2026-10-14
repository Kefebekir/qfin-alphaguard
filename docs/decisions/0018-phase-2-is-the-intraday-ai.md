# 0018: Phase 2 is the intraday AI; its choices are shown day by day

- **Status:** accepted
- **Date:** 2026-10-14
- **Changes:** decision 0017's Phase 2 goal. Its monthly stock ranking is set
  aside as research; Phase 1b's QQQ steps become Phase 2's.

## Context

- **Efe's direction.** QFin-AlphaGuard is an intraday project. Its machine
  learning has to work within the day, and Efe wants to see the system's
  choices day by day. Decision 0017 had pointed Phase 2 at a monthly ranking
  of stocks for the nightly plan.
- **The research** is in `docs/research/intraday-machine-learning.md`, and
  every test in `research/intraday_pilot/`: walk-forward, 2018 to October
  2026 out of sample, after costs, on our own minutes and quotes.
- **What it found:**
  - **One robust intraday edge: the market's trend within the day, on QQQ**
    (decision 0011). At a 2× cap it earned 21.2% a year with a Sharpe ratio
    of 1.62 in 2018–2026, and a Sharpe ratio of 1.26 after its paper's data
    ended (May 2024 – October 2026).
  - **Machine learning did not beat it** anywhere it could act on QQQ:
    choosing or sizing the day, making the half-hourly decisions,
    confirming the rule's decisions, or forecasting the next 15 minutes
    from 36 assets with the method of Aleti, Bollerslev and Siggaard
    (2025).
  - **Single-stock intraday signals are real but not ours to take.** The
    end-of-day reversal is 1–2 bps and fading; stocks-in-play breakouts
    fail with 2 bps of slippage; the open-to-close ML of Ghosh, Neufeld and
    Sahoo is the opening-gap reversal, which lives in the opening auction
    and its first minute. A model that ranks every member at 09:35 finds a
    real signal, but the stocks it picks have a half spread of 14 bps then,
    and at the bid and ask it loses 12.6 bps a day.

## Decision

**1. The intraday AI is a decision engine on QQQ.** The noise-area rules of
decision 0011 make the decisions: at 10:00 and every half hour to 15:30,
long, short or flat, flat at every close. It is the champion every model has
to beat.

**2. Every decision is shown, with its reason.**
- **Now:** the day book (`research/intraday_pilot/daily_choices.py`) replays
  the rules on the stored minutes and shows each day's chart and decisions.
  It matches the research results exactly. Its page holds licensed prices
  and stays in the git-ignored `data/`.
- **In the engine:** the strategy writes a decision log, the same in a
  backtest, in paper trading and live: the time, what it saw, the choice,
  the reason and the order. The day book reads it, and in paper trading
  adds the fills.

**3. Machine learning has to earn its place, forward.**
- A model runs beside the rule in shadow: at every decision it logs its
  forecast and the position it would take, without trading.
- Before its record starts, the model is frozen (commit, training dates,
  hash of its weights) and its test is written down: the paired daily
  difference against the rule, with a t-statistic above 2, counting every
  challenger tried.
- Only a model that passes trades, and only with Efe's decision record.
- The first challengers are the ones with the best evidence: the
  factor-zoo ensemble, a model that sizes the QQQ day, and the 09:35 stock
  ranker. Each is expected to lose; the shadow record says so cheaply.

**4. Phase 2's steps** (the roadmap has the detail):
1. The research, with the day book (this decision).
2. The QQQ strategy in our engine. Efe writes its rules; Claude prepares
   the skeleton, the spec and the tests. It must reproduce the research day
   by day.
3. The decision log, and the day book reading it.
4. Guard's limits for an intraday strategy, set by Efe in `guard.yaml`:
   round trips and reversals, the 2× cap, the daily loss limit.
5. The shadow record for ML challengers: frozen models, their written
   tests, and their forecasts logged at every decision, first on backtest
   replays.

Paper trading on Alpaca, with the challengers in shadow and the day book
every evening, is Phase 3; the broker and instrument for real money from the
UK (decision 0011) belong to Phase 3b.

**5. Gate M2.**
- The engine's backtest reproduces the research result for every day.
- The decision log and the day book work on backtests.
- Every challenger is frozen, with its test written down, before its
  record starts.

**6. What is set aside.**
- Decision 0017's monthly stock ranking, with its pilot, stays in
  `research/ml_pilot/` and its research document. It can come back for the
  nightly plan by a later decision.
- Order-book models (no data, no fast enough connection), stocks-in-play
  breakouts, the end-of-day reversal and open-to-close stock ML.

**7. What does not change:** no look-ahead; UTC timestamps; Guard's limits
set by hand; paper trading only until Efe turns live trading on; no secrets
in the repository; results reported as they are.

## Consequences

- **The honest headline is that ML lost** to a simple rule in every
  intraday test. The project's AI is, for now, a transparent rule, with ML
  measured beside it in the open.
- **Proof takes time.** At the rule's Sharpe ratio of 1.24 out of sample,
  2.6 years of live results reach a t-statistic of 2; a challenger's small
  improvement could take decades to prove. Paper trading first proves that
  the system trades as its backtest says.
- **QQQ needs a UK route** for real money (PRIIPs): a Nasdaq-100 CFD or
  spread bet, or QQQ at Alpaca if its UK accounts allow it.
- **New research data,** all in the git-ignored `data/`: 15 more ETFs'
  minutes (`qfin etf-minutes`), every S&P 500 member's first 16 minutes
  since 2016, and quotes for the stocks the morning tests pick.
