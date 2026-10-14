# 0017: Phase 1 closes; in Phase 2 the ML picks the plan's stocks

- **Status:** accepted
- **Date:** 2026-10-13
- **Changes:** the Phase 2 plan of the roadmap. It reorders decisions 0001
  and 0010: their intraday timing model becomes Phase 2's last, optional
  step.
- **Changed by:** decision 0018. Phase 2 is the intraday AI; the monthly
  ranking below is set aside as research, and its Gate M2 replaced.

## Context

- **Gate M1 holds** (decision 0015). One command gives the same report from
  the same inputs, and it compares the results after costs with SPY and
  with the same stocks at equal weight. The plan loses to both baselines.
- **Efe's request.** He asked for the best machine learning approach for the
  project, found in the literature and made to work. Earlier design choices
  were to give way wherever the evidence says otherwise.
- **The research and a pilot on our own data are in
  `docs/research/machine-learning-for-the-plan.md`.** In short:
  - **Predictability is small,** and mostly in small, illiquid stocks and in
    short-lived signals. Stock-level monthly R² is about 0.4%, and 0.5–0.7%
    among the 1,000 largest stocks.
  - **Costs undo naive ML portfolios.** After costs, the features that matter
    are the slow ones: value and quality.
  - **Timing a day's trades is worth about 1 bp a trade** (decision 0013, and
    RL execution studies).
  - **Our pilot is promising but unproven.**
    - Setting: 20 price and volume features, every S&P 500 member,
      walk-forward 2018–2026.
    - Monthly gradient-boosted trees picking 10 of the year's 50 most liquid
      stocks beat holding all 50 by about 8% a year after 10 bps. That is
      the average over the 21 days a monthly rebalance can start on, with 3%
      to 15% depending on the day.
    - Its deflated Sharpe ratio is 0.70 of the 0.95 needed.

## Decision

**1. Gate M1 is passed, and Phase 1 ends.** Its open items move to where they
matter:

| Item | Moves to |
| --- | --- |
| Map of renamed tickers | Phase 2, step 1: ML training uses every member's history |
| Buffer of decision 0007; decision 0013's schedule | Phase 2, step 5: measured with the ML plan, which replaces the plan they were for |
| Report variants: volatility targeting, leverage | Phase 2, steps 5 and 6 |
| Alpaca's 1 USD minimum in the simulated broker | Phase 2, step 5 |
| A faster nightly download; minutes in the nightly job | Phase 3 |

**2. The ML decides what the nightly plan holds, not when it trades.**
- The model ranks every S&P 500 member by its expected return over the
  coming month (21 sessions), with the coming week (5 sessions) as a second
  horizon.
- The ranking replaces the minimum-variance weights of `plan.py`.
- EqualSpaced and Guard execute the trades as they do now.
- An intraday timing model (decisions 0001 and 0010) is the last, optional
  step. It is tried only after the nightly model passes.

**3. Data first, point in time.**
- **Renamed tickers.** Map them, so no member's history has gaps.
- **Fundamentals** from SEC EDGAR's XBRL company facts. They are free, and
  each figure is used only from its `filed` date. SEC asks every client to
  give a contact e-mail in its User-Agent; Efe decides which address.
- **Industries** from the SIC codes, and **earnings dates** from the
  filings.

**4. Models.**
- Ridge regression and gradient-boosted trees (LightGBM), averaged by rank.
- A challenger: an ensemble of small neural networks.
- Anything more complex, an LLM included, enters only if it beats these
  under the same test. News-based models are not in Phase 2: there is no
  point-in-time news feed, and a model trained on later text knows the
  future.

**5. The portfolio.**
- Long only, within Guard's limits.
- Three constructions are compared net of costs at 500, 5,000 and 30,000 USD:
  1. the top stocks by forecast, at equal weight;
  2. the forecast as a tilt on the minimum-variance plan;
  3. mean-variance with the forecasts, Ledoit-Wolf risk and a trading-cost
     penalty.

**6. The test, and Gate M2.**
- **Method.** Purged walk-forward for results; purged cross-validation and
  nested tuning for choices. Every variant tried goes in a registry.
- **To pass,** the ML plan's active return over the same stocks at equal
  weight must:
  - have a deflated Sharpe ratio of at least 0.95, counting every variant;
  - be positive after 10 bps of costs in most years;
  - and the plan must beat SPY at similar risk.
- **If it does not pass,** the result is "ML adds nothing here", and Phase 3
  paper trades the Phase 1 plan.

**7. Forward evidence.**
- From the first frozen model on, each night's predictions are committed to
  the public repository before the open.
- Phase 3 paper trades the model and compares it every day with its
  backtest.
- The minimum track record length is stated next to every forward result:
  about 4 years at an information ratio of 0.84, and 11 at 0.5.

**8. What does not change**, whatever else does:
- no look-ahead;
- UTC timestamps;
- Guard's limits set by hand;
- paper trading only until Efe turns live trading on;
- no secrets in the repository;
- results reported as they are.

## Consequences

- Phase 2 starts on 13 October 2026 instead of 7 December.
- New dependencies:
  - LightGBM, which needs the system library `libgomp1` (also in the Docker
    image);
  - SEC EDGAR access with a contact e-mail.
- The pilot's scripts are in `research/ml_pilot/`; they read the stored data
  only.
