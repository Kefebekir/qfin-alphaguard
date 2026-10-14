# QFin-AlphaGuard

A trading system for US equities. A nightly job builds a target portfolio, an
intraday engine decides every minute when to execute the changes, and every order
is checked against explicit risk rules before it reaches the broker.

The goal is net profit after costs and tax, measured against simple alternatives
an investor could buy instead. I am building it in the open; the status table says
exactly what works today and what does not.

## Status

| Stage | State |
| --- | --- |
| Project skeleton, tests, CI | done |
| Data layer: daily prices, validation, Parquet, DuckDB | done |
| Docker image and scheduled AWS run | done |
| Covariance estimators (sample, Ledoit-Wolf, EWMA) and minimum-variance optimiser | done |
| Phase 0: event types, risk configuration, repository skeleton | done |
| Phase 1: event-driven backtester with costs and Guard rules | done; Gate M1 passed on 13 October 2026 |
| Phase 2: the intraday AI: QQQ's decisions within the day, shown day by day, with ML measured against them | in progress |
| Phase 3: paper trading on Alpaca | planned |
| Phase 3b: small real-money trading with scaling and stop rules | planned |
| Phase 4: C++ execution engine | planned |
| Phase 5: FPGA market-data and pre-trade risk benchmarks | planned |

Phase 1 is done: `qfin backtest` gives the first full report (below). Phase 2
is the intraday AI
([decision 0018](docs/decisions/0018-phase-2-is-the-intraday-ai.md)): a
decision engine that trades the Nasdaq-100's trend within the day, shows
every decision with its reason, and lets machine learning trade only after
it beats the plain rules forward. Its research is in
[docs/research](docs/research/intraday-machine-learning.md), and its steps
in [docs/ROADMAP.md](docs/ROADMAP.md).

## First results

`qfin backtest` replays the nightly plan on the stored 1-minute bars from 5
January 2016 to 9 October 2026, with Alpaca's costs and Guard. The baselines
are before costs:
- the same stocks at equal weight, bought anew each 1 January and held;
- SPY.

| | A year | Volatility | Sharpe | Max drawdown |
| --- | --- | --- | --- | --- |
| The plan, 500 USD | +10.7% | 12.7% | 0.87 | −23.3% |
| The plan, 30,000 USD | +10.8% | 12.7% | 0.88 | −23.3% |
| The same stocks, equal weight | +17.5% | 19.3% | 0.93 | −31.7% |
| SPY | +15.3% | 17.7% | 0.89 | −33.7% |

**The plan loses to both baselines.** It is the least-variance mix of each
year's 50 most liquid S&P 500 stocks.
- It takes about a quarter less risk than SPY, but earns 4.5 points a year
  less.
- Its Sharpe ratio is lower than both baselines', so taking more risk would
  not close the gap.
- Costs are not the reason: 0.05–0.20% a year.

The goal is still to beat both baselines after costs; Phase 2's forecasts
are the next attempt
([decision 0015](docs/decisions/0015-the-backtest-and-its-first-report.md)).

### Within the day

The intraday research
([docs/research](docs/research/intraday-machine-learning.md)) tested what
the literature offers on our own minutes and quotes, walk-forward from 2018
to October 2026, after costs:
- **One edge holds up: the Nasdaq-100's trend within the day** (decision
  0011). Its rules earned 21.1% a year at a 2× cap with a Sharpe ratio of
  1.61, and a Sharpe ratio of 1.26 after its paper's data ended in April
  2024.
- **Machine learning did not beat it.** Models that chose or sized the day,
  made the half-hourly decisions, or forecast the next 15 minutes from 36
  assets all did worse than the plain rules.
- **Single stocks have real intraday signals, but not ones a small account
  can take:** they are a few basis points, below the spread, or live in the
  opening auction.

So the rules make the decisions, every decision is shown day by day with
its reason, and ML models run beside them in shadow until one proves
better.

An earlier pilot for the nightly plan, a monthly ranking of stocks, is set
aside with its result: about 8% a year over equal weight after costs, but
not yet distinguishable from luck
([decision 0017](docs/decisions/0017-phase-2-ml-picks-the-stocks.md)).

## How it works

The system runs three loops at three speeds:

1. **Nightly, in Python.** Download and validate data, build features, estimate
   returns and risk, optimise target weights and write a `daily_plan.json`.
2. **Every one-minute bar, in the engine.** Update features, score each stock with
   the intraday model and decide whether to send the next slice of the day's
   rebalance or wait.
3. **Every order, in Guard.** Check the order against its account's limits
   (`guard.yaml` for the nightly plan, `guard_qqq.yaml` for QQQ): position and
   order size limits, a price collar, orders a minute and a day, and for the
   nightly plan no same-day reversals and a daily loss limit. An order that only
   reduces a position faces only the price collar and the orders a minute, so a
   position can always be closed. Guard is a rule-based referee, not a model, and
   its limits change only by hand.

Research and the models use the S&P 500 as it was on each day, including
companies that later left the index; the system trades the most liquid of these
stocks. UK retail accounts cannot buy US-domiciled ETFs, so the nightly plan
holds single stocks; the intraday strategy trades QQQ on the paper account,
and real money from the UK will need a Nasdaq-100 CFD or another route
(decision 0011).

### Design choices

- **The intraday model times trades; it does not change targets.** Fewer, larger
  trades keep commissions low and avoid same-day round trips. It also makes the
  model's value measurable: execution price against arrival price, compared with
  an equal-spaced schedule that uses no model.
- **Research code never sits in the live path.** Training and optimisation run at
  night and hand the engine a plan.
- **The FPGA does not make live trading faster.** A round trip through a retail
  broker takes tens to hundreds of milliseconds. The FPGA work is a lab pipeline
  on replayed NASDAQ ITCH data, benchmarked against a C++ model, plus an optional
  independent pre-trade risk gate.
- **Quantum optimisation is research only.** A QUBO formulation of asset selection
  will be compared with classical solvers offline. It is not part of the trading
  system.

More detail: [architecture](docs/ARCHITECTURE.md), [roadmap](docs/ROADMAP.md)
and [decision records](docs/decisions/).

## What counts as success

- Net of costs and tax, beat an equal-weight buy-and-hold of the same stocks and an
  S&P 500 UCITS ETF, at similar risk.
- Evidence comes in layers: long out-of-sample walk-forward backtests first, then
  paper trading to show the live system behaves like the backtest, then small real
  money that grows only while live results stay inside the backtest's expected
  range.

Two things I care about more than the results:

- **No look-ahead.** Every decision may use only data available at that moment,
  and the backtester fills an order no earlier than the next bar. Tests will cover
  this as each module lands.
- **Honest reporting.** If a method loses to the naive baseline, the README says so.
  I ran into this before on a
  [quantum hardware experiment](https://github.com/Kefebekir/quantum-hardware-noise-comparison)
  where error mitigation underperformed the unmitigated baseline, and reporting that
  was more useful than hiding it.

## Known limitations

- **Survivorship bias, mostly removed.** The data covers every stock that was in
  the S&P 500 on each day since 2015, including companies that were later bought
  or delisted. What remains: the membership history is kept by one person rather
  than S&P, and stocks outside the S&P 500 are not covered.
- **1-minute bars only for the trading universe.** Daily bars cover every S&P 500
  member; 1-minute bars, from 2016, only each year's 50 most liquid members. They
  come from EODHD, and Alpaca's SIP history fills the days EODHD lacks, such as
  those of the Chesapeake Energy that went bankrupt in 2020. Neither has
  ViacomCBS's minutes after it became Paramount in February 2022.
- **Adjusted prices are not point-in-time.** Prices are adjusted for splits and
  dividends, so an old price differs from the one quoted on that day. Returns are
  right; dividing by the `adjustment` column gives the price as traded, and
  `volume / split_factor` the shares traded, for share counts, commissions and
  ranking stocks by liquidity without using later dividends.

## Getting started

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Kefebekir/qfin-alphaguard.git
cd qfin-alphaguard
uv sync
uv run pytest

# run the data pipeline on generated data (no network needed)
uv run qfin ingest --synthetic

# run it on real market data: needs an EODHD API key in .env (EODHD_API_KEY=...)
uv run --env-file .env qfin ingest

# 1-minute bars for each year's trading universe: needs Alpaca's keys in .env too
uv run --env-file .env qfin intraday

# replay the nightly plan on the stored bars and report it against the baselines
uv run qfin backtest
```

## Deployment

The data pipeline runs on AWS every weekday morning:

- The Docker image is stored in **Amazon ECR**.
- **EventBridge Scheduler** starts it as an **ECS Fargate** task at 06:45 London
  time, Tuesday to Saturday, so each run picks up the previous trading day's close.
- The task runs `qfin ingest`: it downloads daily prices from **EODHD** for every
  stock that has been in the S&P 500 since 2015, and for SPY, the backtest's
  benchmark; validates them and writes Parquet to **S3** (with versioning, so
  every day's file is kept).
- The EODHD key is kept in **SSM Parameter Store** and handed to the task as a
  secret.
- Logs go to **CloudWatch Logs**.
- Each component has its own IAM role with the minimum permissions it needs: the
  task can only write under `data/` in one bucket, ECS can read only the EODHD key,
  and the scheduler can only start this one task.

IAM policies and the task definition template are in `infra/aws/`.

## Tech

Python 3.12, Polars, Parquet, DuckDB, NumPy, scikit-learn, CVXPY, pytest, ruff,
Docker, AWS (ECR, ECS Fargate, EventBridge Scheduler, S3), GitHub Actions.

Planned: the Alpaca API, C++20, and SystemVerilog on an Artix-7 FPGA.

## License

MIT
