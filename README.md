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
| Phase 1: event-driven backtester with costs and Guard rules | next |
| Phase 2: ML forecasts and intraday execution timing | planned |
| Phase 3: paper trading on Interactive Brokers | planned |
| Phase 3b: small real-money trading with scaling and stop rules | planned |
| Phase 4: C++ execution engine | planned |
| Phase 5: FPGA market-data and pre-trade risk benchmarks | planned |

Nothing from Phase 1 onwards exists yet.

## How it works

The system runs three loops at three speeds:

1. **Nightly, in Python.** Download and validate data, build features, estimate
   returns and risk, optimise target weights and write a `daily_plan.json`.
2. **Every one-minute bar, in the engine.** Update features, score each stock with
   the intraday model and decide whether to send the next slice of the day's
   rebalance or wait.
3. **Every order, in Guard.** Check the order against `guard.yaml`: position and
   order size limits, a price collar, no same-day reversals and a daily loss limit.
   Guard is a rule-based referee, not a model, and its limits change only by hand.

The universe is a small set of liquid US stocks. UK retail accounts cannot buy
US-domiciled ETFs, so single stocks are used instead.

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

- **Survivorship bias.** The ticker list contains companies that still exist and
  are large today. Companies that were large in 2015 but later failed are not
  included, so any backtest on this universe will look better than it would have
  in real time.
- **Daily bars only.** The data layer stores daily open, high, low, close and
  volume. Intraday bars come later in Phase 1.
- **Adjusted prices are not point-in-time.** Prices are adjusted for splits and
  dividends, so an old price differs from the one quoted on that day. Returns are
  right, but share counts and order sizes in a backtest are approximate.
- **Synthetic calendar.** Generated data includes market holidays; real data does
  not. This only affects tests, not results.

## Getting started

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Kefebekir/qfin-alphaguard.git
cd qfin-alphaguard
uv sync
uv run pytest

# run the data pipeline on generated data (no network needed)
uv run qfin ingest --synthetic

# run it on real market data
uv run qfin ingest
```

## Deployment

The data pipeline runs on AWS every weekday morning:

- The Docker image is stored in **Amazon ECR**.
- **EventBridge Scheduler** starts it as an **ECS Fargate** task at 06:45 London
  time, Tuesday to Saturday, so each run picks up the previous trading day's close.
- The task runs `qfin ingest`: it downloads prices, validates them and writes
  Parquet to **S3** (with versioning, so every day's file is kept).
- Logs go to **CloudWatch Logs**.
- Each component has its own IAM role with the minimum permissions it needs: the
  task can only write under `data/` in one bucket, and the scheduler can only start
  this one task.

IAM policies and the task definition template are in `infra/aws/`.

## Tech

Python 3.12, Polars, Parquet, DuckDB, NumPy, scikit-learn, CVXPY, pytest, ruff,
Docker, AWS (ECR, ECS Fargate, EventBridge Scheduler, S3), GitHub Actions.

Planned: the Interactive Brokers API, C++20, and SystemVerilog on an Artix-7 FPGA.

## License

MIT
