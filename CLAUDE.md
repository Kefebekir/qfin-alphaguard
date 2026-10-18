# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

QFin-AlphaGuard is a trading system for US equities, built in the open by
Kutalp Efe Bekir (Efe). A nightly Python job builds a target portfolio, an
intraday engine decides every minute when to execute the changes, and every
order is checked against explicit risk rules (Guard) before it reaches the
broker. `README.md` describes the design and the current status.

## Plan and status

- `docs/ROADMAP.md`: phases, gates and **where we are now**. Read its "Where
  we are" section at the start of a session; update it when a step is done.
- `docs/ARCHITECTURE.md`: the three loops, data flow, execution logic, the risk
  file and the FPGA's role.
- `docs/decisions/`: why the big choices were made. Add a new numbered record
  instead of silently changing a decision.

## Working agreement

Efe owns this project and must be able to explain every line of it in an
interview. Claude works as a pair programmer, not as the author.

- **Efe writes the core logic**: Guard rules, execution timing, the simulated
  broker and anything that decides a trade. Claude prepares skeletons, tests
  and explanations, then reviews Efe's code honestly.
- **Explain before changing.** Describe the approach and the reason for it
  before editing code, and point out the parts worth understanding.
- **Small, reviewable steps.** One logical change per commit, one step per
  pull request. Efe reviews and merges every pull request himself.
- **Talk to Efe in Turkish.** Code, comments, commit messages and pull
  requests stay in English.
- **Report results honestly.** If a method loses to the baseline, say so.

## Commands

```bash
uv sync                               # install dependencies
uv run pytest                         # run the tests
uv run ruff check .                   # lint
uv run ruff format .                  # format
uv run qfin ingest --synthetic        # run the data pipeline without network access
uv run --env-file .env qfin ingest    # real data; needs EODHD_API_KEY in .env
uv run --env-file .env qfin intraday  # 1-minute bars from EODHD and Alpaca, after ingest
uv run --env-file .env qfin etf-minutes QQQ  # QQQ's 1-minute bars from Alpaca
uv run qfin plan                      # the next session's target portfolio, after ingest
uv run qfin backtest                  # the plan on the stored minutes, against SPY; after intraday
uv run --env-file .env qfin live      # today's session of the QQQ strategy on Alpaca's paper account
uv run --env-file .env qfin live-report --s3-bucket qfin-alphaguard-kefebekir  # a session's report
```

CI runs ruff, the tests and a Docker build on every push and pull request.
Both ruff commands must pass before a commit.

## Layout

- `src/qfin_alphaguard/events.py`: immutable event types shared by the
  backtester and the live engine
- `src/qfin_alphaguard/sessions.py`: when the exchange is open, in UTC
- `src/qfin_alphaguard/strategy.py`, `portfolio.py`: the Strategy interface
  and its read-only View, and the books; shared with the live engine
- `src/qfin_alphaguard/plan.py`: the daily plan, the portfolio to hold the
  next day and each stock's volatility
- `src/qfin_alphaguard/rebalance.py`: the band; which stocks to trade at the
  open, and how many shares
- `src/qfin_alphaguard/momentum.py`: QQQ intraday momentum (decisions 0011
  and 0018): the noise-area rules and the strategy around them
- `src/qfin_alphaguard/decision_log.py`: the decision log as tables: each
  DecisionNote and fill, stored as Parquet
- `src/qfin_alphaguard/execution.py`: when the day's trades go out; the
  equal-spaced child orders that Phase 2's timing model has to beat
- `src/qfin_alphaguard/backtest/`: the event loop, the simulated broker,
  `qfin backtest`'s runs and report, and the shadow record (`shadow.py`)
- `src/qfin_alphaguard/challengers.py`, `evidence.py`, `challengers/`: the
  shadow record's frozen challenger cards and the betting test that judges
  them against the rule (decision 0020)
- `src/qfin_alphaguard/live/`: the live engine on Alpaca's paper account
  (decision 0021); `orders.py`, each order's status at the broker, by Efe's
  rules; `alpaca_broker.py`, orders out and their status back; `streams.py`,
  the trade updates and the minute bars as they arrive; `journal.py`, every
  input on disk before the engine acts, replayed after a restart;
  `reconcile.py`, the engine's orders and positions against Alpaca's;
  `engine.py`, the core that decides, with no input or output, and replays a
  day from its journal; `warmup.py`, the strategy relearning the sessions
  before today each morning; `runner.py`, a session from the morning to the
  close (`qfin live`), and the stop that closes by Alpaca's positions;
  `report.py`, a session against backtests of it (`qfin live-report`)
- `guard.yaml`, `guard_qqq.yaml`, `src/qfin_alphaguard/guard/`: the risk limits
  of the nightly plan's and the QQQ strategy's accounts, and the code that
  loads and checks them
- `src/qfin_alphaguard/data/`: downloads from EODHD and Alpaca, validation,
  S&P 500 membership by date, splits and dividends, Parquet and DuckDB
- `src/qfin_alphaguard/features/`, `estimation/`, `optimize/`: returns,
  covariance and the optimiser
- `infra/aws/`: the nightly ECS Fargate job that runs `qfin`, and the live
  engine's weekday task (`LIVE.md`)
- `tests/`: one test module per source module

## Rules the code must keep

- **No look-ahead.** A decision may use only data available at that moment;
  the backtester fills an order no earlier than the next bar.
- **UTC everywhere.** Every timestamp is a timezone-aware UTC `datetime`.
- **Events validate themselves** in `__post_init__` and are frozen dataclasses.
- **Risk limits belong in `guard.yaml` and `guard_qqq.yaml`** and change only
  by hand, never from code or a model.
- **No tests, no merge.** Every new module comes with tests.

## Never

- Commit secrets. API keys and account details stay in `.env`, which is
  git-ignored.
- Place real orders or enable live trading. Paper trading only, unless Efe
  turns it on himself.
- Change the `qfin` command line without updating
  `infra/aws/task-definition.template.json`; the nightly AWS job depends on it.
