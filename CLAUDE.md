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
uv run --env-file .env qfin intraday  # 1-minute bars, after ingest
uv run qfin plan                      # the next session's target portfolio, after ingest
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
- `src/qfin_alphaguard/backtest/`: the event loop and the simulated broker's
  interface
- `guard.yaml`, `src/qfin_alphaguard/guard/`: the risk limits and the code that
  loads and checks them
- `src/qfin_alphaguard/data/`: download, validation, S&P 500 membership by date,
  splits and dividends, Parquet and DuckDB
- `src/qfin_alphaguard/features/`, `estimation/`, `optimize/`: returns,
  covariance and the optimiser
- `infra/aws/`: the nightly ECS Fargate job that runs `qfin`
- `tests/`: one test module per source module

## Rules the code must keep

- **No look-ahead.** A decision may use only data available at that moment;
  the backtester fills an order no earlier than the next bar.
- **UTC everywhere.** Every timestamp is a timezone-aware UTC `datetime`.
- **Events validate themselves** in `__post_init__` and are frozen dataclasses.
- **Risk limits belong in `guard.yaml`** and change only by hand, never from
  code or a model.
- **No tests, no merge.** Every new module comes with tests.

## Never

- Commit secrets. API keys and account details stay in `.env`, which is
  git-ignored.
- Place real orders or enable live trading. Paper trading only, unless Efe
  turns it on himself.
- Change the `qfin` command line without updating
  `infra/aws/task-definition.template.json`; the nightly AWS job depends on it.
