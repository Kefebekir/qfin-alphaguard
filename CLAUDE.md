# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

QFin-AlphaGuard is a trading system for US equities, built in the open by
Kutalp Efe Bekir (Efe). A nightly Python job builds a target portfolio, an
intraday engine decides every minute when to execute the changes, and every
order is checked against explicit risk rules (Guard) before it reaches the
broker. `README.md` describes the design and the current status.

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
uv sync                      # install dependencies
uv run pytest                # run the tests
uv run ruff check .          # lint
uv run ruff format .         # format
uv run qfin --synthetic      # run the data pipeline without network access
```

CI runs ruff, the tests and a Docker build on every push and pull request.
Both ruff commands must pass before a commit.

## Layout

- `src/qfin_alphaguard/events.py`: immutable event types shared by the
  backtester and the live engine
- `src/qfin_alphaguard/data/`: download, validation, Parquet and DuckDB
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
