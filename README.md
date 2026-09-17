# QFin-AlphaGuard

A portfolio construction system that picks which assets to hold, decides how much
of each to hold, checks that decision against a set of explicit risk rules, and
tests the whole thing on historical data without looking ahead.

The asset selection step is also written as a QUBO problem, so classical and
quantum solvers can be run on the same formulation and compared.

I am rebuilding this from scratch in the open. The section below says exactly what
works today and what does not.

## Status

| Stage | State |
| --- | --- |
| Project skeleton, tests, CI | done |
| Data layer (ingest, validation, storage, SQL) | in progress |
| Docker + scheduled cloud run | planned |
| Expected returns and covariance | planned |
| Mean-variance optimizer with cardinality constraint | planned |
| Guard rules and walk-forward backtest | planned |
| QUBO formulation and solver comparison | planned |
| Machine-learned return forecasts | planned |

Nothing below the "in progress" line exists yet.

## Design

Data flows through eight stages:

1. **Ingest** â€” equity prices from yfinance, macro series from FRED.
2. **Validate** â€” missing data, liquidity, calendar alignment, survivorship bias.
3. **Features** â€” log returns, rolling volatility, momentum, built point-in-time.
4. **Estimate** â€” expected returns and a covariance matrix.
5. **Select** â€” which K assets to hold, classically and as a QUBO.
6. **Weight** â€” mean-variance weights for the selected assets.
7. **Guard** â€” explicit rules on position size, turnover and drawdown. This is a
   rule-based referee, not a model. It is meant to be readable and auditable.
8. **Backtest** â€” walk-forward runs with transaction costs, reported against an
   equal-weight benchmark.

Two things I care about more than the results:

- **No look-ahead.** Every feature is built only from data available at that point
  in time. This is enforced in code and covered by tests.
- **Honest reporting.** If a method loses to the naive baseline, the README says so.
  I ran into this before on a
  [quantum hardware experiment](https://github.com/Kefebekir/quantum-hardware-noise-comparison)
  where error mitigation underperformed the unmitigated baseline, and reporting that
  was more useful than hiding it.

## Getting started

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Kefebekir/qfin-alphaguard.git
cd qfin-alphaguard
uv sync
uv run pytest
```

## Tech

Python 3.12, Polars, Parquet, DuckDB, NumPy, pytest, ruff, GitHub Actions.

## License

MIT

