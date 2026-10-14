# Intraday pilot (decision 0018)

Can machine learning make money within the day, for an account like ours,
after costs? And how can Efe see the system's choices day by day? This is
research code, outside the package and the live path. The full write-up is
in [docs/research/intraday-machine-learning.md](../../docs/research/intraday-machine-learning.md).

Every test is walk-forward: a model for a year is fitted on earlier years
only, and the years 2018 to October 2026 are out of sample. Costs are
Alpaca's fees and the spread or slippage stated in each script.

## Data

- **Stored 1-minute bars** (`data/raw/intraday/`): the trading universe's
  50 stocks of each year, QQQ, and the ETFs below.
- **Daily bars** of every S&P 500 member (`data/raw/prices.parquet`).
- **Each year's 50 stocks:** `research/ml_pilot/universes.py` writes them.
- **ETFs:** `uv run --env-file .env qfin etf-minutes SPY IWM DIA TLT GLD HYG XLK XLF XLE XLV XLI XLY XLP XLU XLB`.
- **The first 16 minutes of every S&P 500 member** since 2016, for the gap
  check and the 09:35 model: `opening_minutes.py`, about 35 minutes from
  Alpaca.
- **NBBO quotes** at 09:31, 09:35 and 15:59 for the stocks the morning tests
  pick: `gap_quotes.py`, two to three hours, because Alpaca answers at most
  200 requests a minute. Both downloads keep a file a year and can resume.

## Run

The outputs go under `data/intraday_pilot/`, which git ignores: they hold
licensed prices.

```bash
P=data/intraday_pilot
mkdir -p $P
uv run python research/ml_pilot/universes.py data/raw/prices.parquet $P/universes.json
uv run python research/intraday_pilot/slices.py data/raw data/raw/prices.parquet $P/universes.json $P/slices.parquet
uv run python research/intraday_pilot/eod_reversal.py $P/slices.parquet
uv run python research/intraday_pilot/rest_of_day.py $P/slices.parquet
uv run python research/intraday_pilot/orb.py data/raw data/raw/prices.parquet $P/universes.json
uv run python research/intraday_pilot/qqq_momentum.py data/raw $P/qqq_days.parquet
uv run python research/intraday_pilot/qqq_ml.py $P/qqq_days.parquet
uv run python research/intraday_pilot/qqq_har.py data/raw $P/qqq_days.parquet
uv run python research/intraday_pilot/qqq_halfhour.py data/raw
uv run python research/intraday_pilot/market_leadlag.py data/raw data/raw/prices.parquet $P/leadlag.parquet
uv run python research/intraday_pilot/open_to_close.py data/raw/prices.parquet $P/open_to_close_scores.parquet
uv run --env-file .env python research/intraday_pilot/opening_minutes.py data/raw/prices.parquet $P/opening
uv run python research/intraday_pilot/gap_check.py data/raw/prices.parquet $P/opening
uv run --env-file .env python research/intraday_pilot/gap_quotes.py data/raw/prices.parquet $P/gap_quotes
uv run python research/intraday_pilot/gap_costs.py data/raw/prices.parquet $P/gap_quotes
uv run python research/intraday_pilot/morning_ml.py data/raw/prices.parquet $P/opening $P/morning_scores.parquet
uv run python research/intraday_pilot/morning_picks.py $P/morning_scores.parquet $P/morning_picks.parquet
uv run --env-file .env python research/intraday_pilot/gap_quotes.py data/raw/prices.parquet $P/morning_quotes --picks $P/morning_picks.parquet
uv run python research/intraday_pilot/gap_costs.py data/raw/prices.parquet $P/morning_quotes --scores $P/morning_scores.parquet --signal gbt
uv run python research/intraday_pilot/daily_choices.py data/raw 2024-05-01 data/reports/qqq_day_book.html $P/decision_log guard_qqq.yaml
uv run python research/intraday_pilot/engine_check.py data/raw $P/qqq_days.parquet
uv run python research/intraday_pilot/guard_needs.py data/raw guard.yaml guard_qqq.yaml
```

Set `OMP_NUM_THREADS=8` (and the same for OpenBLAS and MKL) when two
scripts run at once: models that each take every core slow each other down
many times over.

| Script | What it does |
| --- | --- |
| `slices.py` | Each stock-day's returns over the parts of the session (overnight, first half hour, middle, last hours), its volume and realised variance. |
| `eod_reversal.py` | End-of-day reversal (Baltussen, Da and Soebhag, 2025) in our 50 stocks. |
| `rest_of_day.py` | At 10:00, ridge and gradient-boosted trees rank our 50 stocks by their return to the close. |
| `orb.py` | The 5-minute opening range breakout on stocks in play (Zarattini, Barbon and Aziz, 2024), with slippage. |
| `open_to_close.py` | Ghosh, Neufeld and Sahoo (2022) on every S&P 500 member: random forest and boosted trees on 93 return features pick the stocks to hold from the open to the close. |
| `opening_minutes.py` | Downloads the first 16 minutes of every member's session from Alpaca. |
| `gap_check.py` | The opening-gap reversal bought at later, tradeable prices (09:31, 09:35, 09:45). |
| `gap_quotes.py` | Downloads the NBBO quotes, at 09:31, 09:35 and 15:59, of the stocks the gap reversal or a model picks. |
| `gap_costs.py` | Those picks bought at the ask and sold at the bid, or at the official close, after Alpaca's fees. |
| `morning_ml.py` | At 09:35, ridge and boosted trees rank every member by its return to the close, from the gap and the first 5 minutes. |
| `morning_picks.py` | The 09:35 model's daily top 10, for `gap_quotes.py`. |
| `qqq_momentum.py` | QQQ intraday momentum (decision 0011), re-implemented, with each day's record. |
| `qqq_ml.py` | ML at 10:00: skip or resize QQQ momentum's day. |
| `qqq_har.py` | Size QQQ momentum with a HAR volatility forecast. |
| `qqq_halfhour.py` | ML makes QQQ's half-hourly decisions, alone or as a meta-label on the rule. |
| `market_leadlag.py` | Predict QQQ's next 15 minutes from the last 15 minutes of 36 assets: seven models and their ensemble (Aleti, Bollerslev and Siggaard, 2025). |
| `daily_choices.py` | The day book: runs the engine's QQQ strategy, stores its decision log, and shows every decision day by day as a page to open in a browser. |
| `engine_check.py` | Phase 2, step 2: the engine's strategy (`momentum.py`) against `qqq_momentum.py`, day by day, at no cost and at QQQ's measured spread. |
| `guard_needs.py` | Phase 2, step 4: what Guard's limits do to the QQQ strategy, with the nightly plan's `guard.yaml`, the QQQ account's `guard_qqq.yaml` and daily loss limits of 2–5%, against no Guard (decision 0019). |

- **Time:** each script takes seconds to a minute on 20 cores, except
  `open_to_close.py` (about 12 minutes) and the two downloads.
- **Determinism:** the models use fixed seeds; the same data gives the same
  output.

## The day book

`daily_choices.py` runs the engine's QQQ strategy (`momentum.py`) on the
stored minutes, at Alpaca's fees and QQQ's measured half spread, stores its
decision log (`decision_log.py`) and writes `data/reports/qqq_day_book.html`
from it: the days with their results after costs, and for each day its chart
(price, the strategy's noise band, VWAP, fills) and every decision with the
reason the strategy logged. Every order passes Guard with the QQQ
account's limits, `guard_qqq.yaml` (decision 0019), with `max_order_usd`
lifted: it is set for a 30,000 USD account, and this one grows several
times over in ten years. Guard changes none of the strategy's orders then.

The page holds licensed prices, so it stays under the git-ignored `data/`
and is for Efe's own use: do not commit or publish it.
