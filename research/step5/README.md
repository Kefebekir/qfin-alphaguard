# Phase 2, step 5 research: can anything beat the QQQ rule?

Before building the shadow record for ML challengers, this research asked what a challenger
could be: learned trading policies, dealer gamma (from prices and from real options open
interest), calendar and news effects, the overnight return, breadth across stocks and ETFs, and
the source pool's other ideas. **Nothing beat the noise-area rule out of sample.**
`LEDGER.md` lists every test and its result, `SOURCES.md` the 97 sources, and
`docs/research/intraday-machine-learning.md` (section 6) the conclusions.

## How the tests were run

- **Written down first.** A confirmatory test's hypothesis, direction, data and threshold are
  in `preregistrations/`, timestamped and unchanged since (hashes in `LEDGER.md`); its output
  is in `results/`. Several tests at once share a Bonferroni threshold.
- **No look-ahead.** Each test's model is fitted on years before its test year; ranks and
  weights use the past only.
- **Reproducible.** Re-running a script gives byte-identical output.

## Environment

Most scripts run in the project's environment (`uv run python research/step5/<script>.py`).
`policy_nets.py`, `policy_gate_check.py` and `b5.py`, and the Databento downloads, need
packages the project does not depend on (PyTorch on the CPU, hmmlearn, meteostat, databento),
kept in a separate environment under the git-ignored `data/`:

```bash
uv venv data/mlenv --python 3.12
uv export --frozen --no-dev --no-emit-project --no-hashes > data/mlenv/locked.txt
uv pip install --python data/mlenv/bin/python -r data/mlenv/locked.txt
uv pip install --python data/mlenv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python data/mlenv/bin/python -e . hmmlearn meteostat databento
```

## Data

All of it stays in the git-ignored `data/` (licensed or bulky):

- QQQ's, the 15 other ETFs' and the stocks' 1-minute bars (`qfin intraday`, `qfin etf-minutes`);
- `fetch_etf_daily.py`: dividend-adjusted daily bars of the 16 ETFs from EODHD;
- Cboe's VIX, VIX9D, VIX1D, VIX3M, VVIX and VXN histories, and FRED's 3-month T-bill (public
  CSV files, downloaded to `data/raw/cboe`, `data/raw/fred`);
- the FOMC, CPI and payrolls dates (Federal Reserve and BLS calendars, `data/raw/calendar`);
- options open interest from Databento's OPRA statistics: `gamma_download_2016_2022.py`
  (about 13 USD for QQQ and SPY) and `gamma_download_2023_2026.py` (QQQ, about 84 USD; only
  the window in which open interest is published, resumable, with a spending cap). Both read
  `DATABENTO_API_KEY` from `.env` and never print it;
- Treasury auction dates (TreasuryDirect), New York LaGuardia's hourly cloud cover (Meteostat),
  Bitcoin's daily bars (`fetch_btc.py`, Alpaca).

## Scripts

| Script | What it does |
|---|---|
| `policy_data.py` | The rule's decisions and features for 16 ETFs (or stocks, `TICKERS`; or a shifted grid, `OFFSET`) |
| `policy_nets.py` | 26 learned policies (Sharpe-trained networks, GRU, attention, gate, day sizer, kNN, tuned rule) against the rule, walk-forward |
| `policy_gate_check.py` | The gate's checks: leverage, years, costs, what it learned |
| `rule_daily.py` | The rule's daily result after costs, for the explorations |
| `flow_explore.py` | Where the rule earns: time of day, event days, volatility regime |
| `breadth_open.py`, `timing_luck.py`, `breadth_stocks_etfs.py` | Explorations: breadth at the open, a shifted grid, the rule beyond QQQ |
| `gamma_proxies.py`, `gamma_test.py` | Gamma from prices: exploration, then test 2 |
| `hedge_test.py` | Online expert allocation |
| `calendar_effects.py`, `overnight_test.py`, `overnight_validate.py`, `overnight_by_year.py`, `overnight_review.py` | Overnight vs intraday, calendar effects, tests 3 and 4, the independent review |
| `tsmom.py` | Time-series momentum on the ETFs |
| `gex.py`, `gamma_top_third.py`, `gamma_validate.py` | Real dealer gamma: test 5, the exploration, test 6 |
| `b1_b2.py`, `b3.py`, `b4.py`, `b5.py`, `b6.py`, `rsi.py` | Tests 7 and 8 |
| `evidence_power.py` | The shadow record's betting test by simulation: false promotions, time to find an edge, the shares of alpha, the clip (decision 0020) |
| `shadow_demo.py` | The shadow record on real minutes: the rule against a variant deciding every 15 minutes, each in its own account behind `guard_qqq.yaml` |
