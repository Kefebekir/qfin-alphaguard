# ML pilot (decision 0017)

Do price and volume features predict S&P 500 returns out of sample, and does
a long-only portfolio keep anything after costs? This is research code,
outside the package and the live path. The full write-up is in
[docs/research/machine-learning-for-the-plan.md](../../docs/research/machine-learning-for-the-plan.md).

Run it on the stored daily bars (`data/raw/prices.parquet`, with SPY):

```bash
mkdir -p data/ml_pilot
uv run python research/ml_pilot/universes.py data/raw/prices.parquet data/ml_pilot/universes.json
uv run python research/ml_pilot/features.py data/raw/prices.parquet data/ml_pilot/panel.parquet
uv run python research/ml_pilot/walkforward.py data/ml_pilot/panel.parquet data/ml_pilot/pred_h21.parquet 21
uv run python research/ml_pilot/evaluate.py data/ml_pilot/pred_h21.parquet data/ml_pilot/universes.json 21
uv run python research/ml_pilot/yearly.py data/ml_pilot/pred_h21.parquet data/ml_pilot/universes.json 21 gbt
```

Use 1, 5 or 21 sessions as the horizon.

- **Data:** the October 2026 store, with bars to 8 October 2026.
- **Time:** each horizon takes about a minute on 20 cores.
- **Determinism:** the same data gives the same output; the models use fixed
  seeds.

| Script | What it does |
| --- | --- |
| `universes.py` | Each year's 50 most liquid members (decision 0005). |
| `features.py` | 20 price and volume features of every member, ranked across stocks each day; labels from the close of t+1. |
| `walkforward.py` | Ridge and gradient-boosted trees, trained on the years before each test year and purged of the last 1 + h sessions; their average; three single signals. |
| `evaluate.py` | Rank IC, long-only portfolios in both universes before and after 2, 5 and 10 bps, and the deflated Sharpe ratio of the best of the twelve variants. |
| `yearly.py` | The top 10 of the 50 against the 50, year by year. |
