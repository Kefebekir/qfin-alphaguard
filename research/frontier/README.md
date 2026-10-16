# What the ML is missing, and where to look next (16 October 2026)

Efe asked why the ML still does not work: would more data, synthetic data or far more
parameters help, how do people earn from patterns and earnings news, and how can the system
become something that learns and improves. These are explorations, not pre-registered tests:
they measure what our data can and cannot tell, to choose the next tests.

| Script | Question | Result |
|---|---|---|
| `diagnosis.py` → `results/01_diagnosis.txt` | On 16 ETFs' half-hourly decisions, 2018-2026 out of sample: can the next half hour's direction be predicted, and its size? What do more parameters and more data do? | Direction: R² at or below 0, rank IC 0.00-0.01, the sign right 50% of the time. Size (the absolute return): R² +15-18%, IC 0.34-0.39. Trees from 2 to 1,024 leaves: training R² 0.7% → 99.98%, test R² −0.05% → −20%. Sixteen times the data (all ETFs pooled): test R² −0.9% → +0.02%. An IC of 0.02 would show as t 3.6 on QQQ's 32,190 half hours, so a stable signal of that size is not in these inputs. |
| `news_reaction.py` → `results/02_news_reaction.txt` | After a Benzinga headline (Alpaca's free news API) on one of our 98 large stocks in market hours, does the price's first move carry on? | It does, and moves without news reverse: the slope of what follows on the first minute's move is +0.065 (t +4.7) after news, −0.082 (t −5.4) without. But what is left after the first minute is small: 2-3 bps on a first move of 38-63 bps, about the cost of a round trip. |
| `edgar_earnings.py`, `earnings_drift.py` → `results/03_earnings_drift.txt` | "Earnings beat, the stock will rise": after the biggest reactions to an earnings release, does the stock keep moving? | No. 20,072 releases of 590 members, 2015-2026: the best fifth of reactions earn +7.3% on the reaction day; holding them 60 sessions against the worst fifth earns +0.8% a year (t 0.46), and against SPY −0.8%. Martineau (2022) finds the same for large stocks since 2006. |

## Data, all in the git-ignored `data/frontier/`

- **Benzinga articles** from Alpaca's news API, free with the account (`news/<ticker>.jsonl`):
  headline, symbols, `created_at` and `updated_at`. Some headlines are edited after
  publication, so a test that reads the text must use the version known at the time.
- **Earnings releases** from SEC EDGAR (`edgar/earnings.parquet`): every 8-K with Item 2.02
  since 2015 for the 597 members EDGAR's ticker list knows. **The submissions API's
  `acceptanceDateTime` agreed with the filing's own header on only 50.1% of filings**; the
  rest are shifted by twice New York's offset from UTC (Apple's 16:30:33 release of 29 January
  2026 is listed as 02:30:33Z the next day). The times here come from each filing's header.

## Run

```bash
uv run python research/frontier/diagnosis.py data/intraday_pilot/policy/decisions.parquet
uv run --env-file .env python research/frontier/news_reaction.py data/raw data/frontier
uv run --env-file .env python research/frontier/edgar_earnings.py data/raw/prices.parquet data/frontier/edgar
uv run python research/frontier/earnings_drift.py data/raw/prices.parquet data/raw/etf_daily/SPY.parquet data/frontier/edgar/earnings.parquet
```
