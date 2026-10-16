# The news reader: does a language model reading the news help?

After step 5 found that nothing built on prices beats the QQQ rule, and research/frontier
measured why (prices tell how much the market will move, not which way), Efe chose to give
the system a new sense: the news, read by a language model that knows nothing of the future.
Two pre-registered tests (`preregistrations/`, `LEDGER.md`):

- **N1:** does the morning's news tell the QQQ rule's day? The rule sized ×1.5 / ×1 / ×0.5 by
  a forecast from the headlines known by 09:55.
- **N2:** does a member's news tell its next day? Long the tenth of S&P 500 members with news
  that a model forecasts best, short the worst tenth, close to close.

The language model is **ChronoBERT** (He, Lv, Manela and Wu, "Chronologically Consistent
Large Language Models"): a ModernBERT encoder retrained every year on text from before that
year's end only (`manelalab/chrono-bert-v1-<year>1231`, MIT licence). Test year Y reads the
headlines with the vintage of Y - 1, so the model cannot know what happened in the year it is
tested on (vintage 2024 for 2026: amendment 1). Efe trains the model on top of it
(`GUIDE.md`).

## The pipeline

| Step | Script | Output (git-ignored `data/`) |
|---|---|---|
| Every Benzinga article on Alpaca's free news API, 2016 – October 2026 | `news_all.py` | `frontier/news_all/<month>.jsonl` |
| The QQQ rule's every session in the engine, and QQQ's 14-day volatility | `rule_days.py` | `news_reader/rule_days.parquet` |
| Each headline's embedding by every vintage a test year needs, on the GPU | `embed.py` | `news_reader/emb/index.parquet`, `v<vintage>.npy` |
| N1's features a session, N2's a member and session | `features.py` | `news_reader/emb/n1_v<vintage>.*`, `n2_v<vintage>.*` |
| The splits, policies, scores and the holdout lock | `harness.py` | `news_reader/holdout_<test>.json` |

Details the scripts handle:

- **When an article is known.** Alpaca files articles by their last edit, and 2.5% were edited
  more than 5 minutes after publication; such an article counts from its edit, since the
  text kept may be the edited one.
- **Tickers.** Benzinga writes BRK/B, BRK.B and BRK-B for one stock; a ticker reused by a
  later company is our `<ticker>_old`. A symbol goes to whichever stock was in the S&P 500
  that day, and is left out if two fit or none.
- **The holdout runs once.** `harness.run_holdout` needs the chosen variant committed in
  `variants/<test>_chosen.json` and writes a lock with the result and the commit.

## Environments

`news_all.py`, `rule_days.py`, `features.py` and `harness.py` run in the project's environment
(`uv run python ...`). `embed.py` needs PyTorch with CUDA and transformers, kept apart under
the git-ignored `data/`:

```bash
uv venv data/llmenv --python 3.12
uv pip install --python data/llmenv/bin/python torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python data/llmenv/bin/python "transformers>=4.48" numpy polars safetensors huggingface_hub
```

Efe's GPU, an RTX 5070 Laptop (8 GB), embeds about 1,500 headlines a second, headlines
sorted by length; `embed.py` runs beside the download and starts each vintage once its
months are on disk.

## The data, as built on 16 October 2026

- 2,093,052 articles, 1 January 2016 to 15 October 2026, 130 months, no failed request.
- N1: 2,711 sessions from 5 January 2016, 342-497 headlines a morning on average (rising
  over the years). Checked by hand on 5 January 2016 (430) and 11 March 2024, the Monday
  after the switch to summer time (540), with the stored mean embedding.
- N2: 260,186 member-days with news (vintage 2024, every year), from 503,998 member
  articles. Checked by hand on Apple's 3 May 2024, the day after its results (36).

## Run

```bash
uv run --env-file .env python research/news_reader/news_all.py data/frontier/news_all
uv run python research/news_reader/rule_days.py data/raw guard_qqq.yaml data/raw/etf_daily/QQQ.parquet data/news_reader/rule_days.parquet
data/llmenv/bin/python research/news_reader/embed.py data/frontier/news_all data/news_reader/emb
uv run python research/news_reader/features.py data/news_reader/emb data/raw/prices.parquet
```

Then `GUIDE.md`.
