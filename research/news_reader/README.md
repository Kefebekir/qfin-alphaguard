# The news reader: does a language model reading the news help?

After step 5 found that nothing built on prices beats the QQQ rule, and research/frontier
measured why (prices tell how much the market will move, not which way), Efe chose to give
the system a new sense: the news, read by a language model that knows nothing of the future.
Five pre-registered tests (`preregistrations/`, `LEDGER.md`):

- **N1:** does the morning's news tell the QQQ rule's day? The rule sized ×1.5 / ×1 / ×0.5 by
  a forecast from the headlines known by 09:55.
- **N2:** does a member's news tell its next day? Long the tenth of S&P 500 members with news
  that a model forecasts best, short the worst tenth, close to close.
- **N3:** N2 read by PIT-4B, a 4-billion-parameter point-in-time model (Kelly, Malamud,
  Schwab and Xu), each headline by the checkpoint of the year before its own.
- **N4:** N3 read with less noise: the target scaled by volatility, more members a side,
  weights inverse to volatility, statistical sectors neutral; designed after N2's and N3's
  holdouts.
- **N5:** N4's variant 2, frozen, run forward on each session from the headlines as they
  stood at 15:31 (`qfin news-capture`), judged by decision 0020's betting test.

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
| N3: the PIT-4B checkpoints, each pinned by its SHA-256 | `pit_download.py` | `news_reader/pit/<model>-<checkpoint>/`, `manifest.json` |
| N3: the model, rewritten (`pit.py`), checked against a float32 reference and on language | `pit_check.py` | printed |
| N3: each member headline by the checkpoint of the year before, on the GPU | `embed_pit.py` | `news_reader/pit_emb/<model>/<checkpoint>.*` |
| N3's features a member and session, in two sets | `features_n3.py` | `news_reader/pit_emb/n3_rows.parquet`, `n3_<set>.npy` |
| N4: each member-day's volatility and statistical sector | `risk.py` | `news_reader/risk/risk.parquet` |
| N6, before it: do the ten checkpoints see a headline alike? 3,000 headlines read by each | `eyes_check.py` | `news_reader/eyes/`, `results_eyes_check.txt` |
| N6, before it: two ways to bring the older checkpoints to 202412's scale, judged on headlines alone | `align_check.py` | `news_reader/eyes/anchors_*`, `results_align_check.txt` |
| The splits, policies, scores and the holdout lock | `harness.py` | `news_reader/holdout_<test>.json` |
| N5: N4's variant 2 fitted once, frozen as plain arrays | `forward.py` | `models/news-n4-forward.npz` |
| N5: the record's path against the research's, on ten past sessions | `forward_check.py` | `results_forward_check.txt` |
| N5: the record, each session from its 15:31 capture, scored after the next close | `forward_record.py` | `news_reader/forward/days.parquet`, `record.html` |

Details the scripts handle:

- **When an article is known.** Alpaca files articles by their last edit, and 2.5% were edited
  more than 5 minutes after publication; such an article counts from its edit, since the
  text kept may be the edited one.
- **Tickers.** Benzinga writes BRK/B, BRK.B and BRK-B for one stock; a ticker reused by a
  later company is our `<ticker>_old`. A symbol goes to whichever stock was in the S&P 500
  that day, and is left out if two fit or none.
- **N3 runs no code from the models' repositories.** `pit.py` is the published architecture
  written again; it reads only the weights and the config, and refuses a config that differs.
  `test_pit.py` (beside it, as it needs PyTorch) checks it against a plain float64 copy.
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

N5's record, each evening after the nightly job has stored the next session's close (it
copies the captures and the prices from S3 first; PIT-4B on the GPU):

```bash
PYTHONPATH=src data/llmenv/bin/python research/news_reader/forward_record.py
```

Before N2, `architecture_checks.py` measured what news tells in the development years only
(`results_architecture_checks.txt`), for the note on news and the FPGA's network
(`docs/research/news-to-the-fpga-network.md`):

```bash
uv run python research/news_reader/architecture_checks.py
```
