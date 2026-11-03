# Filings: is what companies write slow to reach their prices?

The news is priced on its own day (`research/news_reader/event_study.py`). A long report may
not be: the study of "lazy prices" (Cohen, Malloy and Nguyen, Journal of Finance, 2020) found
that a company whose 10-K or 10-Q changes much from last year's goes on to do worse over the
following months. This asks whether that holds among S&P 500 members.

**No return after 2021 is read until a test is written down.** The news tests had seen every
year before their last test. These filings have not been read at all, so 2022-2026 can still
be a clean holdout, run once, as in `research/news_reader/LEDGER.md`.

| Step | Script | Output (git-ignored `data/filings/`) |
|---|---|---|
| Which company filed for each member, over its own years | `members.py` | `members.parquet` |
| Every member's 10-K and 10-Q as plain text | `download.py` | `filings.parquet`, `text/<cik>/<accession>.txt.gz` |
| How much each report changed from a year before | `change.py` | `change.parquet` |
| A first look, on 2016-2021 only, its rule fixed before its first run | `first_look.py` | `results_first_look.txt` |

```bash
set -a && source <(grep -E "^SEC_USER_AGENT=" .env) && set +a
uv run python research/filings/members.py
uv run python research/filings/download.py
uv run python research/filings/change.py
uv run python research/filings/first_look.py
```

SEC asks for a contact in `SEC_USER_AGENT`, and at most 10 requests a second.

## The first look

24,721 reports were downloaded with none failed, and 21,828 have a report a year before to
compare with. Over 2016-2021, long the fifth that changed least and short the fifth that changed
most made **+14.2 bps a month after costs, t +0.61** (annual Sharpe ratio +0.25, 71 months).

The first three years carry it (+71.5, +54.0, +36.4), then +4.8, -85.3 and +2.6. Almost all of it
is the long side (+12.7, t +1.02); the short side, which is the study's claim, makes +1.9 (t
+0.14). Nothing beside it reaches a t of 2: the best is Jaccard, +27.3 (t +1.24), and those
variations are not to be chosen among. January 2016 ranks only 29 names, since reports filed in
2015 have no score; leaving out the first three months gives +11.6 (t +0.48).

The look reads each whole report, and its raw word counts are mostly common words, so a median
cosine is 0.999 for a 10-K. The study's strongest results are in single sections, which this does
not read.

No test on 2022-2026 has been written, and those years remain unread. `results_first_look.txt`
holds the full output.
