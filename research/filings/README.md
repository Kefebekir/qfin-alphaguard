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
| Each report's risk factors and MD&A, and how much they changed | `sections.py` | `sections.parquet`, `results_sections.txt` |
| The last look, on those two sections, its rule fixed before its first run | `section_look.py` | `results_section_look.txt` |

```bash
set -a && source <(grep -E "^SEC_USER_AGENT=" .env) && set +a
uv run python research/filings/members.py
uv run python research/filings/download.py
uv run python research/filings/change.py
uv run python research/filings/first_look.py
uv run python research/filings/sections.py
uv run python research/filings/section_look.py
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

## The last look

Efe chose one more look, on 13 November 2026: the same rule as the first look's, on each
report's risk factors and MD&A alone (`section_look.py`, its rule in its docstring). With a t of
2.0 or more on 2016-2021, a test on 2022-2026 is written down for his approval and run once.
Below that, B is closed and 2022-2026 stay unread.

`sections.py` finds the two sections in each report's text, from its heading to the next
item's, skipping the table of contents and references to them (`results_sections.txt`). A 10-Q
often has no risk factors of its own, or only says nothing has changed since the 10-K, so its
pooled text is mostly its MD&A.

**The result** (`results_section_look.txt`). Over 2016-2021 the two sections made **+36.6 bps
a month after costs, t +1.86** (annual Sharpe ratio +0.76, 71 months). That is below the gate
of 2.0, so by the rule B is closed and 2022-2026 stay unread.
- It is stronger than the first look's +14.2 (t +0.61), and the short side now adds too
  (+17.8 a month, t +1.68), as the study says it should.
- Beside it, not to be chosen among: Jaccard +37.6 (t +1.86); the risk factors alone +50.9
  (t +1.74); the MD&A alone +10.4 (t +0.50); 10-Qs only +64.9 (t +2.76); 10-Ks only -13.5;
  held six months +49.7 (t +2.73). These were seen after the result, and choosing one of
  them now would be choosing by the result.
