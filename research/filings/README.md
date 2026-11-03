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

```bash
set -a && source <(grep -E "^SEC_USER_AGENT=" .env) && set +a
uv run python research/filings/members.py
```

SEC asks for a contact in `SEC_USER_AGENT`, and at most 10 requests a second.
