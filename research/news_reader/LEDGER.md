# Ledger: the news reader (Phase 2, after step 5)

Every test of the news reader and every model variant tried, in order. A pre-registered test
was written down, with its hypothesis, data, splits and threshold, and approved by Efe before
any of its results was computed; its file in `preregistrations/` is unchanged since (SHA-256
below, of the file as committed). Development results may be looked at freely; each variant
tried on them is listed here before its holdout run.

## Pre-registered tests

| # | Written (UTC) | File | SHA-256 | Hypothesis | Pass if | Result |
|---|---|---|---|---|---|---|
| N1 | 2026-10-16 13:43:43 | `N1_morning_news_qqq.txt` | `857208114180ee6af8dd2dadfd784e925f9595bc8c481dbc783e5448720ef481` | The morning's news (16:00 to 09:55), read by ChronoBERT of the year before, tells the QQQ rule's day; the rule sized ×1.5 / ×1 / ×0.5 by the forecast's third beats the rule at equal risk | one-sided t ≥ 2.24 on 2022 – 9 October 2026 | pending |
| N2 | 2026-10-16 13:43:43 | `N2_member_news_next_day.txt` | `ea92212a00d072e7bc294aeb4e6cb607aee3fedfab4a6c3c86fe8e2ac698190a` | A member's news to 15:30 tells its next close-to-close return against the other members with news; long the top tenth, short the bottom tenth | one-sided t ≥ 2.24 after 10 bps a round trip a side, 2022 – 9 October 2026 | pending |

## Amendments

| # | Written (UTC) | File | SHA-256 | Change |
|---|---|---|---|---|
| 1 | 2026-10-16 13:44:34 | `N_amendment_1.txt` | `11cb20b829668176c245947081e80c967cc8cee97e5c5bce5722e8076d25f196` | No ChronoBERT vintage for 2025 is published: test year 2026 uses vintage 2024, which knows less |

## The pre-registered comparison on the development years

`harness.run(test, None, count_only=True)`: a linear regression on log(1 + the number of
headlines) alone, run on 16 October 2026 to check the pipeline end to end. It is the bar a
variant should clear; it is not one of the variants.

| Test | Days | Mean a day | t | 2018 / 2019 / 2020 / 2021, bps a day |
|---|---|---|---|---|
| N1 | 1,008 | −0.16 bps | −0.15 | +1.76 / +1.79 / −2.93 / −1.24 |
| N2 | 998 | −25.06 bps | −6.26 | −30.43 / −20.65 / −28.73 / −20.53 |

N2's costs are 20 bps a day (its positions turn over daily), so a variant must earn more
than that before costs.

## Variants tried on the development years (2018-2021)

At most 10 for each test. None yet.
