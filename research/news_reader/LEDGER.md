# Ledger: the news reader (Phase 2, after step 5)

Every test of the news reader and every model variant tried, in order. A pre-registered test
was written down, with its hypothesis, data, splits and threshold, and approved by Efe before
any of its results was computed; its file in `preregistrations/` is unchanged since (SHA-256
below, of the file as committed). Development results may be looked at freely; each variant
tried on them is listed here before its holdout run.

## Pre-registered tests

| # | Written (UTC) | File | SHA-256 | Hypothesis | Pass if | Result |
|---|---|---|---|---|---|---|
| N1 | 2026-10-16 13:43:43 | `N1_morning_news_qqq.txt` | `857208114180ee6af8dd2dadfd784e925f9595bc8c481dbc783e5448720ef481` | The morning's news (16:00 to 09:55), read by ChronoBERT of the year before, tells the QQQ rule's day; the rule sized ×1.5 / ×1 / ×0.5 by the forecast's third beats the rule at equal risk | one-sided t ≥ 2.24 on 2022 – 9 October 2026 | **Failed.** Variant 5 (2 components): −1.52 bps a day, t −1.75, over 1,197 days; −3.24 / −4.69 / +1.80 / −1.68 / +0.71 in 2022-2026. The count alone: −3.35, t −3.29 |
| N2 | 2026-10-16 13:43:43 | `N2_member_news_next_day.txt` | `ea92212a00d072e7bc294aeb4e6cb607aee3fedfab4a6c3c86fe8e2ac698190a` | A member's news to 15:30 tells its next close-to-close return against the other members with news; long the top tenth, short the bottom tenth | one-sided t ≥ 2.50 after 1 bp a round trip a side, 2022 – 9 October 2026 (amendments 2 and 3; it was t ≥ 2.24 after 10 bps) | pending |
| N3 | 2026-10-18 17:07 | `N3_member_news_pit4b.txt` | `1e3cd2b547670b44218ab1eec68ddb9a69eb0015ab974d05635883f11e79264c` | N2 read by PIT-4B, a 4-billion-parameter point-in-time model, base or fine-tuned, its last token's state or its mean | one-sided t ≥ 2.50 after 1 bp a round trip a side, 2022 – 9 October 2026 | pending |

## Amendments

| # | Written (UTC) | File | SHA-256 | Change |
|---|---|---|---|---|
| 1 | 2026-10-16 13:44:34 | `N_amendment_1.txt` | `11cb20b829668176c245947081e80c967cc8cee97e5c5bce5722e8076d25f196` | No ChronoBERT vintage for 2025 is published: test year 2026 uses vintage 2024, which knows less |
| 2 | 2026-10-18 17:07 | `N_amendment_2.txt` | `5f2e4fa602cf251208e609c8d6c0c851fd6c78d9018c539c3e8d63b74cbd3a16` | N2's cost, from a placeholder of 10 bps a round trip a side to 1 bp: an order on the close fills at the official close, and Alpaca's paper account took a whole-share one on 18 October 2026. Only the count-only development check had been computed |
| 3 | 2026-10-18 17:07 | `N_amendment_3.txt` | `fc469e8f37780caafb8866dd7d233935b2cdc403d61e86d13af563d6048acd08` | N3 joins: N2 and N3 share the half of the news tests' 2.5% that N1 did not use, so N2 passes at t ≥ 2.50, not 2.24 |

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

At most 10 for each test. Each row is written before the next variant is run.

**N1's choice for the holdout, fixed 2026-10-17 12:43:56 UTC before variants 4-6 were run:** the variant
with the highest mean a day on the development years goes to the holdout, whatever it is.
Variants 4-6 are Efe's choice of 1, 2 and 8 components (penalty 10.0), the last of N1.
By that rule, variant 5 (2 components, +0.98 bps a day) goes to the holdout; its gain is
2018's, and 2020 and 2021 were down, as with the count alone.

| # | Test | Run (UTC) | Settings (`efe_model.make`) | Mean a day | t | 2018 / 2019 / 2020 / 2021, bps a day | Note |
|---|---|---|---|---|---|---|---|
| 1 | N1 | 2026-10-17 | components 16, penalty 10.0 | −0.18 bps | −0.22 | +1.74 / −1.13 / +0.09 / −1.42 | No better than the count alone (−0.16); signs mixed across years |
| 2 | N1 | 2026-10-17 | components 4, penalty 1000.0 | −0.25 bps | −0.31 | +3.45 / +1.57 / −3.60 / −2.38 | The count's pattern, stronger: up in 2018-2019, down in 2020-2021. Before it, on training data only: the penalty barely changes which days are sized (rank correlation 0.97-0.998 from 100 to 100,000), the number of components does (0.48 for 4) |
| 3 | N1 | 2026-10-17 | components 64, penalty 1000.0 | −1.11 bps | −1.20 | −2.11 / +0.49 / −0.87 / −1.98 | The worst: more detail, more noise learned; down in three of four years |
| 4 | N1 | 2026-10-17 | components 1, penalty 10.0 | −0.63 bps | −0.60 | +2.19 / +0.14 / −3.31 / −1.52 | The dominant topic axis alone: the count's pattern |
| 5 | N1 | 2026-10-17 | components 2, penalty 10.0 | +0.98 bps | +0.88 | +7.65 / +1.23 / −3.23 / −1.69 | The count's pattern again: up in 2018-2019, down in 2020-2021 |
| 6 | N1 | 2026-10-17 | components 8, penalty 10.0 | −2.13 bps | −2.12 | −4.01 / +0.68 / −2.89 / −2.33 | Down in three of four years |

## N1's holdout (run once, 17 October 2026, commit 0893980)

`data/news_reader/holdout_N1.json`; a second run is refused.

| | Days | Mean a day | t | 2022 / 2023 / 2024 / 2025 / 2026, bps a day |
|---|---|---|---|---|
| Variant 5: 2 components, penalty 10 | 1,197 | −1.52 bps | −1.75 | −3.24 / −4.69 / +1.80 / −1.68 / +0.71 |
| The count alone (the comparison) | 1,197 | −3.35 bps | −3.29 | −3.47 / −6.13 / −4.51 / −0.12 / −2.26 |

**N1 failed: the morning's headlines, read by ChronoBERT of the year before and a linear
model, do not tell the QQQ rule's day.** No variant cleared the count alone on development;
the chosen one lost to the rule on the holdout, in three of its five years. Sizing the rule
by how busy the morning was hurt it in 2022-2026 (t −3.29): a relation fitted on earlier
years that ran the other way after 2021, as the price-based gamma proxies did (step 5,
test 2). That is an observation, not a test.
