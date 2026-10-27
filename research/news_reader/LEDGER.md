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
| N2 | 2026-10-16 13:43:43 | `N2_member_news_next_day.txt` | `ea92212a00d072e7bc294aeb4e6cb607aee3fedfab4a6c3c86fe8e2ac698190a` | A member's news to 15:30 tells its next close-to-close return against the other members with news; long the top tenth, short the bottom tenth | one-sided t ≥ 2.50 after 1 bp a round trip a side, 2022 – 9 October 2026 (amendments 2 and 3; it was t ≥ 2.24 after 10 bps) | **Failed.** Variant 2 (64 components): −2.22 bps a day after costs, t −0.78, over 1,197 days; −4.19 / +5.03 / −5.87 / −1.46 / −5.26 in 2022-2026. The count alone: −3.10, t −1.19 |
| N3 | 2026-10-18 17:07 | `N3_member_news_pit4b.txt` | `1e3cd2b547670b44218ab1eec68ddb9a69eb0015ab974d05635883f11e79264c` | N2 read by PIT-4B, a 4-billion-parameter point-in-time model, its last token's state or its mean (amendment 4: the base model only) | one-sided t ≥ 2.50 after 1 bp a round trip a side, 2022 – 9 October 2026 | **Failed.** Variant 1 (64 components, base-last): +4.24 bps a day after costs, t +1.18, over 1,197 days; +0.84 / +7.67 / +5.33 / −11.34 / +22.88 in 2022-2026. The count alone: −3.10, t −1.19 |
| N4 | 2026-10-21 13:05 | `N4_member_news_less_noise.txt` | `e9027c6b00801b2ddcede19fc3fc5618076234eae02d19bcdb6c4f405d788bd5` | N3 read with less noise: the target scaled by volatility, a fifth or a third per side, weights inverse to volatility, statistical sectors neutral; tuned on 2018-2021. Designed after N2's and N3's holdouts were seen, so its run on those years is a second look, and a pass leads only to the shadow record | one-sided t ≥ 2.50 after 1 bp a round trip a side, 2022 – 9 October 2026 (its own 0.625%) | **Failed.** Variant 2 (64 components, base-last, target scaled): +6.53 bps a day after costs, t +1.86, over 1,197 days; +1.32 / +5.78 / +5.97 / −9.42 / +35.55 in 2022-2026; without 2026, +0.92. Explored beside it (amendment 6): variants 6, 10 and 8, +3.09, +1.03 and +0.97 |
| N5 | 2026-10-21 14:23 | `N5_n4_forward_record.txt` | `2e9395f7d7c6f826d5234e900084abd554c1c557ec2d52aeceff22e32ce68c60` | N4's variant 2, frozen (fitted once on 2016 – 9 October 2026), run forward each session in the shadow record: headlines captured at 15:31, read by PIT-4B-202412, scored after the next close | the betting test against no trade, clip 0.085, alpha 1.25% (wealth 80); reviewed 31 December 2028 | pending: the record has not started |

## Amendments

| # | Written (UTC) | File | SHA-256 | Change |
|---|---|---|---|---|
| 1 | 2026-10-16 13:44:34 | `N_amendment_1.txt` | `11cb20b829668176c245947081e80c967cc8cee97e5c5bce5722e8076d25f196` | No ChronoBERT vintage for 2025 is published: test year 2026 uses vintage 2024, which knows less |
| 2 | 2026-10-18 17:07 | `N_amendment_2.txt` | `5f2e4fa602cf251208e609c8d6c0c851fd6c78d9018c539c3e8d63b74cbd3a16` | N2's cost, from a placeholder of 10 bps a round trip a side to 1 bp: an order on the close fills at the official close, and Alpaca's paper account took a whole-share one on 18 October 2026. Only the count-only development check had been computed |
| 3 | 2026-10-18 17:07 | `N_amendment_3.txt` | `fc469e8f37780caafb8866dd7d233935b2cdc403d61e86d13af563d6048acd08` | N3 joins: N2 and N3 share the half of the news tests' 2.5% that N1 did not use, so N2 passes at t ≥ 2.50, not 2.24 |
| 4 | 2026-10-18 22:40 | `N_amendment_4.txt` | `10402a55ace2155dd4bda24ba0bd0e26a5b2b03a5d24949877dde4620e853d86` | N3 drops PIT-4B-FT before any headline was read with it: its fine-tuning data (Tulu 3, and instructions Qwen2.5-72B wrote) dates from 2024, filtered of time-specific rows by a classifier, so it is not point-in-time. Two feature sets remain, base-last and base-mean |
| 5 | 2026-10-19 10:22 | `N_amendment_5.txt` | `1d2dd258bc192f7be47dbfbb4e9f87fecc8a6d7b08bd0effee1f3aeb1e5e4ffb` | N2 and N3: members tied at a tenth's edge are all in (the breakpoint convention), not chosen by the order of rows, which was alphabetical: counting alone's −7.06 bps a day was the alphabet's. Efe chose this rule among four scored on counting alone; it is not changed again |
| 6 | 2026-10-21 13:39 | `N_amendment_6.txt` | `1a9495bc571202c596bc6d61629f496f114cbaa7d7dcaa7c0f494d95ba38d7b6` | N4: the test unchanged (variant 2 once, t ≥ 2.50); by Efe's decision variants 6, 10 and 8 also run once on the same years, as exploration, not tests, beside the bar four tries sharing N4's 0.625% would need, t ≥ 2.96 |

## N3's checkpoints

Pinned on 2026-10-18 at 17:42 UTC, before any N3 result was computed: each checkpoint's
model.safetensors by the SHA-256 Hugging Face records for it at the commit shown, the same
file under Diamegs and anonymneurips2027 for all twenty (17,754,502,408 bytes each).
`pit_download.py` fetches each at that commit and checks the SHA-256; `embed_pit.py` checks
it again, and the config against `pit.py`'s copy of the architecture, before reading a
headline with it. PIT-4B-201511's embeddings had started a few minutes before these rows were
written; its file was the one pinned here.

Amendment 4 dropped PIT-4B-FT: its ten rows are not used.

| Model | Checkpoint | Reads the headlines of | Commit | SHA-256 of model.safetensors |
|---|---|---|---|---|
| PIT-4B | 201511 | 2016 | `93c8f9a581c4` | `5a4f8a684ce2475835c1ce20634fccf7bdfe96fd9e3cd1de79efeb3b96c14deb` |
| PIT-4B | 201612 | 2017 | `c15a07ff8754` | `0c334768a1d8f4a9de4aae4352d1a862e63a78c6b15e1a5be8c8506f84aa0aef` |
| PIT-4B | 201712 | 2018 | `2d7488301496` | `a078bf244d5c7c4d28b8051481514ed2b3b9127916e1c85c9539e07c62ea85ae` |
| PIT-4B | 201812 | 2019 | `9bba5a6507f6` | `7800b00e863397206ffb326a81d69d75f19ffbff58e34f6940fc28aaaaa4fd62` |
| PIT-4B | 201912 | 2020 | `2fa4c167939f` | `45e100b618bafeac7ebebad90bd78d91058054c3d433ff9db41e2fbf33ab8c4a` |
| PIT-4B | 202012 | 2021 | `7d733359a9f9` | `332d7d186a4eec0c9a0a2f5fd00aaa6670fe306f07e4fd916ebde4d576068067` |
| PIT-4B | 202112 | 2022 | `4d3ed330c349` | `1c3614973394a234fac3ab146d96a2419a162116753ce37095ca75a35a816408` |
| PIT-4B | 202212 | 2023 | `360b0fcffa36` | `932b14a41afcb723368785cf6189adcb37ee13938e6f75ed4c8acc2c5146ae34` |
| PIT-4B | 202312 | 2024 | `572c0dd6c26d` | `dcb0baccbd5329e6d8c5616cc125860fbc843ef14066c48e11bb605ce71318dd` |
| PIT-4B | 202412 | 2025, 2026 | `71fe135d8e93` | `38072fc6b67258ff7e58edd951508106b5b89c1daf7700c80ebd030c03c7451e` |
| PIT-4B-FT | 201511 | 2016 | `32bd64b647eb` | `4dcc064e968bcc325fb748c32e37147c2e990291fd996bcc87232da131f8be06` |
| PIT-4B-FT | 201612 | 2017 | `2c65885f4a85` | `5546907b4ecc523a39d6bb71593d8cdf855834971f6eacf81a21463ae0985521` |
| PIT-4B-FT | 201712 | 2018 | `b03125e27592` | `ef401d069c80479b324e97d148681cd71ccdfb6fceef3c8b9bd68bbd6915e2f8` |
| PIT-4B-FT | 201812 | 2019 | `a93820f680b7` | `2b89f956ea1c8becbbf862a5a8ffb46b26bdd5fe8211a9465e0b185204b2273b` |
| PIT-4B-FT | 201912 | 2020 | `dd482b6a647f` | `d971e83fc3b22af7280809ecc7321f57c14d0e2ab65e8903424bd997cde83f9d` |
| PIT-4B-FT | 202012 | 2021 | `faa015fb0354` | `8d1bce70dfb4650c3422817f961214b8805de10b5a6713a8ecba006177c2a13a` |
| PIT-4B-FT | 202112 | 2022 | `efe304f98759` | `7e36705b27e1cc68ce1e7ed511948c3fe922e16e4d7179811e80cebaf4d6cf64` |
| PIT-4B-FT | 202212 | 2023 | `fc4eaefde502` | `9b946256c09c129e1c42a211c5ab9048ed0e5652df5a41de257d7560d996dac2` |
| PIT-4B-FT | 202312 | 2024 | `bf786611e285` | `4c7cf18f8ff29038650fac16e0bfe1cf134e670e53ea815e8e7de6c301c93856` |
| PIT-4B-FT | 202412 | 2025, 2026 | `c82f0c11e583` | `673ed853950105f974110f41cf24e02b06fef62a68bd14d9aaf90bdccaf44dfa` |

How the embeddings are computed, fixed here before any N3 result (the pre-registration says
"in bfloat16 by the project's own copy"):

- The precision is the models' training: float32 weights under bfloat16 autocast, so the
  products, the per-head norms and attention in bfloat16, the residual stream and its norms
  in float32. On 12 headlines of 2016 read by PIT-4B-201511, the copy's states have cosine
  ≥ 0.99998 with a float32 reference written apart from it (`pit_check.py`); a model run
  wholly in bfloat16, as the repository's example loads it, ≥ 0.99996.
- Every norm's epsilon is the config's, 2^-23.
- A headline is its tokens alone, at most 64, with no token before them, as the model card
  reads text; the mean is over those tokens.

## The pre-registered comparison on the development years

`harness.run(test, None, count_only=True)`: a linear regression on log(1 + the number of
headlines) alone, run on 16 October 2026 to check the pipeline end to end. It is the bar a
variant should clear; it is not one of the variants.

| Test | Days | Mean a day | t | 2018 / 2019 / 2020 / 2021, bps a day |
|---|---|---|---|---|
| N1 | 1,008 | −0.16 bps | −0.15 | +1.76 / +1.79 / −2.93 / −1.24 |
| N2 | 998 | −25.06 bps | −6.26 | −30.43 / −20.65 / −28.73 / −20.53 |
| N2 and N3, at 1 bp a round trip a side (amendment 2), run 18 October 2026 | 998 | −7.06 bps | −1.76 | −12.43 / −2.65 / −10.73 / −2.53 |
| N2 and N3, at 1 bp, members tied at a tenth's edge all in (amendment 5), run 19 October 2026 | 998 | −1.22 bps | −0.41 | −6.33 / −4.16 / +2.70 / +2.79 |

N2's first row is at the placeholder's 20 bps a day (10 bps a round trip on each side; its
positions turn over daily). At amendment 2's 1 bp the costs are 2 bps a day, and the count
alone loses 5.06 bps a day before them in both rows. N3 reads N2's member-days
(`features_n3.py` reproduces N2's 260,186 rows and their counts exactly), so its comparison
is N2's, to the last digit.

N2's two rows before the last chose among members tied at a tenth's edge by the order of
rows, which was alphabetical (amendment 5). Counting alone gives one forecast to every member
with a single headline, about two thirds of them, so its long side was the end of the
alphabet. The last row is the bar a variant should clear.

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
| 1 | N2 | 2026-10-19 10:25 | components 16, penalty 10.0 | −6.96 bps | −1.65 | −4.79 / −8.42 / −7.78 / −6.82 | −5.74 bps a day below counting alone (−1.22); scored again, ties at the edge included (amendment 5): it was −6.78 bps (t −1.61) |
| 2 | N2 | 2026-10-19 10:26 | components 64, penalty 10.0 | +4.03 bps | +0.93 | +6.02 / +12.69 / −2.60 / +0.09 | +5.25 bps a day above counting alone (−1.22) |
| 3 | N2 | 2026-10-19 10:33 | components 4, penalty 10.0 | −4.81 bps | −1.13 | −6.28 / +1.13 / −10.63 / −3.49 | −3.59 bps a day below counting alone (−1.22) |
| 1 | N3 | 2026-10-19 10:35 | components 64, penalty 10.0, features base-last | +6.34 bps | +1.43 | +6.96 / +12.47 / +4.63 / +1.36 | +7.56 bps a day above counting alone (−1.22) |
| 2 | N3 | 2026-10-19 10:39 | components 64, penalty 10.0, features base-mean | +3.89 bps | +0.83 | +1.37 / −1.46 / +10.76 / +4.84 | +5.11 bps a day above counting alone (−1.22) |
| 3 | N3 | 2026-10-19 10:40 | components 128, penalty 10.0, features base-last | +3.48 bps | +0.78 | −2.49 / −1.00 / +13.60 / +3.72 | +4.70 bps a day above counting alone (−1.22) |
| 4 | N3 | 2026-10-19 10:43 | components 32, penalty 10.0, features base-last | +5.49 bps | +1.28 | −0.96 / +12.47 / +12.38 / −1.97 | +6.71 bps a day above counting alone (−1.22) |
| 5 | N3 | 2026-10-19 10:43 | components 64, penalty 1000.0, features base-last | +6.23 bps | +1.41 | +6.38 / +11.95 / +5.55 / +1.07 | +7.45 bps a day above counting alone (−1.22) |
| 6 | N3 | 2026-10-19 10:47 | components 128, penalty 1000.0, features base-last | +3.23 bps | +0.73 | −2.48 / −1.14 / +12.49 / +3.94 | +4.45 bps a day above counting alone (−1.22) |
| 7 | N3 | 2026-10-19 10:48 | components 32, penalty 1000.0, features base-last | +5.84 bps | +1.36 | −0.32 / +12.57 / +12.75 / −1.67 | +7.06 bps a day above counting alone (−1.22) |
| 1 | N4 | 2026-10-21 13:21 | components 64, penalty 10.0, features base-last, target scaled, side fifth, weights equal, sectors none | +3.37 bps | +1.19 | +5.33 / +13.89 / −2.39 / −3.27 | +4.59 bps a day above counting alone (−1.22) |
| 2 | N4 | 2026-10-21 13:24 | components 64, penalty 10.0, features base-last, target scaled, side tenth, weights equal, sectors none | +8.74 bps | +2.07 | +3.97 / +22.52 / −0.18 / +8.58 | +9.96 bps a day above counting alone (−1.22) |
| 3 | N4 | 2026-10-21 13:25 | components 64, penalty 10.0, features base-last, target plain, side fifth, weights equal, sectors none | +2.09 bps | +0.68 | +3.12 / +1.73 / +2.24 / +1.28 | +3.31 bps a day above counting alone (−1.22) |
| 4 | N4 | 2026-10-21 13:25 | components 64, penalty 10.0, features base-last, target plain, side tenth, weights inverse, sectors none | +7.43 bps | +1.79 | +6.98 / +12.78 / +10.09 / −0.09 | +8.65 bps a day above counting alone (−1.22) |
| 5 | N4 | 2026-10-21 13:26 | components 64, penalty 10.0, features base-last, target plain, side tenth, weights equal, sectors neutral | +5.40 bps | +0.73 | +7.58 / −6.07 / +23.08 / −3.37 | +6.62 bps a day above counting alone (−1.22) |
| 6 | N4 | 2026-10-21 13:29 | components 64, penalty 10.0, features base-last, target scaled, side tenth, weights inverse, sectors none | +8.49 bps | +2.18 | +5.17 / +21.99 / +2.53 / +4.25 | +9.71 bps a day above counting alone (−1.22) |
| 7 | N4 | 2026-10-21 13:31 | components 32, penalty 10.0, features base-last, target scaled, side tenth, weights inverse, sectors none | +4.90 bps | +1.23 | −6.70 / +14.78 / +12.39 / −0.99 | +6.13 bps a day above counting alone (−1.22) |
| 8 | N4 | 2026-10-21 13:32 | components 128, penalty 10.0, features base-last, target scaled, side tenth, weights inverse, sectors none | +7.79 bps | +1.89 | +2.55 / +13.05 / +16.19 / −0.66 | +9.01 bps a day above counting alone (−1.22) |
| 9 | N4 | 2026-10-21 13:35 | components 64, penalty 10.0, features base-mean, target scaled, side tenth, weights inverse, sectors none | +0.32 bps | +0.08 | +1.63 / +7.35 / +1.40 / −9.02 | +1.54 bps a day above counting alone (−1.22) |
| 10 | N4 | 2026-10-21 13:36 | components 96, penalty 10.0, features base-last, target scaled, side tenth, weights equal, sectors none | +8.49 bps | +1.89 | −0.68 / +16.65 / +12.58 / +5.31 | +9.72 bps a day above counting alone (−1.22) |

**N2's choice for the holdout, fixed 2026-10-19 10:46 UTC by Efe after three variants:**
variant 2 (components 64, penalty 10.0; `variants/N2_chosen.json`), the highest mean a day on
the development years (+4.03 bps after costs, t +0.93), as the pre-registration says. No more
N2 variant is run.

**N3's choice for the holdout, fixed 2026-10-21 12:42 UTC by Efe after seven variants:** variant 1
(components 64, penalty 10.0, base-last; `variants/N3_chosen.json`), the highest mean a day on
the development years (+6.34 bps after costs, t +1.43), as the pre-registration says. It was
fixed after N2's holdout had failed; N3's own variants were all run before that. No more N3
variant is run.

**N4's choice for the holdout, fixed 2026-10-21 13:39 UTC after ten variants:** variant 2
(components 64, penalty 10.0, base-last; target scaled, side tenth, weights equal, sectors
none; `variants/N4_chosen.json`), the highest mean a day on the development years (+8.74 bps
after costs, t +2.07), as the pre-registration says. Variants 6, 10 and 8 also run on the same
years as exploration (amendment 6).

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

## N2's holdout (run once, 21 October 2026, commit 62358fe)

`data/news_reader/holdout_N2.json`; a second run is refused.

| | Days | Mean a day | t | 2022 / 2023 / 2024 / 2025 / 2026, bps a day |
|---|---|---|---|---|
| Variant 2: components 64, penalty 10.0 | 1,197 | −2.22 bps | −0.78 | −4.19 / +5.03 / −5.87 / −1.46 / −5.26 |
| The count alone (the comparison) | 1,197 | −3.10 bps | −1.19 | +2.53 / −0.11 / −4.56 / −5.91 / −8.70 |

**N2 failed on its holdout: t −0.78 < 2.50.**

A member's headlines to 15:30, read by ChronoBERT of the year before and a linear model, do not
tell its next close-to-close return against the other members with news. Before costs the
chosen variant made −0.22 bps a day, nothing; after them it lost in four of five years, 2023
the exception. Its development result (+4.03 bps a day, t +0.93, the highest of three) did not
carry over. It lost 0.88 bps a day less than counting alone, well within noise.

## N3's holdout (run once, 21 October 2026, commit fd9767c)

`data/news_reader/holdout_N3.json`; a second run is refused.

| | Days | Mean a day | t | 2022 / 2023 / 2024 / 2025 / 2026, bps a day |
|---|---|---|---|---|
| Variant 1: components 64, penalty 10.0, features base-last | 1,197 | +4.24 bps | +1.18 | +0.84 / +7.67 / +5.33 / −11.34 / +22.88 |
| The count alone (the comparison) | 1,197 | −3.10 bps | −1.19 | +2.53 / −0.11 / −4.56 / −5.91 / −8.70 |

**N3 failed on its holdout: t +1.18 < 2.50.**

A member's headlines to 15:30, read by PIT-4B of the year before and a linear model, did not
pass: +4.24 bps a day after costs, t +1.18, against 2.50. Unlike N2's, its result leans the
right way: before costs +6.24 bps a day, 7.34 a day ahead of counting alone, positive in four
of five years and close to its development result (+6.34). But 2025 lost 11.34 bps a day,
2026's +22.88 is nine months, and a t of +1.18 comes from noise about one time in eight. By
the pre-registration N3 is recorded as failed, and nothing trades on it. The lean is an
observation, not a finding: testing it again needs data not yet seen, the months from now on.

## N4's holdout (run once, 21 October 2026, commit 708ac05)

`data/news_reader/holdout_N4.json`; a second run is refused.

| | Days | Mean a day | t | 2022 / 2023 / 2024 / 2025 / 2026, bps a day |
|---|---|---|---|---|
| Variant 2: components 64, penalty 10.0, features base-last, target scaled, side tenth, weights equal, sectors none | 1,197 | +6.53 bps | +1.86 | +1.32 / +5.78 / +5.97 / −9.42 / +35.55 |
| The count alone (the comparison) | 1,197 | −3.10 bps | −1.19 | +2.53 / −0.11 / −4.56 / −5.91 / −8.70 |

**N4 failed on its holdout: t +1.86 < 2.50.**

A member's headlines to 15:30, read by PIT-4B and a linear model trained on a target scaled by
volatility, did not pass: +6.53 bps a day after costs, t +1.86, against 2.50; noise gives as
much about one time in 32. The result rests on 2026. Its 194 days made +35.55 bps a day (t
+3.00 on their own); the 1,003 days of 2022-2025 made +0.92, which is nothing, and 2025 lost
9.42 a day. The three variants explored beside it, close kin and as good on development, made
+3.09, +1.03 and +0.97 bps a day (t +0.98, +0.31, +0.32): development's ranking among them did
not carry over, and in all four 2026 was the best year. N4 was designed after these years had
been seen, so even a pass would have been a second look. It is recorded as failed, and nothing
trades on it. Whether 2026's strength is a lasting change, a change in the news, or chance,
only days not yet seen can tell.

## N4's exploratory runs on the holdout years (amendment 6; not tests)

Each once, after the test's own. Four tries sharing N4's 0.625% would each need t ≥ 2.96; a variant other than the test's counts only after a forward test in the shadow record.

| | Days | Mean a day | t | 2022 / 2023 / 2024 / 2025 / 2026, bps a day |
|---|---|---|---|---|
| Variant 2 (the test) | 1,197 | +6.53 bps | +1.86 | +1.32 / +5.78 / +5.97 / −9.42 / +35.55 |
| Variant 6: components 64, penalty 10.0, features base-last, target scaled, side tenth, weights inverse, sectors none | 1,197 | +3.09 bps | +0.98 | −1.24 / −0.18 / +6.67 / −8.25 / +22.91 |
| Variant 10: components 96, penalty 10.0, features base-last, target scaled, side tenth, weights equal, sectors none | 1,197 | +1.03 bps | +0.31 | −0.67 / −1.12 / +2.83 / −8.68 / +16.17 |
| Variant 8: components 128, penalty 10.0, features base-last, target scaled, side tenth, weights inverse, sectors none | 1,197 | +0.97 bps | +0.32 | −1.12 / −5.52 / +2.10 / −0.18 / +12.01 |

## Reported with the holdouts (not part of the pass)

Each choice run again on the holdout years by `dev_report.py reports`, reproducing its locked mean and days. Bps a day, t in brackets. The long side is against the mean of the day's members with news, the short side is that mean less it.

| | After costs | Before costs | Long side | Short side | Best 10 days' share | Without them |
|---|---|---|---|---|---|---|
| N2 | −2.22 (−0.78) | −0.22 (−0.08) | −0.64 (−0.35) | +0.42 (+0.22) | — | −5.30 |
| N3 | +4.24 (+1.18) | +6.24 (+1.74) | +1.64 (+0.67) | +4.60 (+2.19) | 108% | −0.33 |
| N4 | +6.53 (+1.86) | +8.53 (+2.43) | +3.71 (+1.50) | +4.82 (+2.41) | 63% | +2.42 |

| Day by day | Bps a day | t | Days |
|---|---|---|---|
| N3 less N2 | +6.46 | +1.61 | 1,197 |
| N4 less N3 | +2.30 | +1.07 | 1,197 |

2026 by month, bps a day after costs (days):

- N2: 01 −26.08 (20), 02 −58.01 (19), 03 +13.28 (22), 04 +3.15 (21), 05 +47.77 (20), 06 −64.77 (21), 07 −3.67 (22), 08 +32.00 (21), 09 +2.15 (21), 10 +1.81 (7)
- N3: 01 −1.47 (20), 02 +5.56 (19), 03 +29.90 (22), 04 +24.90 (21), 05 +104.51 (20), 06 −7.18 (21), 07 −34.69 (22), 08 +58.64 (21), 09 +42.25 (21), 10 −16.28 (7)
- N4: 01 −1.28 (20), 02 +38.33 (19), 03 +11.53 (22), 04 +64.59 (21), 05 +145.53 (20), 06 −3.02 (21), 07 −9.64 (22), 08 +57.74 (21), 09 +48.70 (21), 10 −40.93 (7)

## Reading the reports (21 October 2026)

- N3 less N2, +6.46 bps a day (t +1.61): reading the same member-days with PIT-4B rather
  than ChronoBERT did better day by day, though short of any bar.
- In N3 and N4 the short side carries more than the long (+4.60, t +2.19; +4.82, t +2.41):
  the members whose headlines read worst fell behind the others with news the next day.
- N3's result is ten days: without its ten best of 1,197 it made −0.33 bps a day. N4's is
  less so: +2.42 a day without its ten best, and it was positive on 56% of 2026's days
  against 50% in 2022-2025, so 2026 was more than a few days.
- N4's 2026 was strong in February, April, May, August and September and weak in January,
  June, July and October's first week; its best days fall in late May and late August, the
  heart of the earnings seasons. That is an observation to carry into a forward test, not a
  finding: these years have been seen.

## N5's record, before its first day (22 October 2026)

- **Frozen** at 10:44 UTC: `data/models/news-n4-forward.npz`, SHA-256
  `eafc16cd647c3f2ec545d2eb35d810a6fe1d04ef3032b41837b2fbe7fc4bcb15`, N4's variant 2 fitted
  by Efe on 259,967 member-days, 5 January 2016 to 9 October 2026. Its card,
  `challengers/news-n4-forward.yaml`, names the scorer's commit and the first day, 23
  October 2026.
- **The capture** (`qfin news-capture`, 15:31 New York time each session) asks for the
  articles created from 30 days before the window. An article created earlier and edited
  inside the window counts from its edit, and the record cannot see it: of 2016-2026's
  503,998 member article-pairs, 0.033% were created more than 30 days before they counted.
  The first margin, 2 days, missed 4 of 2,325 pairs in ten sessions (PR #72).
- **The record's path against the research's** (`forward_check.py`,
  `results_forward_check.txt`), ten sessions, 25 September to 8 October 2026: the same 2,325
  article-member pairs and 1,364 member-days; each headline's embedding to a cosine of
  1.000000; the same sides on 10 of 10 days; scores equal to `harness.n4_days`' to 0.000000
  bps. What the record measures is what N4's holdout measured.

## The checkpoints seen side by side (22 October 2026; an observation, not a test)

Measured for N6 on headlines alone (`eyes_check.py`, `results_eyes_check.txt`). It changes
nothing in N3's, N4's or N5's results or rules.

- **One coordinate system, drifting.** The ten PIT-4B checkpoints put the same headline in
  nearly the same place: its vectors' cosine is 0.75–0.995 between neighbouring checkpoints
  and about 0 between different headlines. The drift adds up. Against 202412, the
  checkpoints that read 2016–2018 give 0.58–0.69, though the headlines' arrangement stays
  alike (CKA 0.93–0.95).
- **Checkpoints 202012 and 202112, which read 2021 and 2022, have one dimension that
  dominates.** Dimension 1484 carries 34% and 18% of their variance, against 2% for 202412.
  Fitted on all years' rows, N3's and N4's principal components gave those years' rows a
  direction the other years hardly have. Scaling each dimension by its spread within each
  checkpoint brings their CKA with 202412 from 0.36 and 0.67 to 0.82 and 0.97.
- **A rotation fitted on 1,500 headlines does not help.** It scores below no rotation on the
  other 1,500, except between distant checkpoints.

Then two ways to bring the nine older checkpoints to 202412's scale were compared, on
headlines alone (`align_check.py`, `results_align_check.txt`). The rule was committed before
the run (1bad30a).

- **A** scales each dimension by its mean and spread on 20,000 anchor headlines.
- **B** adds a ridge map to 202412's scaled reading, fitted on those anchors.

On the 3,000 headlines neither had seen, B agrees with 202412 on 0.062 more of each
headline's ten nearest neighbours than A, on average over the nine checkpoints. Its cosine is
above A's for all nine; the oldest goes from 0.57 to 0.87. **The rule chose B.**

B's map is fitted on 202412's readings, so it may carry a little of what 202412 knows of
later years into the older years' features. That can only flatter N6's results on the years
already seen; it cannot reach a forward record, which 202412 reads itself.
