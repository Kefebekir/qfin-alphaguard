# Training the news reader: Efe's guide

This is the model part of pre-registrations N1, N2 and N3. Everything around it is fixed and
tested: the data, the ChronoBERT vintages, the policies, the scores and the holdout lock
(`harness.py`, tested by `tests/test_news_harness.py`). What you build is one small model, and
the choices you make with it are counted.

## What the model sees and predicts

- **N1, the QQQ rule's day.** One row a session. `X`: the mean ChronoBERT embedding of the
  headlines known from 16:00 the day before to 09:55 (768 numbers), and log(1 + how many
  there were). `y`: the rule's day return over QQQ's 14-day volatility. The harness sizes
  the rule ×1.5 on days your forecast is in the top third of its training forecasts, ×0.5 in
  the bottom third.
- **N2, members' next day.** One row a member and session with news. `X` likewise for that
  member's headlines up to 15:30. `y`: its next close-to-close return less the average of the
  members with news that day. The harness goes long your top tenth and short your bottom
  tenth each day; members tied at a tenth's edge are all in (amendment 5).

## The model the pre-registrations allow

Principal components of the embedding numbers (768 for N1 and N2, 4,096 for N3), fitted on the training years, then a ridge
regression on them and the log count. Your two choices: the number of components and the
ridge penalty. In scikit-learn:

```python
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make(components=16, penalty=10.0):
    """A fresh, untrained model: PCA on the embedding, the count as it is, then ridge."""
    embedding = slice(0, -1)  # every column but the last: the embedding (768 or 4,096)
    count = slice(-1, None)  # the last column, log(1 + count)
    return lambda: Pipeline(
        [
            (
                "features",
                ColumnTransformer(
                    [
                        (
                            "pca",
                            PCA(n_components=components, random_state=0),
                            embedding,
                        ),
                        ("count", "passthrough", count),
                    ]
                ),
            ),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=penalty)),
        ]
    )
```

`make(16, 10.0)` returns a function that builds a new model; the harness calls it once for
each test year, so no year's model has seen that year.

## Step by step

1. **Development run.** Save your model as `research/news_reader/efe_model.py` and run:

   ```bash
   uv run python -c "
   import sys; sys.path.insert(0, 'research/news_reader')
   import harness, efe_model, json
   print(json.dumps(harness.run('N1', efe_model.make(16, 10.0)), indent=2))
   "
   ```

   It fits and scores 2018, 2019, 2020 and 2021, each on the years before it, and prints the
   mean daily difference against the rule, its t-statistic, and each year. Look as often as
   you like: these years are for choosing.
2. **Count every variant.** Each (components, penalty) pair you run is a variant. Before the
   next one, add a line for it under "Variants tried" in `LEDGER.md`: the test, the settings,
   the development result. At most 10 for each test.
3. **Compare with counting.** `harness.run('N1', None, count_only=True)` scores a model that
   reads only how many headlines there were. A variant worth choosing should beat it on
   development: otherwise the language model adds nothing.
4. **Choose, then commit.** Write the chosen variant in `variants/N1_chosen.json`, for example
   `{"components": 16, "penalty": 10.0}`, add the line to the ledger, and commit both.
5. **The holdout, once.**

   ```bash
   uv run python -c "
   import sys; sys.path.insert(0, 'research/news_reader')
   import harness, efe_model, json
   print(json.dumps(harness.run_holdout('N1', efe_model.make(16, 10.0)), indent=2))
   "
   ```

   It refuses to run if the choice is not committed, or if the holdout has been run; it
   stores the result in `data/news_reader/holdout_N1.json` with the commit. Pass: t ≥ 2.24.
   Write the result in the ledger whatever it is.
6. **If it passes:** the model is frozen on a challenger card (`challengers/README.md`) and
   judged forward in the shadow record; it trades only after it is promoted there and you
   decide so.

N2 is the same with `'N2'` in place of `'N1'`.

## N3: N2 read by PIT-4B

N3 is N2 with a larger reader: PIT-4B, a 4-billion-parameter point-in-time model. Each
headline was read by the checkpoint of the year before its own (`embed_pit.py`), so one set
of features serves every test year. Three things differ from N2:

- **The embedding has 4,096 numbers**, not 768. Pick the columns from the end, as the `make`
  above does: every column but the last is the embedding, the last is the count. Then one
  `make` serves N1, N2 and N3, and a forgotten width cannot silently mix the count into the
  embedding. (Against picking them by a list of numbers, it moves forecasts only by float32
  rounding, about 1e-6.)
- **Two feature sets**, and which one a variant uses is part of the variant: `base-last`
  and `base-mean` (the state at the headline's last token, or the mean over its tokens). At
  most 10 variants across the two, each in the ledger with its feature set. (The fine-tuned
  PIT-4B-FT was dropped before any result, amendment 4: its fine-tuning data was written in
  2024.)
- **The run names the set:**

  ```bash
  uv run python -c "
  import sys; sys.path.insert(0, 'research/news_reader')
  import harness, efe_model, json
  print(json.dumps(harness.run('N3', efe_model.make(16, 10.0), features='base-last'), indent=2))
  "
  ```

  and `variants/N3_chosen.json` names it too, for example
  `{"components": 16, "penalty": 10.0, "features": "base-last"}`; the holdout refuses a
  choice without one. The comparison with counting is N2's: `harness.run('N3', None,
  count_only=True)` reads the same member-days.

A year of N3 holds up to 240,000 rows of 4,097 numbers (about 4 GB in float32), so run it
when the embedding job is not using the computer's memory.

The pre-registration also lets a variant add the member's return that day, its 20-day
volatility and its momentum from 12 months to 1 month; the harness does not supply them yet.

## What to expect

Most ideas fail; our own measures say the news's direction is mostly priced within a minute
(research/frontier). A failed test is a result: it goes in the ledger and saves money.

## N4: N3 read with less noise

N3's holdout made +4.24 bps a day after costs, but its long-short swings by about 124 bps a
day: a Sharpe ratio near 0.5. N4 (`preregistrations/N4_member_news_less_noise.txt`) keeps N3's
features and your model, and adds four levers against the noise. Each variant sets all four;
those you leave out take N3's values, and all four go in the ledger.

| Lever | Values (N3's first) | What it does |
|---|---|---|
| `target` | `plain`, `scaled` | `scaled` divides the target by the member's volatility, so a volatile stock's noise weighs less in training |
| `side` | `tenth`, `fifth`, `third` | how many members go long and how many short each day; more members, less noise |
| `weights` | `equal`, `inverse` | `inverse` weights each member by one over its volatility within its side |
| `sectors` | `none`, `neutral` | `neutral` ranks each statistical sector apart and holds each sector as much long as short |

A member's volatility is the standard deviation of its last 20 daily returns, ending the
session before the day. The sectors are 11 groups a year, formed from the year before's
returns (`risk.py`): energy, banks, technology, utilities and so on fall together without any
outside data.

```bash
uv run python research/news_reader/dev_report.py N4 64 10.0 base-last target=scaled side=fifth
```

The bar is N3's counting alone (−1.22 on development). As before: at most 10 variants on
2018-2021, the highest goes once to 2022 – 9 October 2026, which passes at t ≥ 2.50. Those
years were seen through N2's and N3's holdouts before N4 was designed, so a pass only earns
N4 a place in the shadow record.

One way to read the levers apart: start from N3's best settings (64 components, base-last),
change one lever a variant, and keep what helps.

## The reports beside the pass

After the holdouts, the pre-registrations ask for more than the pass: the result before costs,
the long side and the short side apart, and N3 less N2 and N4 less N3, day by day. One command
runs each committed choice again on the holdout years, day by day, and stops unless each
reproduces its locked mean and days:

```bash
uv run python research/news_reader/dev_report.py reports
```

It adds them to the ledger and to `data/news_reader/dev/reports.html`, with a diagnostic of
2026: each month, and how much of the result the ten best days carry.

## N6: our own network on the frozen eyes

N6 replaces N4's PCA and ridge with a network you train: PIT-4B stays frozen as the eyes, and
everything after it is learned. Its rules are pre-registration N6 (approved 22 October 2026):
at most ten variants on 2018-2021, each written in the ledger before it runs; a variant has to
beat both its placebos; the best of those that do is chosen, then looks once at 2022-2026,
then is judged forward.

What it reads, built once (`n6_eyes.py`, `n6_context.py`, `n6_data.py`):

- each member-day's headlines one by one, up to the last 32, in 202412's scaled reading
  (method B), with how many hours before 15:30 each became known;
- its context from the closes of the session before: the member's returns, volatility, traded
  value, 52-week high, beta and sector; the market's returns, VIX family, dispersion and
  co-movement; the scheduled FOMC, CPI and payrolls days.

The network (`n6_model.py`): a learned projection of each headline to `dim` numbers (started
from the training headlines' principal components, or at random), attention that weighs the
headlines, a small network on the context, and a head that joins them. It is trained on N4's
scaled target (`loss=mse`) or on the long-short Sharpe ratio (`loss=sharpe`), with a second
head on the size of the move (`aux`).

The levers, as `name=value` (defaults in brackets): `dim` (32), `attention` (16), `context`
(32), `hidden` (32), `dropout` (0.1), `loss` (mse), `aux` (0.1), `lr` (0.001), `decay`
(0.0001), `epochs` (10), `days_per_batch` (8), `start` (pca), and, by amendment 7, `inputs`
(all; `news` leaves the context out). A model has at most 1,100,000
parameters (decision 6, amendment 8): `dim` 32 uses about 135,000 in all, 64 about 269,000,
128 about 535,000 and 256 about 1.06 million. Up to about 600,000 fits the Arty A7-100T's
on-chip memory in int8; above it the weights wait in the board's DDR.

Before the first run, a check that shows no score:

```bash
PYTHONPATH=src data/llmenv/bin/python research/news_reader/n6_run.py smoke
```

A run (it writes its row in `LEDGER.md` first, and refuses an eleventh, a name or settings
already run):

```bash
PYTHONPATH=src data/llmenv/bin/python research/news_reader/n6_run.py v1 dim=32 loss=mse
```

When you are done (ten at most), the choice by the rule, written to
`variants/N6_chosen.json`, after which no variant runs:

```bash
PYTHONPATH=src data/llmenv/bin/python research/news_reader/n6_run.py choose
```

For each development year 2018-2021 it trains five seeds on the years before (less five
sessions) and averages them, does the same for the two placebos (the day's headlines dealt
again among its members, and the context alone), and scores all three with N4's variant 2's
portfolio, beside N4's variant 2 itself. About 20 minutes on the RTX 5070. The page is
`data/news_reader/n6/dev/<name>.html`.

Two things the tests found. A projection learned from 4,096 numbers memorises when the
training rows are few, so `start=pca` begins where N4 stood and learns from there. And a
model must beat both placebos: one that does not is making money from something other than
the headlines.
