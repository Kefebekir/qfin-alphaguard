# Training the news reader: Efe's guide

This is the model part of pre-registrations N1 and N2. Everything around it is fixed and
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
  tenth each day.

## The model the pre-registrations allow

Principal components of the 768 embedding numbers, fitted on the training years, then a ridge
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
    embedding = list(range(768))  # the first 768 columns
    count = [768]  # the last column, log(1 + count)
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

## What to expect

Most ideas fail; our own measures say the news's direction is mostly priced within a minute
(research/frontier). A failed test is a result: it goes in the ledger and saves money.
