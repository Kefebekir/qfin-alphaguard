# KUTBIT

Efe's own intraday model with leverage ([decision 0023](../../docs/decisions/0023-kutbit.md)).
It is separate from the QQQ rule: its own code, its own Guard file and its own accounts.

| Part | Where | State |
|---|---|---|
| What it sees at each minute: 120 minutes of chart and a few numbers, the same in training and live | `src/qfin_alphaguard/kutbit/features.py` | done |
| What happened after each minute: the return 5, 15, 30 and 60 minutes on, and the best and worst moments along the way | `labels.py` | done |
| The data set: every instrument's sessions of 2016-2026, minute by minute, with what was known before each | `dataset.py` → `data/kutbit/`, `results_dataset.txt` | done |
| The simulator: fills, costs, margin and stops | | next; its broker rules are Efe's |
| The decision layer | | to come; its rule is Efe's |
| The training harness and its placebos | | to come; Efe designs and trains the networks |

```bash
uv run python research/kutbit/dataset.py
```

## The data set

- **Instruments:** each year's trading universe, the 50 most liquid S&P 500 members chosen the
  evening before the year, and the 16 ETFs of G1's gym.
- **A session**, on a grid of one bar a minute, as traded. A minute without a trade carries the
  last close with no volume and is marked unseen. Each session also keeps:
  - the previous close, the volatility and the volume usual at each minute, from the 20
    sessions before it, on its own basis, splits taken into account;
  - the headlines naming it, by the minute they were first published, and those since the
    previous close;
  - the VIX's last close.
- **No look-ahead.** The inputs at a minute come only from bars up to it and from what was known
  before the open. `features.py`'s tests change every later bar and find the inputs unchanged.
  `dataset.py` checks the same on real sessions.

`days.parquet` has one row for each instrument's session. Its `year` and `row` point into that
year's `<year>.npz`.

**What it holds** (`results_dataset.txt`):
- About 16,500 sessions a year of 66 instruments; 12,819 in 2026, to 9 October. 776 MB in all.
- About 99% of sessions are usable, but 91.7% in 2016, whose first weeks have no minutes before
  them.
- 98.7-99.98% of minutes had a trade, year by year.
- A 30-minute position moved 24.8 bps on average, up or down, and its worst moment along the way
  was 24.9 bps against it.
- **Splits.** Five sector ETFs split 2-for-1 on 5 December 2025: XLB, XLE, XLK, XLU and XLY.
  `dataset.py` found them from the overnight jump, and Alpaca's corporate actions list the same
  five on the same day.
- **A gap.** AVGO was in 2017's universe, but the minute store has nothing of it before 2018.
