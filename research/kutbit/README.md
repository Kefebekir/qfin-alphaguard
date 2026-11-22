# KUTBIT

Efe's own intraday model with leverage (decisions [0023](../../docs/decisions/0023-kutbit.md),
[0024](../../docs/decisions/0024-kutbit-architecture.md),
[0025](../../docs/decisions/0025-kutbit-hedge.md) and
[0026](../../docs/decisions/0026-kutbit-first-agent.md)).
It is separate from the QQQ rule: its own code, its own Guard file and its own accounts.

| Part | Where | State |
|---|---|---|
| What it sees at each minute: 120 minutes of chart and a few numbers, the same in training and live | `src/qfin_alphaguard/kutbit/features.py` | done |
| What happened after each minute: the return 1, 5, 15, 30 and 60 minutes on, and the best and worst moments along the way | `labels.py` | done |
| What the forecaster learns: that return with QQQ's part taken out, in the usual moves of the minutes held | `labels.py` (`targets`), `target_report.py`, `results_target.txt` | done |
| The data set: every instrument's sessions of 2016-2026, minute by minute, with what was known before each | `dataset.py` → `data/kutbit/`, `results_dataset.txt` | done |
| The skill test: how much of the target a linear model of the same inputs forecasts; the ruler Efe's network must beat | `skill.py` → `data/kutbit/skill/skill.html` | ready; Efe runs it |
| The hedge study: how much of a book's market part to hedge with QQQ, under a gross cap | `hedge_study.py`, `results_hedge.txt` | done |
| What follows an extreme minute, on 2016-2018: the ground for Efe's first agent | `events.py`, `results_events.txt` | done |
| The event gym: every extreme minute's 16 numbers, the outcome of all 91 actions, the bars, the placebo's random walks and the shuffled outcomes | `gym.py`, `results_gym.txt` | done |
| The harness: feeds the agent minute by minute in time order, never an outcome before it is known; chooses its setting on 2019; runs the placebo; measures it once on 2020-2021 against the bars; writes the page of what it did | `harness.py` | done |
| Efe's first agent: at an extreme minute, follow, bet against or let it be, and how much to hedge | rules: decision 0026; design A in `AGENT_GUIDE.md`; `agent.py`, Efe's, checked by `tests/test_kutbit_agent.py` | written by Efe; its first placebo passed but it traded on noise, so its trust and each stock's correction now count only beyond two standard errors (0026's amendment of 22 November); next, Efe types the change, then the placebo again, the choice on 2019 and the measurement |
| The bank of real paths: the shapes of what may follow, in the target's units, with QQQ's path beside each | | next |
| The simulator: fills, costs, margin and stops | | to come; its broker rules are Efe's |
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
  - the usual size of a move at each minute (`swing`) and its beta to QQQ, from the same 20
    sessions;
  - what it did apart from QQQ over the 30 minutes after each minute, in the 5 sessions before
    (`habit`);
  - the headlines naming it, by the minute they were first published, and those since the
    previous close;
  - the VIX's last close.
- **No look-ahead.** The inputs at a minute come only from bars up to it and from what was known
  before the open. `features.py`'s tests change every later bar and find the inputs unchanged.
  `dataset.py` checks the same on real sessions.

`days.parquet` has one row for each instrument's session. Its `year` and `row` point into that
year's `<year>.npz`.

**What it holds** (`results_dataset.txt`):
- About 16,500 sessions a year of 66 instruments; 12,819 in 2026, to 9 October. 1.3 GB in all.
- About 99% of sessions are usable, but 91.7% in 2016, whose first weeks have no minutes before
  them.
- 98.7-99.98% of minutes had a trade, year by year.
- A 30-minute position moved 24.8 bps on average, up or down, and its worst moment along the way
  was 24.9 bps against it.
- **A move is read against its own minute.** Against the day's volatility alone, a move of the
  first five minutes was about four times a midday one. Against `swing` the two are alike: 0.77
  and 0.76.
- **The market's part.** A stock's beta to QQQ has a median of 0.69. On 2016-2021, taking
  QQQ out at that beta removes 29% of the variance of a stock's 30-minute moves, and the mean
  correlation between two stocks falls from 0.28 to 0.06.
- **`habit` says little, but something.** On 2016-2021 its correlation across stocks with the
  next 30 minutes' move apart from QQQ is +0.005 (t 2.9), the pattern of Heston, Korajczyk and
  Sadka (2010). A trade needs far more than that; it is one input among the others.
- **Splits.** Five sector ETFs split 2-for-1 on 5 December 2025: XLB, XLE, XLK, XLU and XLY.
  `dataset.py` found them from the overnight jump, and Alpaca's corporate actions list the same
  five on the same day.
- **A gap.** AVGO was in 2017's universe, but the minute store has nothing of it before 2018.

## The target

`targets` turns the labels into what the forecaster learns: the return of a long position hedged
with QQQ at the beta known before the open (`apart`), over the usual size of a move in the minutes
held (`scale`), which is `z`.

On 3,000 sessions of stocks from 2016-2021 (`results_target.txt`):

| Held | Hedged move | Share of it a forecast must capture to pay 2 bps |
|---|---|---|
| 1 minute | 7.3 bps | 27% |
| 5 minutes | 15.3 bps | 13% |
| 15 minutes | 24.6 bps | 8% |
| 30 minutes | 32.9 bps | 6% |
| 60 minutes | 43.5 bps | 5% |

- **The cost is the same however long a position is held, and the move grows.** A one-minute
  trade has to be right about a quarter of the move to pay for itself; an hour's, a twentieth.
  The hedge's own cost is not in the 2 bps.
- **`z` is in nearly the same units through the day:** 0.90 in the first half hour, 0.83 at
  midday and 0.76 after 15:00 at 30 minutes, where the raw moves differed about fourfold.
- **Its tails are fat:** a kurtosis of 7 at 30 minutes and 20 at one minute, against 3 for a
  normal distribution. The shape of what follows has to come from real past moves.

## The skill test

The first rung of decision 0024's ladder, with its rules written there before it runs. A ridge
regression of each horizon's `z` on the 971 inputs the network will see, fitted on 2016-2019 and
measured once on 2020-2021, beside two known patterns. It is a ruler, not a variant.

Efe runs it, in this order: a check of speed that shows no score (about 12 seconds), the
random-walk placebo, then the real run, which waits until the placebo passes. The placebo's
full run took about 7 minutes on 17 November.

```bash
OMP_NUM_THREADS=8 uv run python research/kutbit/skill.py smoke
OMP_NUM_THREADS=8 uv run python research/kutbit/skill.py random
OMP_NUM_THREADS=8 uv run python research/kutbit/skill.py real
```

The page is `data/kutbit/skill/skill.html`.

## The hedge study

Decision 0025's evidence. On 876,570 stock windows of 30 minutes in 2016-2021, an edge is
planted in the move apart from QQQ, and QQQ is given none. For three books (one stock, ten long,
five long and five short), each gross cap and each risk appetite γ, it finds the share of the
cap and the share of the market part hedged that the account would value most, after costs.

```bash
PYTHONPATH=src uv run python research/kutbit/hedge_study.py
```

It takes about a minute and writes `results_hedge.txt`. In short:

- once an edge pays its cost, the cap fills, and at γ 1 or 2 the hedge's room under the cap is
  worth more than the risk it takes out;
- the book's own shorts are the best hedge: five long and five short leave a beta of 0.10;
- careful appetites (γ 4, 8) and jumpy markets hedge more;
- the price of leaving ten long unhedged at 10x is in the bad days: 15% of days below −5%,
  against 2% hedged in full, with an edge of 0.1 in every window.

## Extreme minutes

Efe's idea for the first agent: when a stock makes an extreme move in one minute, follow it,
bet against it or let it be, and choose how much to hedge. `events.py` measures what followed
such minutes on 2016-2018, the years an agent learns on; 2019 and later are not read.

```bash
PYTHONPATH=src uv run python research/kutbit/events.py
```

It takes about 15 seconds and writes `results_events.txt`. On 16,343 fresh extreme minutes, at
least 4 usual moves apart from QQQ with none in the 30 minutes before:

- they reverse a little: −2.4 bps apart from QQQ over 15 minutes (t −5.8), and 55% reverse;
- more at the open (−7.9 bps) and when QQQ moved too (−5.1 bps over 30 minutes); with a
  headline, no clear reversal;
- the bounce to the next open, −1.0 bps, cannot be caught;
- betting against all of them earns about a round trip of 2 bps and no more; at the open and
  without a headline, +7.6 bps before costs;
- the hedge takes out 15% of the variance in general, but 42% when QQQ moved too: that is
  where it belongs.

## The event gym

Decision 0026's ground for Efe's first agent. At each fresh extreme minute it gives the 16
numbers the agent sees, the outcome of all 91 actions (a side, a size from a quarter to four
times equity, a hold of 5, 15 or 30 minutes, a hedge of none, half or all), which actions Efe's
limits allow, the account's utility at γ 2, and the bars: doing nothing, betting against every
event, against those without a headline, against those at the open, and the best simple
policy of each kind of event. It builds the same events on the skill test's random walks, for
the placebo, and deals outcomes between stocks, for the shuffled bar.

```bash
PYTHONPATH=src uv run python research/kutbit/gym.py learning   # 2016-2018 in memory, ~20 s
PYTHONPATH=src uv run python research/kutbit/gym.py build      # Efe: 2016-2021, both sources
```

`learning` writes `results_gym.txt`. On 2016-2018, after 5 bps a round trip, betting against
every event loses 3.1 bps an event; only the open pays, and the best simple policy, chosen on
the same years, earns 0.8 bps an event. Kinds with fewer than 100 events are let be, since the
best of 91 actions on a few events is chance.

## The harness

It runs Efe's agent, from `agent.py` (`make(settings)` and `SETTINGS`), over the event gym as
live. At each month's start the agent learns afresh on every event before the month; at each
minute's close it is given the events whose outcomes have just become known, then asked for its
action at that minute's extreme minutes. It is never given an outcome before it is known, and
an action Efe's limits forbid is refused.

```bash
PYTHONPATH=src uv run python research/kutbit/gym.py build        # once
PYTHONPATH=src uv run python research/kutbit/harness.py placebo   # random walks first
PYTHONPATH=src uv run python research/kutbit/harness.py choose    # the setting, on 2019
PYTHONPATH=src uv run python research/kutbit/harness.py measure   # once, on 2020-2021
```

Each run's record keeps the SHA-256 of `agent.py`, `gym.py`, `harness.py` and `skill.py`:
`choose` waits for a placebo that passed on today's code, `measure` for that placebo and a choice
made on today's code, and `measure` runs only once. The pages are
`data/kutbit/agent/choice.html` (2019) and `data/kutbit/agent/agent.html` (2020-2021). The
agent on dealt outcomes learns from outcomes dealt within each month, so it never learns from
beyond the month it trades in.
