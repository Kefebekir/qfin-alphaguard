# 0026: KUTBIT's first agent: extreme minutes, on the stock's side

- **Status:** accepted; the agent's trust and each stock's correction amended on 22 November
  2026, after the placebo's first run and before the choice and the measurement (see the
  amendment at the end)
- **Date:** 2026-11-19 (agreed with Efe on 18 and 19 November)
- **Builds on:** decision 0023 (variants, placebos, the card), 0024 (the ladder; development
  retrained monthly) and 0025 (the hedge is a decision; KUTBIT never bets on the market).
- **Variant:** 1 of the 10 that 0023 allows.

## Context

- **Efe's wish, 18 and 19 November.** A stock-based bot that looks ahead, puts its risk on the
  right side and takes the initiative. Its first job: when a stock makes an extreme minute,
  follow it, bet against it or let it be, and choose how much to hedge. It does not bet on
  QQQ, whose direction the QQQ rule already trades; QQQ is only its hedge and its context.
- **What followed extreme minutes on 2016-2018** (`research/kutbit/events.py`,
  `results_events.txt`, #113). There were 16,343 fresh extreme minutes: at least 4 usual moves
  apart from QQQ, with none in the 30 minutes before.
  - On average they reversed a little: −2.4 bps apart from QQQ over 15 minutes (t −5.8).
  - They reversed more at the open (−7.9 bps) and when QQQ moved too (−5.1 bps over 30
    minutes). With a headline, no clear reversal.
  - Betting against all of them earned about one round trip of 2 bps. Betting against those
    without a headline at the open earned +7.6 bps, before costs.
  - The hedge took out 15% of the variance in general and 42% when QQQ moved too.
  - These pockets were found by splitting the learning years. Whether they hold is for this
    record's measurement.
- **Why an agent and not a rule.** Which context deserves which side, size, hold and hedge is
  a joint choice, and the pockets may combine in ways a table does not show.

## Decision

**1. When it acts.** Only at a fresh extreme minute of a stock, as `events.py` defines it. It
decides at that minute's close and is filled at the next bar's open, so it never sees what it
is forecasting.

**2. What it sees: 16 numbers.** Each is turned to the extreme minute's direction where it has
one, so "the same way as the move" is always positive.
- The minute itself: the move's size in usual moves; its direction; its range; its volume
  against the usual; QQQ's move in its own units, signed to the stock's direction.
- Its context: headlines naming the stock in the 30 minutes up to it; the close's distance
  from the session's VWAP.
- The session so far: the time; the gap at the open; the move since the open; the place in
  the day's range; the day's typical move so far.
- Before the open: the VIX; the stock's beta; headlines since the previous close; `habit`.

All 971 inputs of 0024 are a later variant.

**3. What it chooses: up to 91 actions.**
- Either it lets the move be, or it chooses, all together:
  - a side: with the move, or against it;
  - a size: a quarter, a half, one, two or four times the account's equity, as far as the
    position cap allows, so that a thin edge can be taken small and a strong one large;
  - a hold: 5, 15 or 30 minutes;
  - a hedge: none, half or all of the position's market part (0025).
- It never bets on QQQ. The hedge stays between none and all.
- An action whose unhedged market exposure would pass Efe's limit M is masked; the agent
  never chooses it.

**4. Fills, costs and its reward.**
- Entry is at the next bar's open. Exit is at the close of the hold's last bar, or the
  session's last bar if that comes first.
- A stock's round trip costs 5 bps, since spreads widen at extreme minutes. 10 bps, costs
  doubled, is reported beside it, as 0023 requires; so is 2 bps.
- QQQ's round trip costs 0.5 bps.
- The reward is the account's utility at Efe's risk appetite γ = 2, about half Kelly (0025),
  chosen by Efe on 19 November: the agent goes large when it sees a strong edge, and an
  agent twice too sure of its edge still bets no more than Kelly.
- The position cap is 4x of equity, Efe's wish. M is 2x, so a position above 2x is at
  least partly hedged.
- Efe may change γ, the cap and M before any training, never after.

**5. Learning, with every action's outcome known.** At an event, the outcome of every action
is in the stored minutes, so the agent learns from all of them at once. This stage is a
contextual bandit. Learning to manage an open position minute by minute is the second stage, and real
reinforcement learning. It is built only if this stage passes.

**6. It learns as it goes, at two speeds, as 0024 asks.**
- **Every minute.** At each minute's close, every event whose outcome has just become known
  is given to the agent with the outcome of all its actions. An outcome is known once the
  longest hold, 30 minutes, has ended, or the session has. The agent may update itself
  before its next decision.
- **Each stock has a part of its own,** pulled toward what the agent learned from all
  stocks. On 2016-2018 a stock had about 10 events a month, too few to learn from alone; the
  more events a stock has had, the more its own part may say. This is where the agent learns
  each stock's character and takes its own initiative.
- **Every month,** the part shared by all stocks learns afresh on every event before the
  month, so the latest few events cannot overwrite what thousands taught it.
- The agent never sees an outcome before it is known.
- **Settings are chosen on 2019:** the network's size, its learning rates, and how strongly a
  stock's own part is pulled toward the shared one. Every month of 2019 runs as above, from
  every event since 2016.
- **It is measured once on 2020-2021,** the same way, as live.
- 2022-2026 are not read.

**7. The bars it must beat, under the same fills, costs and γ.**
- Doing nothing.
- Betting against every event: 15 minutes, 1x of equity, hedged in full.
- Betting against events without a headline, the same way.
- Betting against events without a headline at the open, the same way.
- The best simple policy: for each kind of event (at the open or not, QQQ moved or not, a
  headline or not), the action with the highest utility on every event before the month,
  chosen afresh each month as the agent learns afresh. A kind with fewer than 100 events
  before the month lets the move be: on 2016-2018 one kind had 2 events, and the best of 91
  actions on a few events is chance, not a rule.
- The same agent trained on shuffled outcomes. Each event's outcomes are dealt to another
  stock's event in the same part of the day. That agent learns only what is common to all
  events.

**8. The test, on 2020-2021.**
- The agent passes if its mean utility an event beats that of every bar. The t of the margin
  over the best bar, clustered by day, must reach 2.
- After 5 bps its certainty equivalent must be above zero.
- Failing either, it is written down as failed and not used.

**9. The placebo.**
- The same procedure runs on random-walk prices of every session (the skill test's walk).
- There the agent may not beat doing nothing at a t of 2, since nothing can be learned.
- It runs first. The real measurement waits for it.

**10. What Efe sees.** A page that shows, on 2019 and, after the measurement, on 2020-2021:
- the shares of its actions by kind of event, and where it hedges;
- its certainty equivalent against each bar, at 2, 5 and 10 bps;
- its forecast utility against what followed, by tenths;
- sample events: the chart before and during the hold, the action it chose and the outcome;
- where it chose differently from the best simple policy, and what that earned or cost.

**11. Who writes what.**
- Claude: the event gym (states, actions, every action's outcome, masks, bars, shuffled and
  random-walk data); the harness that gives the agent each minute's events and, once known,
  their outcomes, in time order; the measurement, the page, the tests and a guide, step by
  step.
- Efe: the agent itself, with PyTorch only as a tool:
  - its network, its loss and its learning loop;
  - how it updates itself each minute, and each stock's own part;
  - every run of it.

## Consequences

- Next: the event gym; the harness and the page; then the guide; then Efe's agent, the
  placebo, the choice on 2019 and the measurement.
- The skill test and the forecaster of 0024 go on as before. This agent is a separate variant
  on the same data.
- If it passes, the second stage manages the open position minute by minute.
- If it fails, that is written down as it is, and the next variant starts from what its page
  showed.

## Amendment of 22 November 2026: the agent acts only on what it has proved

- **What happened.** Efe ran the placebo on 20 November, on the random walks of 2016-2021
  (150,517 events). It passed: on the 52,247 events of 2020-2021 the agent did not beat doing
  nothing (−0.25 bps of utility an event, t −2.45). But where nothing can be foreseen it should
  not have traded at all, and it made 658 trades, which lost about 20 bps of utility each. The
  choice on 2019 and the measurement had not run.
- **Why: two numbers it acted on were never asked for proof.**
  - Its trust, the slope of what followed on what it forecast over the latest fifth, is itself
    measured with noise, and keeping it between none and all cut off its negative draws but
    kept its positive ones. On the random walks it came out at 0.21, 0.14 and 0.21 for the
    three holds at the start of 2020, where it should have been nothing. The network's edges
    on noise, about 35-50 bps on the trades it made, came through at a fifth of their size and
    cleared the costs.
  - Each stock's correction, its mean miss pulled toward nothing by `pull` events, was not
    asked for proof either. The random walks take each minute's usual move from the real
    prices, so spring 2020's swung widely, and after them some stocks' corrections reached
    7.6 bps, above the 5 bps a round trip costs.
- **The check, on the random walks only.** Efe fitted the agent at six month starts, four
  months apart, and counted the trades of the four months after each:

  | Start | t of the slope (5, 15, 30 min) | Trust as it was | Trust cut | Trades as it was / trust cut |
  | --- | --- | --- | --- | --- |
  | Jan 2020 | +1.5, +1.2, +2.1 | 0.21, 0.14, 0.21 | 0, 0, 0.01 | 47 / 0 |
  | May 2020 | +0.7, +1.6, +1.4 | 0.11, 0.25, 0.28 | 0, 0, 0 | 551 / 418 |
  | Sep 2020 | −0.6, +1.2, +0.4 | 0, 0.14, 0.08 | 0, 0, 0 | 33 / 0 |
  | Jan 2021 | −0.2, +0.2, −0.3 | 0, 0.02, 0 | 0, 0, 0 | 0 / 0 |
  | May 2021 | −0.7, −1.8, −2.9 | 0, 0, 0 | 0, 0, 0 | 170 / 170 |
  | Sep 2021 | −1.4, −2.3, −1.2 | 0, 0, 0 | 0, 0, 0 | 0 / 0 |

  The slopes' t lay between −2.9 and +2.1, as noise does. The trades left once the trust was
  cut came from the stocks' corrections alone: counted with no network, the corrections open
  0, 434, 0, 0, 170 and 0 trades in the same months, and none once each is cut by two of its
  errors. With a pull of 20 they open 1,273, and none once cut.
- **The change: every number the agent acts on must clear two of its standard errors, the t
  the gate asks of the agent itself** (`PROOF = 2`).
  - The trust is the slope less two of its standard errors, kept between none and all. The
    error counts a day's events together, as the gate does.
  - A stock's correction is its pulled mean miss cut toward nothing by two of its errors: the
    root of its squared misses over its count plus `pull`. The agent keeps each stock's sum of
    squared misses beside its sum of misses.
  - Nothing else changes: the network and its learning, the settings to choose among (a hidden
    layer of 32 or 64, a pull of 20 or 100), the actions, γ, the costs, the bars and the gate.
- **Why it is fair.** The change was decided on random walks alone, before the choice on 2019
  and the measurement, so it cannot have been chosen to suit a result on real prices. Efe
  agreed to cut the trust on 21 November; the corrections follow the same rule, found in his
  check on 22 November. What it costs: a proven edge is acted on at less than its measured
  size until it has more proof. At a t of 5 the trust is 60% of the slope, at 10 it is 80%.
- **Checked on made-up events** (`tests/test_kutbit_agent.py`, 23 tests). With a planted edge
  the agent still passes the gate, at a t of 22.1 over the best bar (22.8 before). On made-up
  noise its trust is nothing (the slope alone gave 0.10-0.17) and it makes no trade (9
  before). All 10 bugs planted in the two cuts were caught.
- **Next.** The change goes into `agent.py` as the guide's steps 1, 4 and 5 show; then Efe
  runs the placebo again, the choice on 2019 and the measurement.
