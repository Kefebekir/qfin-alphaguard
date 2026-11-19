# 0026: KUTBIT's first agent: extreme minutes, on the stock's side

- **Status:** accepted
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

**3. What it chooses: 37 actions.**
- Either it lets the move be, or it chooses, all together:
  - a side: with the move, or against it;
  - a size: half or all of the position cap;
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
- The reward is the account's utility at Efe's risk appetite γ = 4 (0025).
- The position cap is 2x of equity, so half is 1x. M is 2x.
- Efe may change γ, the cap and M before any training, never after.

**5. Learning, with every action's outcome known.** At an event, the outcome of every action
is in the stored minutes, so the agent learns from all 37 at once. This stage is a contextual
bandit. Learning to manage an open position minute by minute is the second stage, and real
reinforcement learning. It is built only if this stage passes.

**6. Years and schedule.**
- Settings are chosen on 2019: the network's size, its learning rate, how long it learns.
  For each month of 2019 it learns on every event before that month, from 2016 on.
- It is measured once on 2020-2021, the same way: each month by an agent that learned on
  every event before it, as live (0024).
- 2022-2026 are not read.

**7. The bars it must beat, under the same fills, costs and γ.**
- Doing nothing.
- Betting against every event: 15 minutes, all of the cap, hedged in full.
- Betting against events without a headline, the same way.
- Betting against events without a headline at the open, the same way.
- The best simple policy: for each kind of event (at the open or not, QQQ moved or not, a
  headline or not), the action with the highest utility on every event before the month,
  chosen afresh each month as the agent learns afresh.
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
  random-walk data), the measurement, the page, the tests and a guide, step by step.
- Efe: the agent itself (its network, its loss and its learning loop, with PyTorch only as a
  tool), and every run of it.

## Consequences

- Next: the event gym and the page; then the guide; then Efe's agent, the placebo, the choice
  on 2019 and the measurement.
- The skill test and the forecaster of 0024 go on as before. This agent is a separate variant
  on the same data.
- If it passes, the second stage manages the open position minute by minute.
- If it fails, that is written down as it is, and the next variant starts from what its page
  showed.
