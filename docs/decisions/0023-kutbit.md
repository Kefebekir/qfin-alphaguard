# 0023: KUTBIT, Efe's own intraday model with leverage

- **Status:** accepted
- **Date:** 2026-11-15
- **Builds on:** decisions 0014 and 0019 (Guard), 0020 (the shadow record) and 0021 (the
  live engine). It changes nothing in the QQQ rule.

## Context

- **The QQQ rule stays as it is.** It earns over the years at a steady risk: a Sharpe ratio of
  1.31 at any leverage on 2016–2026, and 1.26 after its paper's data ended. Efe is content with
  it, and changing it would break what works.
- **Efe wants a second model, entirely his own, called KUTBIT.** He asked for one that:
  - reads each instrument's chart;
  - measures its own expected gain and its own risk;
  - trades only when that is worth it under settings he chooses;
  - keeps adapting to each stock;
  - uses leverage.
- **What we already know:**
  - G1 (a network trained by evolution on 16 ETFs), 26 learned policies and N1 all lost to
    the rule.
  - Single stocks showed intraday signals of a few basis points, below their costs.
  - Most day traders lose. In Brazil, 97% of those who kept at it for more than 300 days lost
    money (Chague, De-Losso and Giovannetti). In Taiwan, fewer than 1% earned predictably
    after fees (Barber, Lee, Liu and Odean).
- **Why not reinforcement learning alone.**
  - Our orders do not move prices. Reinforcement learning is strongest where an action changes
    the world, as in games.
  - Here a model-free agent would have to learn both what the market will do and what to do
    about it from one very noisy reward. That needs more data than exists, so it memorises.
  - The better split is to learn the distribution of what comes next, and then to compute the
    decision from it exactly.
- **Leverage scales an edge; it does not create one.** On the rule's history:

  | Leverage | Sharpe ratio | Max drawdown |
  |---|---|---|
  | 1x | 1.31 | −10% |
  | 4x | 1.31 | −38.5% |
  | 10x | 1.31 | −80% |

## Decision

**1. KUTBIT is separate from the QQQ rule.**
- It has its own code and its own Guard file, `guard_kutbit.yaml`, which Efe writes by hand.
- It has two accounts of its own:
  - a second Alpaca paper account, which Efe opens, at up to 4x within the day, the most
    Alpaca allows on an account of 25,000 USD or more;
  - a virtual account in the shadow record at 10x, with margin simulated.
- It never shares the QQQ rule's account.

**2. Its instruments.**
- The stocks whose minutes are stored (each year's 50 most liquid S&P 500 members) and the 16
  ETFs of G1's gym.
- It trades within the day only and is flat at every close.

**3. Its six parts.**
1. **Inputs.**
   - The last one to two hours of minute bars.
   - Volume against the instrument's own normal for that time of day, the distance from
     VWAP, and the place in the day's range.
   - Strength against QQQ and against its sector ETF, the time of day and VIX.
   - The number of headlines in the last 30 minutes, from the Benzinga capture.
   - Everything is scaled by the instrument's own volatility.
2. **The forecaster.**
   - For 5, 15, 30 and 60 minutes ahead, it gives the distribution of the return (its
     quantiles) and the worst move against a position along the way. At 10x, the worst moment
     on the way is what closes an account, not the last price.
   - Five seeds are averaged, and when they disagree KUTBIT waits.
3. **Adaptation.**
   - One network for all instruments, each with an identity vector of its own.
   - Adaptive conformal calibration: each risk band widens when it fails to hold and narrows
     when it holds too often, which keeps its coverage right over time even when the market
     changes (Gibbs and Candès 2021).
   - The network is retrained each night on the latest data. Weights do not change within the
     session.
4. **The decision.**
   - Each minute, for each action (wait, long or short at the account's leverage, at each
     horizon), it computes the expected log growth of the account after costs. The chance of
     being closed out is part of that number.
   - It takes the best action, and waits unless that action clears Efe's threshold.
   - It does not flip on small changes, since with costs the best trading moves only part
     of the way to its aim (Gârleanu and Pedersen 2013).
5. **Fine-tuning for gain.**
   - The whole chain is tuned on the account's log growth in the simulator ("decision-focused
     learning"; Elmachtoub and Grigas 2022).
   - This is a variant, set beside the version without it.
6. **Hard rules in Guard, by hand.** A daily loss limit, a stop on each trade, a cap on trades a
   day, flat at the close, and no full leverage around headlines and earnings.

**4. Efe's settings.** Those of the decision layer can change without retraining.

| Setting | What it does |
|---|---|
| Leverage | 4x on paper, 10x in the virtual account |
| Threshold | The least expected gain after costs worth a trade |
| Risk penalty | How strongly KUTBIT avoids volatility |
| Stop on each trade | The most a trade may lose before it is closed |
| Daily loss limit | KUTBIT stops for the day after this loss |
| Trades a day | The most it may open in one session |

**5. Who writes what.**
- **Efe:**
  - the decision layer's rule;
  - the simulator's margin and stop rules (its broker);
  - `guard_kutbit.yaml`;
  - the networks and their training, which he runs himself.
- **Claude:**
  - the data, the inputs and the labels;
  - the simulator's skeleton and its tests;
  - the training harness and the placebos;
  - the pages.

**6. Evidence.**
- **Development** on 2016–2021, walk-forward: each year's model is trained on the years before
  it. At most ten variants are tried, under a choice rule written down first.
- **Placebos.** A variant must earn more than it does:
  - on random-walk prices of the same volatility, where nothing can be learned, so any gain
    there is a fault in the simulator;
  - on labels shuffled between instruments.
- **Further tests.**
  - Calibration: what the forecaster calls a 5% chance happens about 5% of the time.
  - Costs doubled.
- **Once, a second look at 2022–2026.**
- **The card.** It freezes the learning procedure, not the weights: architecture, inputs,
  nightly retraining, calibration and Efe's settings. Any change starts a new version.
- **The record** runs forward in both accounts and is judged against no trade by the betting
  test of decision 0020.
  - Leverage changes the scale, not the edge, so both accounts count as one card.
  - The card's share is the 0.625% of the 2.5% that N5 (1.25%) and N6 (0.625%) leave.

**7. Real money is not part of this decision.** KUTBIT trades on paper and in its virtual account;
`live_trading` stays `false`, and turning it on is Efe's alone.

## Consequences

- **KUTBIT will wait most of the time.** A trade must expect more than its costs and its risk
  penalty, under Efe's threshold.
  - A round trip costs about 2 basis points of the price, which at 10x is about 20 basis points of
    the account.
  - The predictable part of a stock's short moves has been a few basis points at most (step 5),
    so few moments clear that.
- **The live feed may be thin.** It is Alpaca's free IEX feed. For single stocks IEX's share of
  volume is small, so whether its minute bars suffice has to be checked, as decision 0021 checked
  them for QQQ.
- **Building it takes several pull requests,** each reviewed by Efe:
  1. the data and labels;
  2. the simulator;
  3. the decision layer;
  4. the training harness and placebos;
  5. the development rules, written before any training.
- **Hardware waits.** The design stays small enough to move to a board later. No board is
  chosen yet.
