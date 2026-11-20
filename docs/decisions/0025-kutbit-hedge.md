# 0025: KUTBIT's hedge is a decision, not a rule

- **Status:** accepted
- **Date:** 2026-11-17 (agreed with Efe the same day)
- **Changes:** decision 0024's section 1, where each position's market part was hedged with
  QQQ, and its section 5, where the decision maximised the expected log growth. Everything
  else in 0024 stands.

## Context

- **Efe's question.** With QQQ as the hedge, would KUTBIT pass over a very good opportunity
  because it "did not agree with QQQ"? No rule asks that. But the QQQ leg takes room under the
  gross cap: at 10x, a fully hedged long book with a beta of 0.72 can hold only 5.8x of stocks.
- **0024's risk table did not show that room.** It compared the same 10x of stocks with and
  without the hedge, so its hedged rows held about 7x of QQQ besides. Under one gross cap, the
  hedge also takes exposure away from the stocks.
- **The hedge study** (`research/kutbit/hedge_study.py`, `results_hedge.txt`) weighed the two:
  - 876,570 stock windows of 30 minutes in 2016-2021 (2022-2026 not read);
  - an edge planted in the move apart from QQQ, and none given to QQQ;
  - 2 bps a round trip for a stock, 0.5 bps for QQQ, both legs counting towards the cap;
  - three books: one stock; ten stocks long; five long and five short;
  - for each, the share of the cap and the share of the market part hedged that maximise the
    account's certainty equivalent at risk appetite γ (1 is Kelly, 2 about half Kelly).

## What the study found

1. **An edge must first pay its cost.** An edge of 0.05 in z, about 1.9 bps over 30 minutes,
   never trades; 0.07 is thin.
2. **Once an edge pays, the cap fills.** The risk of a 30-minute window is small beside a few
   bps of edge, so the size the Kelly criterion asks for is far above 10x.
3. **Then the hedge's room is often worth more than the risk it takes out.** Ten long at 10x,
   an edge of 0.1, the share of the market part hedged, by risk appetite:

   | γ | 1 | 2 | 4 | 8 |
   |---|---|---|---|---|
   | Hedged | none | none | half | three quarters |

   At 4x, hardly any hedge at any γ; a stronger edge hedges less.
4. **The book's own shorts are the best hedge.** Five long and five short leave a beta of
   0.10. QQQ is hardly needed, and the cap goes to positions with an edge: at 10x and an edge
   of 0.1, such a book hedged in full earned 1.9% a day, against 1.0% for ten long hedged in
   full.
5. **Jumpy markets hedge more.** Ten long at 10x with an edge of 0.1:
   - at γ 2, a quarter of the market part is hedged on the days whose QQQ volatility before the
     open was in the top third, and nearly none on the others;
   - at γ 4, about half on those days and up to a quarter on the others.
6. **An edge in z pays more where moves are large.** Its bps grow with the usual move, so at
   the open and on jumpy days an edge of 0.1 pays well; at midday it barely pays its cost.
7. **Leaving it unhedged costs its price on the bad days.** Ten long at 10x, an edge of 0.1 in
   every window of the day:

   | Hedged | Mean day | Std | Worst day | Days below −5% | Days below −10% |
   |---|---|---|---|---|---|
   | In full | +1.00% | 3.19% | −17.1% | 2.1% | 0.15% |
   | Half | +1.44% | 4.69% | −23.8% | 6.0% | 0.92% |
   | None | +2.21% | 8.96% | −37.5% | 15.1% | 6.2% |

   The means come from a planted edge in every window. Real edges are rarer and uncertain,
   and the bad days stay.

## Decision

**1. The hedge is chosen every minute, with the sizes.**
- The book's market part is what its longs and shorts leave after offsetting each other: its
  net beta times its exposure.
- The decision layer hedges a share `k` of it with QQQ, from none to all.
- It chooses `k` together with the positions' sizes, by the same computation over the bank of
  real paths, after costs, under Efe's caps and stop. Both legs count towards the cap.

**2. It never bets on the market.**
- QQQ is given no edge, since KUTBIT never forecasts its direction.
- `k` stays between none and all, so the hedge never adds market exposure or turns it round.
- QQQ is traded only as the hedge, and never on the QQQ rule's account.

**3. Efe's risk appetite, γ.** The decision maximises the account's certainty equivalent at
constant relative risk aversion γ. γ = 1 is 0024's expected log growth; 2 is about half Kelly.
Efe sets γ, by hand.

**4. Efe's limit on the market part left, M.**
- The market exposure left unhedged may not exceed M times the account's equity.
- Efe sets M by hand in `guard_kutbit.yaml`.
- M bounds the bad days of finding 7, whatever γ says.

**5. Shorts before QQQ.**
- The book's longs and shorts offset each other before any QQQ is traded.
- The computation finds this by itself, since an offset takes no room under the cap.

**6. What the paths must hold.**
- The bank of real paths keeps, beside each instrument's path, QQQ's path over the same
  minutes. The computation then sees what an unhedged part would have done, in calm and in
  crashes alike.
- QQQ's size today, not its direction, is estimated from its moves so far against its usual
  ones.

**7. Every choice is recorded beside its alternatives.**
- The shadow record keeps the account as decided.
- Beside it, it keeps the same positions hedged in full and not at all.
- Efe then sees, day by day, what the choice of hedge earned or cost.

**8. Reinforcement learning stays a variant,** as in 0024, here for choosing `k` against this
rule.

## Consequences

- 0024's section 1 now reads: the leverage is spread over several positions, long and short,
  each with a cap. The market part they leave together is hedged with QQQ by the share the
  decision layer chooses.
- 0024's section 5 now maximises the certainty equivalent at Efe's γ.
- The skill test is unchanged: the network still forecasts the move apart from QQQ.
- Next, in order:
  1. the bank of real paths, with QQQ's path beside each instrument's;
  2. the simulator, whose broker, margin and stop rules Efe writes, with both legs in the cap;
  3. the decision layer, Efe's, reading γ and M from `guard_kutbit.yaml`.
- Efe still has to set γ, M, each position's cap and the stop.
- Assumptions still to be replaced:
  - costs of 2 bps for a stock and 0.5 for QQQ, until the per-instrument cost table exists;
  - holds of 30 minutes;
  - edges that are planted, not forecast.
