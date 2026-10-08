# 0001: The intraday model times trades; it does not change targets

- **Status:** accepted
- **Date:** 2026-10-07

## Context

The intraday ML model could either tilt the day's target weights up and down
during the day, or only decide when to execute the change the nightly plan
asks for. Three constraints matter:

- Every order pays a fixed minimum commission (IBKR: 0.35–1 USD) or a
  percentage FX fee on other brokers, so many small trades are expensive.
- US intraday trading rules are in transition: FINRA removed the 25,000 USD
  pattern day trader minimum on 4 June 2026, but brokers have until
  20 October 2027 to adopt the new rules, and the old rule can apply until then.
- The model's contribution has to be measurable on its own.

## Decision

The intraday layer only times execution. It splits the gap between current
and planned positions into at most `max_children` orders of at least
`min_trade_usd` each, sends one when the model's score in the trade's
direction passes a threshold that falls during the day, and sends whatever is
left before the deadline. It never trades a stock in the opposite direction on
the same day.

## Consequences

- Few, large orders keep commissions low.
- No same-day round trips, so the design does not depend on how pattern day
  trading rules are applied.
- The model is judged by one number: execution price against arrival price,
  compared with equal-spaced orders that use no model.
- An intraday tilt strategy stays out of scope; it may come back as a separate
  experiment (roadmap Phase 6).
