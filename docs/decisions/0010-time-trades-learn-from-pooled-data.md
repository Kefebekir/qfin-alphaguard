# 0010: Time the plan's trades, don't trade the intraday signal; learn from pooled data

- **Status:** accepted
- **Date:** 2026-10-12

## Context

Efe asked whether the system should aim at each day's profit with a model
that corrects itself after every signal, separately for each stock, and with
leverage trades 1,000 to 10,000 times a day.

This was measured on the trading universe's minutes from 2016 to 2026: 7.5
million 5-minute slots, 93 stocks. A prediction was made every 5 minutes (66
decisions a day), and every model was judged only on 2020–2026. At each
decision point the strategy bought the predicted top fifth.

| Model, next 5 minutes | Rank IC | Gain a trade, before costs | After costs |
| --- | --- | --- | --- |
| Reversal rule: the last 5 minutes' rise falls back | 0.0225 | +0.31 bps | −2.69 bps |
| Fitted once on 2016–2019 | 0.0225 | +0.32 bps | −2.68 bps |
| One model, corrected after every outcome | 0.019–0.022 | +0.30 bps | −2.70 bps |
| One model per stock, corrected after every outcome | 0.015 | +0.20 bps | −2.80 bps |

Over the next 30 minutes the picture is the same. The model fitted once was
best (IC 0.015). The models corrected after every outcome were worse
(0.009–0.013), and one model per stock was worst (0.008). All lost 2.7–3 bps
a trade after costs.

A trade costs about 1.5 bps at Alpaca, and a round trip about 3 bps. This is
half the bid-ask spread, the slippage while the order goes out, and small
SEC and FINRA fees on sales; step 6 measured 1.3 bps against the decision
price. At 15:00 UTC on 9 October 2026 the spreads were NVDA 0.9 bps, AAPL
1.8, COST 3.5 and JNJ 5.8.

- **Intraday returns are predictable, by too little to trade.** The effect
  is statistically very strong (t = 30 at 5 minutes), mostly the reversal of
  the last minutes' moves, but it is worth about 0.3 bps a trade: a tenth of
  the cost. Part of it is the bounce between bid and ask, which an order
  that takes the price cannot earn.
- **Correcting after every outcome predicted worse.** Updating after each
  outcome follows noise, and a single stock gives too few examples. The
  relation is stable enough that more data beats faster updates.
- **More trades multiply the loss.** Rotating a 500 USD account every 5
  minutes costs about 10 USD a day for an expected gain of about 1 USD.
  Leverage multiplies the loss as much as any gain.
- **The rules allow frequent trading only in part.** FINRA's change to the
  pattern day trader rule took effect on 4 June 2026, but brokers have until
  20 October 2027 to adopt it (decision 0001). Even where it applies, the
  costs remain.

## Decision

- **The intraday model only times the trades the nightly plan asks for.**
  Decision 0001 stands: the model never trades its own signal. Its worth is
  measured as execution price against arrival price.
- **Models predict first and learn after (prequential), on all stocks
  together.** They are retrained often: weekly, later daily. A new model
  replaces the old one only through the champion/challenger rule. No model
  updates itself after each signal or for each stock.
- **More trades only where a trade's expected gain beats its cost,
  measured.** Guard's limits stay: 3 orders a stock a day, no same-day
  reversal, 10 orders a minute.
- **Reports show the days too:** daily profit and loss, the share of winning
  days, and the worst day. A strategy with a Sharpe ratio of 0.8 wins on
  only about 52% of days, so a single day says little.

## Consequences

- **The 30-minute intraday target serves timing.** The measured effect
  suggests it saves a fraction of a bp a trade, and a simple rule (wait after
  a sharp move) may capture most of it. Gate M2 compares the model with
  equal-spaced timing.
- **Returns have to come from the nightly forecast** of which stocks will do
  better than the others, not from trading during the day.
