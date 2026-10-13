# 0014: Guard in the backtest: what each limit measures and does

- **Status:** accepted
- **Date:** 2026-10-13
- **Refines:** decision 0013's schedule: on a day with sales, buys go a minute
  after them

## Context

Step 9 puts Guard between the strategy and the simulated broker. `guard.yaml`
gives each limit in one line. The backtest needs more: what each limit
measures, whether it drops an order or cuts it, and what Guard knows when it
judges. Two facts from the data shaped the answers.

- **More stocks than orders a minute.** The plan holds 9 to 37 stocks
  (median 19, on one day a month from 2016 to October 2026). So the first
  children at 10:30 are more than the 10 orders a minute Guard allows. Guard
  would drop the same stocks, the last in alphabetical order, at every child.
- **Sales pay for buys.** The band pays for a day's buys with its sales
  (decision 0006). If Guard counted an open sale's cash before the sale
  filled, a buy could fill while the sale did not, and the account would
  borrow. Without Guard, the account closed a day of 2025 with −1.15 USD of
  cash at 30,000 USD, and −0.24 USD at 5,000 USD.

## Decision

**What Guard knows.**
- The View at the moment of the order. The View now also says when each
  price was seen and what the account was worth at the last close.
- Its own record of the orders it let through that day. It does not count on
  the strategy to count them.

**Rules that drop an order**, checked in this order:

| Rule | The order is dropped when |
| --- | --- |
| `daily_loss_limit_pct` | the account is 2% or more below its value at the last close (the starting cash on the first day). Nothing more goes out that day, even after a recovery. It is not a halt: a halt stops a backtest for good, and the limit is a daily one. |
| `stale_data_sec` | the stock's last price is more than 15 seconds old; in the backtest, its last bar ended before the minute of the order. No price is stale too. |
| `price_collar_bps` | the limit price is more than 50 bps from the last price, on either side |
| `max_orders_per_min` | 10 orders went out in the last 60 seconds |
| `max_trades_per_symbol_day` | 3 orders in the stock went out today |
| `no_same_day_reversal` | an order in the stock went the other way today |

**Rules that cut an order**, applied together. The order goes out with the
smallest quantity they allow, and the event names the rule that set it. An
order cut to nothing is dropped.

| Rule | The most an order may be |
| --- | --- |
| `max_order_usd` | 2,000 USD at its limit price |
| `allow_short` (false) | for a sale: the shares held, less those already in open sales |
| `max_weight` | for a buy: what keeps the stock at 25% of the account. The stock counts at its last price; its open buys and this one count at their limits. |
| `max_gross_exposure` | for a buy: what keeps every position (at its last price) and every open buy (at its limit) within 1.0 times the account. An open sale counts only once it fills. |

**Buys count at their limits**, 0.5% over the price, so there is room for a
price that rises before the buy fills. A fully invested plan therefore keeps
its last 0.5% of buys in cash.

**`live_trading` is the live engine's to check** (Phase 3); a backtest never
reaches a broker.

**The strategy keeps within the limits.** Guard is the safety net and has
nothing to do on an ordinary day. EqualSpaced:
- sends at most `max_orders_per_min` children a minute, the earliest due
  first; the rest go on the next bar;
- on a day with sales, sends each buy a minute after them: at 10:31, 13:01
  and 15:31;
- sends a child only on a bar of its own stock, so its price is never stale.

## Evidence

2025 on the real 1-minute bars: the nightly plan with EqualSpaced and
Alpaca's costs.

| Account | Return without Guard | Return with Guard | What Guard did |
| --- | --- | --- | --- |
| 30,000 USD | +10.01% | +9.83% | dropped 128 orders by the loss limit, on 4 days (4, 7, 10 and 21 April); cut 16 buys to the cash, on 11 days |
| 5,000 USD | +10.00% | +9.82% | dropped 119 orders by the loss limit; cut 8 buys to the cash and dropped 1 |
| 500 USD | +10.56% | +10.22% | dropped 79 orders by the loss limit, on 3 days (4, 7 and 10 April) |

- **The other eight rules never fired**: stale data, the collar, orders a
  minute, orders a stock a day, reversals, order size, short sales and a
  stock's weight. The strategy keeps within them.
- **What Guard cost.** 0.18 points at 30,000 and at 5,000 USD. At 500 USD
  the loss limit was the only rule that acted, and it cost 0.34 points: after
  the April falls, the plan's trades waited a day. That is one year. Guard's
  limits are safety limits set by hand, not tuned for profit; the step 10
  backtest measures them over ten years.

## Consequences

- **The View has two new fields**, `price_times` and `last_close_value`. The
  live engine must fill them from the feed and from the broker's equity at
  the last close (Phase 3).
- **`EqualSpaced` takes Guard's `max_orders_per_min`.**
- **The loss limit in live trading.** In Phase 3, Efe decides whether it
  should halt the engine until he restarts it, as a halt does.
- **Buys under 1 USD.** Alpaca refuses a fractional buy under 1 USD, but the
  simulated broker fills it. A buy Guard cuts below that is therefore filled
  in the backtest. In 2025 this happened to 1 buy at 30,000 USD and 3 at
  5,000 USD, worth 0.41 to 0.98 USD each: too small to matter. It is noted
  for the simulated broker.
