# 0019: Guard for an intraday strategy: a position can always be closed

- **Status:** accepted
- **Date:** 2026-10-14
- **Refines:** decision 0014. It changes which orders each limit applies
  to, lets `max_weight` go above 1, lets the daily loss limit be switched
  off, and adds a second limits file, `guard_qqq.yaml`, for the QQQ
  strategy's account.

## Context

Phase 2's step 4 puts the QQQ strategy (decision 0011, `momentum.py`)
behind Guard. It holds one ETF, long or short, at up to 2× its account,
turns within the day, and is flat a minute before every close. `guard.yaml`
was written for the nightly plan: long only, no leverage, 3 orders a stock
a day and no reversals. The strategy will later take its decisions from a
model (decision 0018), so the limits have to hold whatever decides.

`research/intraday_pilot/guard_needs.py` measures the strategy on the
stored QQQ minutes from 2016 to 9 October 2026, from 30,000 USD, at Alpaca's
fees and QQQ's measured half spread of 0.3 bps.

- **Under `guard.yaml`, the strategy cannot work.** It earns 0.6% a year
  instead of 16.1%: Guard drops 3,942 short sales, 2,601 orders past 3 a
  day and 691 reversals, and cuts 1,860 orders to 2,000 USD.
- **Guard could keep a position from being closed.** With the numbers
  loosened as far as Guard then allowed (shorts, 2× gross, `max_weight` 1,
  no caps on orders, the 2% daily loss limit kept), the daily loss and gross
  limits dropped the strategy's closing orders too. On `main`'s code
  (386143b), 412 sessions ended with a position still held, the account
  closed below half its start on 2 June 2017, and on 29 January 2018 it was
  worth −705 USD and the run stopped.
- **Three more gaps.**
  - A short position counted for nothing in `max_weight` and
    `max_gross_exposure`.
  - A buy that covers a short counted as new exposure.
  - `max_weight` could not go above 1, so one ETF at 2× could not be
    allowed.
- **What the industry does.** FIA's *Best Practices for Automated Trading
  Risk Controls and System Safeguards* (July 2024):
  - a kill switch "may allow for risk-reducing orders while preventing
    risk-increasing orders" (section 1.5);
  - a maximum intraday position is long or short, and "both current
    positions and working orders should be evaluated" against it (section
    1.2).
- **Why limits still matter when a model decides.** SEC Rule 15c3-5 makes
  a broker with market access check every order against pre-set limits.
  Knight Capital lost more than 400 million USD in 45 minutes in August
  2012, when old code sent orders that its controls did not stop; the SEC
  found those controls broke the rule.

## Decision

**1. An order that only reduces a position faces only the checks on the
order itself.**
- Those checks are `price_collar_bps` and `max_orders_per_min`. The order
  still counts towards later orders' limits.
- **Reduces** means it takes the position towards zero but not past it,
  once the open orders on its side have filled. Two closing orders therefore
  cannot together sell more than is held.
- **No other limit can stop it:**
  - not the daily loss limit, which stops new risk, not its removal;
  - not `max_order_usd`, `max_weight` or `max_gross_exposure`. The order is
    bounded by the position, and the position was bounded when it was built.
    A cut would leave the rest held overnight;
  - not `stale_data_sec`, `max_trades_per_symbol_day` or
    `no_same_day_reversal`.
- **Why the two checks stay:** a closing order with a wrong price, or a loop
  of them, is still an error.
- **For a model this is the main guarantee:** the limits bound what it can
  add, and it can always get out.

**2. An order through zero is dropped.**
- A sale of more than is held, or a buy of more than is short, is dropped,
  as Alpaca requires: close first. The strategy already does this in two
  steps.
- With `allow_short: false`, a sale is still cut to the shares held, as
  before.

**3. Shorts count as exposure.** A stock's exposure, long or short, for
`max_weight` and `max_gross_exposure` is:
- its position at its last price;
- plus each open buy at its limit;
- plus the part of each open sale beyond the shares held, at the last
  price.

An open order that reduces the position adds nothing.

**4. `max_weight` may be above 1.** It must still be at most
`max_gross_exposure`.

**5. `daily_loss_limit_pct` may be `false`: no daily loss limit.**
- Only `false` written out switches it off. An empty value is still an error.
- Where there is a limit, it now stops only new risk.
- **Efe's choice for QQQ is none.** His reason: if the strategy sees how
  to make up the loss, it should carry on. For example, it is long, sees
  the market turning, sells and goes short, and the gain from the fall
  makes up the loss. That decision is the model's to make.

**6. The QQQ strategy's account has its own file, `guard_qqq.yaml`.**
- `guard.yaml` stays the nightly plan's, unchanged. Each strategy trades in
  its own account; Alpaca allows several paper accounts.
- The file is named for the account, not the rules, because a model will
  trade the same account later.

| Limit | Value | Why |
| --- | --- | --- |
| `allow_short` | true | the strategy goes short below the noise band |
| `max_weight`, `max_gross_exposure` | 2.1 | the strategy's 2×. Its largest position asked for, at the limit price against the account at the decision, was 2.039× |
| `max_order_usd` | 70,000 | about 2.3× a 30,000 USD account; raised by hand as the account grows |
| `price_collar_bps` | 50 | as `guard.yaml`; the strategy sets its limits at this collar |
| `max_orders_per_min` | 10 | as `guard.yaml`; the strategy sent at most 1 a minute |
| `max_trades_per_symbol_day` | 12 | after 12 orders in a day, no new risk. The strategy sent at most 10 a day (99th percentile 8) |
| `no_same_day_reversal` | false | the strategy turns by design |
| `daily_loss_limit_pct` | false | Efe's choice, above |
| `stale_data_sec` | 15 | as `guard.yaml` |
| `live_trading` | false | paper only; a test pins it, and 2.1× at most |

## Evidence

The QQQ strategy over 2016 – 9 October 2026, with the new Guard
(`guard_needs.py`):

| Run | A year | Sharpe | Guard's events | Nights held |
| --- | --- | --- | --- | --- |
| No Guard | +16.1% | 1.32 | — | 0 |
| `guard.yaml` (the nightly plan's) | +0.6% | 0.98 | 9,094 | 0 |
| `guard_qqq.yaml` | +14.2% | 1.31 | 2,089 orders cut to 70,000 USD | 0 |
| `guard_qqq.yaml`, `max_order_usd` lifted | +16.1% | 1.32 | none; every fill the same | 0 |
| the same, daily loss limit 5%, 4% or 3% | +16.1% | 1.32 | none; every fill the same | 0 |
| the same, daily loss limit 2% | +16.0% | 1.32 | 18 orders dropped, on 6 days | 0 |

- **The order cap acts only because the account grows.** The account grows
  from 30,000 to 148,724 USD. From March 2018 it is past the size the
  70,000 USD cap was set for, and the cap cuts orders. That is its job: it
  is raised by hand as the account grows. With it lifted, the other limits
  change nothing.
- **What a daily loss limit would have done.** The account was 4.36% below
  its last close at its worst decision. A limit of 3% or more would never
  have acted. At 2%, it would have stopped new risk on 6 days, and the run
  would end 1% lower.
- **The nightly plan under the new Guard** (`qfin backtest`, 5 January 2016
  – 9 October 2026). Old Guard (`main`, whose run gives the same
  fingerprints as its stored report) → new Guard:

  | Plan | A year | Sharpe | Worst day | Orders the loss limit drops |
  | --- | --- | --- | --- | --- |
  | 500 USD | +10.71% → +10.69% | 0.86 → 0.87 | −7.7% → −6.6% | 509 → 232 |
  | 5,000 USD | +10.81% → +10.80% | 0.86 → 0.87 | −7.7% → −6.5% | 774 → 404 |
  | 30,000 USD | +10.84% → +10.84% | 0.87 → 0.88 | −7.7% → −6.5% | 828 → 434 |

  - After a 2% loss, the plan's sales now go out, and only its buys wait.
  - Volatility falls from 12.85% to 12.66%, and the worst day is about a
    point better.
  - The fingerprints change with the log.

The tests break each part of the new rules one at a time: all 32 such
changes fail a test.

## Consequences

- **Efe can add a daily loss limit to `guard_qqq.yaml` at any time,** by
  hand. 5% would never have acted on the rules, and would stop a model that
  misbehaves from opening new positions after a 5% loss on the day.
- **`max_order_usd` follows the account.** Before paper trading, the paper
  account's starting cash and this limit have to match.
- **The research runners use `guard_qqq.yaml`.** The day book replays
  behind the real Guard, with `max_order_usd` lifted for the ten-year run.
  `engine_check.py` still runs without Guard: it checks the strategy
  against the research, not the limits.
- **Phase 3, the live engine:**
  - A cancel is not instant there. Until the broker confirms it, Guard still
    counts the order as open. The live engine has to wait for that
    confirmation before it replaces a closing order, or Guard could judge
    the new order as going through zero.
  - The C++ Guard and the FPGA gate read the same files, with the same
    rules.
