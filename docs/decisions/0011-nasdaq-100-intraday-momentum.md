# 0011: Add intraday momentum on the Nasdaq-100 as a second strategy

- **Status:** accepted
- **Date:** 2026-10-12
- **Updated:** 2026-10-12, with QQQ's missing minutes filled (at the end)

## Context

Efe wants a model that reads the day ahead, trades a few times a day and uses
leverage for a large return. Decision 0010 showed that forecasting single
stocks minute by minute loses to costs. A different idea has a literature
behind it: index futures and ETFs trend within the day. The move up to late
afternoon predicts the last half hour (Gao, Han, Li and Zhou, 2018;
Baltussen, Da, Lammers and Martens, 2021). The explanation is that dealers in
options and leveraged ETFs hedge in the direction of the day's move.
Zarattini, Aziz and Barbon (2024, "Beat the Market") trade it on SPY:

- **The Noise Area.** Its band is the average absolute move from the open at
  each minute over the last 14 days, around the open and the previous close.
- **Trades.** At each HH:00 and HH:30 the strategy goes long above the band
  and short below it. The band or VWAP, whichever is tighter, is the trailing
  stop. Everything is closed by the end of the day.
- **Size.** min(4, 2% / the last 14 days' volatility).

The paper reports 19.6% a year and a Sharpe ratio of 1.33 for 2007 – April
2024.

**How it was tested:**

- **Data.** 1-minute bars from 2016 to October 2026, from Alpaca's SIP
  history. QQQ's bars match EODHD's from March 2024, the months both have.
- **Execution.** The paper's rules, filled at the next minute's open, at
  Alpaca's costs (0.001 USD a share plus 0.15 bps a side).
- **Out of sample.** May 2024 to October 2026 is out of sample: the paper's
  data end in April 2024.
- **Replication.**
  - The paper's example days give the same positions: long on 31 January
    2022 and short on 29 April 2022, both held to the close.
  - On 20 January 2022 the long closes at 13:30 instead of the paper's
    13:00. At 13:00 the price was 2 cents above our VWAP.
  - On SPY, 2016 – April 2024 gives 18.4% a year and a Sharpe ratio of 1.25
    at the paper's costs. The paper's sample is longer: it starts in 2007.

| Instrument, up to 4× | 2016–2026: a year, Sharpe | May 2024 – Oct 2026: a year, Sharpe |
| --- | --- | --- |
| **QQQ (Nasdaq-100)** | **21.4%, 1.40** | **18.1%, 1.16** |
| SPY (S&P 500) | 13.5%, 0.95 | −0.1%, 0.06 |
| DIA (Dow) | −1.4%, −0.04 | −2.9%, −0.17 |
| IWM (Russell 2000) | 0.6%, 0.11 | −10.0%, −0.65 |
| The trading universe's 50 stocks, 1× | −5.3%, −1.64 | −4.0%, −1.23 |

The simpler rule (the first half hour picks the side of the last) lost on SPY
and QQQ after costs: Sharpe −0.35 and −0.22.

**On QQQ, the result holds up under every check:**

- **Leverage cap:**

  | Cap | A year, worst drawdown | Sharpe | Out of sample |
  | --- | --- | --- | --- |
  | 1× | 12.3%, −7% | 1.51 | 1.20 |
  | 2× | 18.8%, −14% | 1.51 | 1.24 |
  | 4× | 21.4%, −22% | 1.40 | 1.16 |

  For the 2× cap, the 95% bootstrap interval of the Sharpe ratio is 1.03 to
  1.97 over 2016–2026, and 0.12 to 2.17 out of sample alone.
- **Parameters:** a lookback of 7–30 days, decisions every 15–60 minutes, a
  volatility target of 1–2.5% and caps of 1–3× give Sharpe ratios of 1.18 to
  1.51 over 2016–2026, and 0.74 to 1.24 out of sample.
- **Years:** 10 of 11 were positive; 2016 lost 6% at 4×, and 2025 made 6%.
- **Execution:**
  - Filling 1 or 2 minutes late gives Sharpe ratios of 1.39 and 1.28 (2×).
  - Closing at 15:59 instead of the close changes nothing.
  - EODHD's minutes give the same out-of-sample result (15.7% and 1.21
    against 16.2% and 1.24).
- **Its days:** only 26% of days win, but 68% of months do. The worst month
  lost 6.7% and the worst day 3.9% (2×).

Why only the Nasdaq-100 is not proven. The largest leveraged ETFs (TQQQ,
SQQQ) and heavy options trading are on it, which fits the hedging story.
Single stocks reverse within the day instead (0010).

## Decision

- **A second strategy.** Intraday momentum on the Nasdaq-100 runs beside the
  nightly plan, with the paper's rules, sized at min(cap, 2% / 14-day
  volatility), and flat at every close.
- **Cap of 2×** for this strategy. Its leverage is intraday and costs no
  interest, unlike the overnight leverage decision 0009 measured. The cap is
  set by hand in `guard.yaml` before any real money.
- **Proven in our engine before anything else.** It is built as its own
  `Strategy` in our engine and must reproduce these numbers in our backtest.
  Then it trades QQQ on Alpaca's paper account for two to three months.
- **Real money waits for two things:**
  - a broker and instrument a UK resident may trade with small capital
    through an API: QQQ at Alpaca if its UK accounts allow it (US ETFs are
    usually closed to UK retail investors under PRIIPs), otherwise a
    Nasdaq-100 CFD or spread bet with small sizes;
  - paper results inside the backtest's range.

## Consequences

- **Guard.** `no_same_day_reversal` and `max_trades_per_symbol_day: 3` forbid
  this strategy's round trips and reversals. Efe sets limits for it by hand,
  with the Guard work of step 9.
- **Expect less live than in the backtest.** SPY's edge vanished after
  publication, and QQQ's could too. If live results fall below the
  backtest's range, the strategy stops (Phase 3b's stop rules).
- **Machine learning** is tried here first as a filter or size for the day
  (volatility, dealer gamma), measured against the plain rules (Phase 2).
- **Data.** QQQ's minutes come from Alpaca's free SIP history, and EODHD's
  fill the days Alpaca lacks.
- **Not used:** single stocks, DIA, IWM and the first-half-hour rule, which
  failed the tests above.

## Update: QQQ's missing minutes filled

Alpaca's QQQ minutes lack four days, which step 7f fills from EODHD's:

- **2 and 3 May 2018.** Alpaca has only the opening minute.
- **2 and 22 February 2016.** Alpaca lacks 9 and 50 minutes.

The research above ran with these holes. It could not trade on 3 May
2018, and on 4 May it took the opening minute for the previous close.
With the days filled, 13 days' results change: the filled days and some
in the weeks after them, whose noise area uses them. The totals drop a
little:

| Cap | 2016–2026: a year, Sharpe | Before |
| --- | --- | --- |
| 1× | 12.0%, 1.48 | 12.3%, 1.51 |
| 2× | 18.3%, 1.47 | 18.8%, 1.51 |
| 4× | 20.8%, 1.37 | 21.4%, 1.40 |

- **Out of sample:** the results from May 2024 do not change.
- **Parameters:** the 13 settings give Sharpe ratios of 1.16 to 1.48
  (before: 1.18 to 1.51).
- **Years:** 10 of 11 are still positive. 2018 made 62% at 4× instead of
  70%.
- **Its days at 2×:** 26% of days and 67% of months win. The worst month
  lost 6.7% and the worst day 3.9%, as before.
- **Not run again:** the bootstrap and the execution checks. The Sharpe
  ratio moved by 0.04, well inside the bootstrap interval of 1.03 to 1.97.

The decision stands.
