# Architecture

QFin-AlphaGuard runs three loops at three speeds. Research code never sits in
the live path: the nightly job hands the engine a plan, the engine decides only
*when* to trade, and Guard decides whether an order may go out at all.

```
NIGHTLY (Python, once a day)
  data ─► features ─► Alpha AI (μ, σ) ─► optimiser ─► daily_plan.json
                                                          │ before the open
EVERY 1-MINUTE BAR (engine)                               ▼
  IBKR 5 s bars ─► bar builder ─► features + ML score ─► strategy
                                                          │ order intent
EVERY ORDER (engine + risk)                               ▼
  guard.yaml ─► Guard ─► OMS ─► (optional FPGA risk gate) ─► IBKR
                           │
                           └─► fills, PnL, every decision ─► log store ─► back to the nightly job
```

The engine starts in Python (Phase 3) and moves to C++ (Phase 4).

## Components

| Component | Runs | Input → output | Technology |
| --- | --- | --- | --- |
| Data layer | nightly + live stream | raw data → clean, point-in-time bars | Polars, Parquet, DuckDB |
| Features | nightly (batch) + every bar (incremental) | bars → feature vector | Python; C++ copy in Phase 4 |
| Alpha AI | trained weekly; predicts nightly and every bar | features → μ, σ, short-term score | scikit-learn, LightGBM, ONNX |
| Optimiser | nightly | μ, Σ, constraints → target weights | CVXPY, Ledoit-Wolf covariance |
| Strategy | every bar | plan + score + position → order intent | inside the engine |
| Guard | every order + continuous monitoring | intent → approve, reduce or reject; halt | rules from `guard.yaml` |
| Engine (OMS) | continuously, event-driven | approved order → broker; fill → position, PnL | Python (ib_async), then C++ |
| Log store | event-driven + nightly | every decision → JSONL, compacted to Parquet | DuckDB |
| FPGA | lab benchmark; optional live risk gate | market message or order → decision or approval | SystemVerilog, Vivado, Artix-7 |
| Quantum (research) | offline | same portfolio problem → comparison with classical | Qiskit, `research/quantum` |

The likely source of profit is the ML forecasts and cost control. The C++ and
FPGA work is measured engineering, valued on its own; it does not make a
retail-broker strategy more profitable.

## Data flow

| # | From → to | Content | Frequency | Format |
| --- | --- | --- | --- | --- |
| 1 | Providers → data layer | daily and intraday bars, macro series | nightly | EODHD, IBKR history, FRED → Parquet |
| 2 | Data layer → features | clean point-in-time bars | nightly | DuckDB → Polars |
| 3 | Features → Alpha AI | feature matrix + target | train weekly, predict nightly | Parquet |
| 4 | Alpha AI → optimiser | μ and σ forecasts, model version | nightly | Parquet |
| 5 | Optimiser → engine | `daily_plan.json`: target weights, each stock's volatility, execution settings | once, before the open | JSON, versioned |
| 6 | `guard.yaml` → Guard | risk limits | only by hand, through a commit | YAML |
| 7 | IBKR → engine | live prices as 5-second bars | continuous | TWS API via IB Gateway |
| 8 | Inside the engine | 1-minute bar → features → score → intent | every bar | in memory |
| 9 | Strategy → Guard → IBKR | order intent → approved order | per order | in memory, TWS API |
| 10 | IBKR → engine | order status, fills | per event | TWS API |
| 11 | Engine → log store → nightly job | decisions, fills, slippage, PnL | per event; read nightly | JSONL → Parquet |

IBKR real-time bars are 5 seconds long; the engine builds each 1-minute bar
from 12 of them.

## Daily plan

Written by the nightly job, read by the engine at start-up. The types are in
`src/qfin_alphaguard/events.py` (`DailyPlan`, `ExecutionSettings`).

Until the forecasts of Phase 2 exist, `qfin plan` (`plan.py`) builds it from
the daily bars before the day: the minimum-variance portfolio of the year's
trading universe, with Ledoit-Wolf covariance over the last 252 sessions and no
stock above Guard's `max_weight`. It holds at most
`floor(capital / (max_children · min_trade_usd))` stocks, so that each can be
built in `max_children` orders: 10 at 30,000 USD. Each stock's own daily
volatility, the standard deviation of its returns over the same 252 sessions,
goes with the plan for the band (decision 0006).

```json
{
  "date": "2027-04-12",
  "model_version": "alpha-v1.3",
  "target_weights": {"AAPL": 0.15, "MSFT": 0.15, "NVDA": 0.10, "AMZN": 0.10,
                     "GOOGL": 0.10, "JPM": 0.15, "XOM": 0.10, "JNJ": 0.15},
  "volatility": {"AAPL": 0.016, "MSFT": 0.014, "NVDA": 0.032, "AMZN": 0.019,
                 "GOOGL": 0.018, "JPM": 0.015, "XOM": 0.015, "JNJ": 0.011},
  "execution": {
    "band_pct": 2.0,
    "max_children": 3,
    "min_trade_usd": 1000,
    "theta0": 0.6,
    "min_gap_min": 30,
    "start_after_open_min": 5,
    "deadline_before_close_min": 30
  }
}
```

## Execution logic

The intraday layer never changes the target; it only chooses when to close
the gap between the current position and the plan. Fewer, larger orders keep
commissions low and avoid same-day reversals, and the model's value becomes
measurable: execution price against arrival price, compared with an
equal-spaced schedule that uses no model.

For each stock *i*, with plan weight w, account value E, price p, shares held
n, ML score s in [-1, 1]:

```
Δ_i   = round(w_i · E / p_i) − n_i                    shares to trade today
N_i   = min(max_children, floor(|Δ_i| · p_i / min_trade_usd))   number of child orders
u_i,t = s_i,t · sign(Δ_i)                             score in the trade's direction
θ(t)  = theta0 · (1 − 2 · (t − t0) / (T − t0))        threshold, falls from theta0 to −theta0
send a child order when u_i,t ≥ θ(t)
```

1. **At the open:** compute Δ. Skip the stock if its weight gap is within its
   own band, `k` days of its daily volatility times its weight (decision 0006),
   or the trade is below `min_trade_usd`.
2. **Split** the trade into at most `max_children` child orders of at least
   `min_trade_usd` each.
3. **Every bar:** send a child when u ≥ θ(t), at least `min_gap_min` minutes
   after the previous one in the same stock; otherwise wait.
4. **Deadline:** `deadline_before_close_min` minutes before the close, send
   what is left. If the model fails its health check, fall back to
   equal-spaced children.
5. **Guard** approves, reduces or rejects each child. No order in the opposite
   direction in the same stock on the same day.
6. **Order:** a limit order within `price_collar_bps` of the last price; an
   unfilled order is reconsidered on the next bar.

## Backtest event loop

`backtest/engine.py` replays bars in time order. Bars that end at the same
moment form a slice, and every slice is handled in the same order:

1. The simulated broker fills waiting orders from the slice's bars. Each of
   those orders was sent at the end of an earlier slice.
2. The portfolio books the fills and the strategy hears of them.
3. The strategy sees the complete slice and answers with order intents and
   cancellations, dated with the slice's end.
4. Each intent passes Guard, which approves it, reduces its quantity, rejects
   it or halts trading; what passes becomes an order sent at that moment.

So an order decided after bar t can fill no earlier than bar t+1. The engine is
also the referee: it stops the run if a fill is dated before its order, lies
outside its bar, is priced where the bar never traded, breaks its limit, or
exceeds what is left of the order, and if Guard changes anything but the
quantity. At each close it cancels open orders and logs the portfolio's value. At each
open, before the strategy hears of the session, it applies that day's splits
(held shares and the last price follow the ratio; a fraction of a share is paid
in cash) and dividends (cash per share held), derived from the daily bars in
`data/corporate.py`.
The strategy sees only a read-only `View` (time, session, cash, positions, last
prices, open orders), so the same `Strategy` class can run live in Phase 3.

## Risk file

Only Efe changes this file, by hand and through a commit. The Python Guard,
the C++ Guard and the FPGA gate all read the same file: `guard.yaml` at the
repository root, loaded by `src/qfin_alphaguard/guard/config.py`. Every
setting is required, an unknown or repeated setting is an error, and every
value is checked for type and range before the engine may trade.

```yaml
allow_short: false             # no short selling
max_weight: 0.25               # one stock is at most 25% of the portfolio
max_gross_exposure: 1.0        # no leverage
max_order_usd: 2000            # largest single order
price_collar_bps: 50           # limit price within 0.5% of the last price
max_orders_per_min: 10         # stops order storms
max_trades_per_symbol_day: 3   # at most 3 orders per stock per day
no_same_day_reversal: true     # no opposite-direction order in a stock on the same day
daily_loss_limit_pct: 2.0      # no new orders after a 2% intraday loss
stale_data_sec: 15             # no trading on data older than 15 seconds
live_trading: false            # set to true by hand for a live account
```

If a daily plan breaks `guard.yaml`, the engine rejects it and only allows
orders that reduce positions that day.

## Trading day (US/Eastern, regular session)

1. **Overnight:** download and validate data, build features, retrain weekly
   with a champion/challenger rule, optimise, write `daily_plan.json`. If any
   step fails, no plan is written.
2. **Before the open:** the engine loads the plan and `guard.yaml`, checks the
   IB Gateway connection, data freshness and positions. With no valid plan it
   starts in reduce-only mode.
3. **First 5 minutes:** no trading (wide spreads at the open).
4. **Every minute until 30 minutes before the close:** build the bar, update
   features, score, decide (send a child or wait), Guard, order. Every step is
   logged.
5. **All day:** process fills; watch the daily loss limit, data freshness and
   the connection. A triggered rule stops new orders; the kill switch cancels
   all open orders.
6. **Near the close:** cancel open orders. Positions are carried overnight.
7. **After the close:** reconcile with the broker, write the daily report
   (PnL, slippage, deviation from backtest) and compact the logs.

## The FPGA's two roles

A retail broker round trip takes tens to hundreds of milliseconds, so the FPGA
does not speed up live trading. It has two other jobs:

1. **Lab pipeline (the main FPGA project).** A PC replays a NASDAQ ITCH 5.0
   sample file over UDP. The FPGA parses Ethernet/UDP and ITCH, keeps the top
   of book, computes streaming features (returns, EMA, variance) and a risk
   check, and returns a decision with a hardware timestamp. The same file runs
   through a C++ golden model; outputs are compared bit for bit and latency,
   throughput and jitter are measured.
2. **Live risk gate (optional).** The C++ engine sends each order to the board,
   which approves or rejects it against `guard.yaml`. The point is an
   independent failure domain, not speed; a physical kill switch connects here.

Board: Digilent Arty A7-100T (101,440 logic cells, 240 DSP slices, 4,860 Kbit
block RAM, 10/100 Mb/s Ethernet). Calculations are fixed-point, with the error
against a float reference measured and reported.

## Latency budget (rough estimates, to be replaced by measurements)

| Step | Typical range |
| --- | --- |
| FPGA computation | 50 ns – 1 µs |
| C++ bar-to-decision | 1 – 50 µs |
| PC ↔ FPGA round trip | 20 – 500 µs |
| Python bar-to-decision | 1 – 10 ms |
| Broker path, order to acknowledgement | 30 – 300 ms |
