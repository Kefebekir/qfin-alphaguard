# QFin-AlphaGuard

A trading system for US equities, built in the open. A rule trades the
Nasdaq-100's trend within the day on a paper account, every order passes explicit
risk rules before it reaches the broker, and machine-learning models run beside it
in shadow, each frozen before its record starts.

The goal is a model of my own that works forward, on days it has never seen, and
that runs on its own hardware; a backtest that looks good is not the goal. Real
money comes last, and only if the system beats simple alternatives after costs and
tax. This page says what works today and what does not.

## Status

| Stage | State |
| --- | --- |
| Data, an event-driven backtester with costs, and the risk rules (Guard) | done |
| The QQQ intraday rule in the engine, every decision logged and shown day by day | done |
| Paper trading on Alpaca, on AWS every weekday | running; the gate is 8 clean weeks |
| Machine-learning challengers in shadow | N5 and N6 are recording forward |
| Hardware: the network on an FPGA, then the whole engine on an Arm-based board | planned, see [below](#hardware-the-next-experiment) |
| C++ execution engine | planned |
| Small real-money trading, with scaling and stop rules | planned; off until I turn it on by hand |

The steps, gates and current position are in [docs/ROADMAP.md](docs/ROADMAP.md).

## How it works

1. **The rule decides.** At 10:00 and every half hour to 15:30 New York time,
   the strategy compares QQQ with its noise band and VWAP and decides to be long,
   short or flat
   ([decision 0011](docs/decisions/0011-nasdaq-100-intraday-momentum.md)). Its
   size is capped at 2×, and it is flat a minute before the close. Each decision
   is logged with what the strategy saw and its reason in words.
2. **Guard checks every order.** It is a rule-based referee, not a model:
   position and order size limits, a price collar, and orders a minute and a day.
   An order that only reduces a position faces only the price collar and the
   orders a minute, so a position can always be closed. The limits are in
   `guard_qqq.yaml` and change only by hand.
3. **The live engine keeps its own record**
   ([decision 0021](docs/decisions/0021-phase-3-the-live-engine.md)). Every bar,
   order and order event is on disk before the engine acts on it, and a restart
   replays the day through the same rules. Its orders and positions are reconciled
   with Alpaca's. After the close, a report replays the session and compares it
   with a backtest of that day.
4. **Models run beside it, in shadow.** A challenger trades only in its own
   virtual account until it proves better forward.

The backtester and the live engine share the same event types, strategy interface
and Guard. Research code never sits in the live path. A decision may use only data
available at that moment, and a backtest fills an order no earlier than the next
bar.

## What the evidence says

### The rule

The QQQ rule is the one edge that held up in our walk-forward tests from 2018 to
October 2026, after costs: 21.1% a year at a 2× cap with a Sharpe ratio of 1.61,
and 1.26 after its paper's data ended in April 2024.
Everything else below was measured against it, or against no trade.

### Paper trading

**The record is still thin.** Of the first 12 sessions (19 October – 3 November
2026), the engine stopped before the open on 9, because IEX's minute bars from
before the open reached its core. No position was at risk. The fix went live on 4
November, and on each of the three sessions since, all 13 decisions matched a
backtest of that day. Those three days made +413.70 USD on 30,000; they show the
engine works and say nothing yet about what the rule earns.

### The models

The news reader asks whether a language model reading 2.09 million Benzinga
headlines (2016 – October 2026) can help. Each headline is read by a model that
knows nothing after the year before it. I train the models myself.

| Test | What it tried | Result |
| --- | --- | --- |
| N1 | The morning's headlines size the rule's day | not the expected result: −1.52 bps a day against the rule |
| N2 | A member's next day, read by ChronoBERT | not the expected result: −2.22 bps a day |
| N3 | The same, read by PIT-4B (4 billion parameters) | not the expected result: +4.24, t +1.18 |
| N4 | N3 with less noise | not the expected result: +6.53, t +1.86, under its bar of t ≥ 2.50 and resting on 2026 |
| N5 | N4's variant 2, frozen and scored forward | recording since 23 October 2026 |
| N6 | My own network on PIT-4B's frozen readings | second look +6.11, t +1.99; recording since 30 October 2026 |

Results are basis points a day after costs on 2022 – 9 October 2026; N6's second look
runs through 2026 as well. N5's and N6's records are days old and prove nothing yet. They are the test that
matters: every year of the news has been seen by now, so only new days can judge
them. An edge with a Sharpe ratio of 2 is found in about two years, and one of 1
takes eight or more.

### What did not work

- **The nightly minimum-variance plan loses to both baselines:** 10.7–10.8% a year
  against 15.3% for SPY and 17.5% for the same stocks at equal weight, and with a
  lower Sharpe ratio. Costs are not the reason
  ([decision 0015](docs/decisions/0015-the-backtest-and-its-first-report.md)). The
  job still runs each night; the plan is no longer the focus.
- **Nothing beat the rule out of sample** in the intraday research: 26 learned
  policies, dealer gamma, calendar and news effects, regimes, breadth and the
  overnight return ([ledger](research/step5/LEDGER.md)).
- **A network trained by evolution** lost 7.88 bps a day to the rule on 2022 –
  October 2026 (t −2.92).
- **The rule on Nikkei 225 futures** made −5.4% a year after one tick a fill
  (t −1.82).
- **How much a company's SEC reports change** from a year before: a first look at
  10-Ks and 10-Qs on 2016–2021 made +14.2 bps a month after costs, t +0.61, which
  cannot be told from luck. 2022–2026 stay unread as a clean holdout.

## How a model earns its place

- **The test is written down first.** Each rule is committed before any result, and
  every variant goes into a [ledger](research/news_reader/LEDGER.md) before it runs.
- **Few tries, on old years.** At most ten variants on 2018–2021, and each must beat
  placebos such as the same headlines dealt out at random within the day.
- **Frozen before the record.** A [card](challengers/) names the commit, the training
  dates and the hash of the weights before the first scored day.
- **Judged by a bet.** Each day's result is a bet that stays valid however often it is
  read; a t-test read every evening would promote a model with no edge about a quarter
  of the time ([decision 0020](docs/decisions/0020-the-shadow-record.md)).

## Hardware: the next experiment

Not started. The aim is to try a small device that runs a model of my own without a
PC: a quantized neural network in an FPGA, beside an Arm processor running the rest
of the engine.

1. **The network on an FPGA, with a PC as its host.** The small part of the network
   that makes the decision, quantized from its first training run, goes on a Digilent
   Arty A7-100T. Its output must equal the Python model's bit for bit.
2. **The whole engine on an Arm-based board.** The design then moves to an **AMD Kria
   KV260**, which has a quad-core Arm Cortex-A53 and FPGA fabric on one chip (Zynq
   UltraScale+). The Arm side runs Linux and the engine: the connection to the broker,
   TLS, Guard and the journal. The FPGA side runs the network.

What it will and will not show:

- **It does not make trading faster.** A round trip through a retail broker takes
  30–300 ms, far more than an FPGA can save. The value is engineering: a model that
  runs deterministically, at a measured latency, in its own failure domain.
- **The language model stays off the board.** PIT-4B has 4 billion parameters. It reads
  the headlines on a larger machine, and the board receives numbers.
- **Hardware does not make a model good.** Whether a model earns anything is for its
  forward record to say, in Python. The board stays on the paper account.

The design is in
[docs/research/news-to-the-fpga-network.md](docs/research/news-to-the-fpga-network.md).

## What I care about

- **No look-ahead.** The tests cover it, and news and filings are read only as they
  stood on the day.
- **Honest reporting.** If a method loses to the naive baseline, this page says so. I
  ran into this before on a
  [quantum hardware experiment](https://github.com/Kefebekir/quantum-hardware-noise-comparison)
  where error mitigation underperformed the unmitigated baseline, and reporting that
  was more useful than hiding it.
- **Evidence in layers.** Long out-of-sample tests first, then paper trading to show
  the live system behaves like the backtest, then small real money that grows only
  while live results stay inside the backtest's expected range.

## Known limitations

- **Paper fills are not real fills.** Measured slippage on the paper account may
  flatter a real account.
- **The live feed is IEX, part of the market.** On history the rule made the same
  decision on IEX's minutes as on the consolidated tape 98.4% of the time.
- **Survivorship bias, mostly removed.** The data covers every stock that was in the
  S&P 500 on each day since 2015, including those later bought or delisted. The
  membership history is kept by one person rather than S&P, and stocks outside the
  index are not covered.
- **1-minute bars only for the trading universe.** Daily bars cover every S&P 500
  member; 1-minute bars, from 2016, cover each year's 50 most liquid members.
- **Adjusted prices are not point-in-time.** Returns are right; dividing by the
  `adjustment` column gives the price as traded.
- **Real money needs another route.** UK retail accounts cannot buy US-domiciled ETFs,
  so QQQ from the UK will need a Nasdaq-100 CFD or something similar
  ([decision 0011](docs/decisions/0011-nasdaq-100-intraday-momentum.md)).

## Getting started

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Kefebekir/qfin-alphaguard.git
cd qfin-alphaguard
uv sync
uv run pytest

# run the data pipeline on generated data (no network needed)
uv run qfin ingest --synthetic

# real data: needs EODHD_API_KEY in .env
uv run --env-file .env qfin ingest

# 1-minute bars: needs Alpaca's keys in .env too
uv run --env-file .env qfin intraday
uv run --env-file .env qfin etf-minutes QQQ

# the nightly plan on the stored minutes, against SPY and equal weight
uv run qfin backtest

# today's session of the QQQ strategy, on Alpaca's paper account only
uv run --env-file .env qfin live
```

## Deployment

Each job is an **ECS Fargate** task started by **EventBridge Scheduler**, from one
Docker image in **Amazon ECR**.

| Task | When | What it does |
| --- | --- | --- |
| `qfin ingest` | 06:45 London, Tuesday to Saturday | Downloads and validates the daily prices |
| `qfin live` | 09:15 New York, weekdays | Runs the session on the paper account |
| `qfin news-capture` | 15:31 New York, weekdays | Stores the news as it stands, for the forward records |
| `qfin live-report` | 16:30 New York, weekdays | Writes and e-mails the session's report |

Data goes to **S3**, with versioning. Keys are in **SSM Parameter Store** and reach
the tasks as secrets, logs go to **CloudWatch Logs**, and alerts are sent by **SNS**.
Each component has its own IAM role with the minimum permissions it needs. The
policies, task definitions and the live engine's setup are in `infra/aws/`.

## Tech

Python 3.12, Polars, Parquet, DuckDB, NumPy, scikit-learn, CVXPY, PyTorch for the
research models, pytest, ruff, Docker, the Alpaca API, AWS (ECR, ECS Fargate,
EventBridge Scheduler, S3, SSM, SNS), GitHub Actions.

Planned: C++20, and SystemVerilog and hls4ml on an AMD Artix-7 and the Kria KV260.

More detail: [architecture](docs/ARCHITECTURE.md), [roadmap](docs/ROADMAP.md) and
[decision records](docs/decisions/).

## License

MIT
