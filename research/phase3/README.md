# Phase 3: paper trading, the questions to answer before it starts

| Script | Question | Result |
|---|---|---|
| `iex_vs_sip.py` → `results/01_iex_vs_sip.txt` | Can the rule decide on Alpaca's free live feed, IEX's trades alone, when the research used the consolidated tape (SIP)? Decision 0008's plan. | **Yes.** On 2021 – 9 October 2026 (Alpaca keeps IEX's minutes back to 2021), IEX has a trade in 92.4% of the session's minutes, and the rule deciding on IEX holds the same position as on SIP at 98.4% of 17,136 decisions (every decision the same on 85.1% of days). Filled at SIP's prices, it earns 17.3% a year (Sharpe 1.44) against 16.4% (1.36) deciding on SIP: +0.36 bps a day, t +0.78, daily results correlated 0.973. The 99 USD a month SIP plan is not needed. |

| `live_core_check.py` → `results/02_live_core_check.txt` | Does the live engine's core (`live/engine.py`) decide as the backtest's engine does, when bars reach it 0.2 s late and Alpaca answers each order late and in parts? | **Yes, exactly.** The QQQ strategy behind Guard, on IEX's minutes of 2021 – 9 October 2026, 1,448 sessions: the same 18,557 decision notes, 2,985 orders and 707 Guard events, and the books at each close agree within 6e-11 USD. With fills slowed to 0.2% of each bar's volume, so the strategy cancels open orders 2,817 times and the core must wait for each cancel before its replacement goes: again the same 18,557 notes, 5,178 orders and 736 Guard events. Every session, written to its journal as JSON and replayed through a fresh core, gives the same orders and books. |
| `live_core_check.py ... morning` → `results/03_morning_warm_up.txt` | The live engine starts afresh each morning: does a strategy warmed on the 20 sessions before (`live/warmup.py`) decide as the backtest's, which learned every session as it went? | **Yes, exactly.** Built afresh and warmed each of the 1,448 mornings, the strategy makes the same 18,557 decisions, the same orders and Guard events, with normal fills and with slowed ones. Alpaca's IEX history for the 20 sessions before 12 October 2026 is the stored minutes exactly (7,586), and Monday 19 October's warm-up took 0.9 s to fetch and 0.04 s to play. |
| `report_demo.py` → `results/04_report_demo.txt` | What will `qfin live-report` say of a session? Tried before the first paper session on Friday 16 October 2026, played from Alpaca's IEX history through the live core, each order filled at the next minute's open. | Short from 10:00 to the close, +884 USD (+2.95%). The session replays from its journal; backtests on IEX's history and on SIP make the same 13 decisions of 13 and earn within 7 USD of it; the 389 bars are Alpaca's exactly. |

```bash
uv run --env-file .env python research/phase3/iex_vs_sip.py data/raw
uv run python research/phase3/live_core_check.py data/raw 2021-01-04 2026-10-09 guard_qqq.yaml 0.1
uv run python research/phase3/live_core_check.py data/raw 2021-01-04 2026-10-09 guard_qqq.yaml 0.002
uv run python research/phase3/live_core_check.py data/raw 2021-01-04 2026-10-09 guard_qqq.yaml 0.1 morning
uv run --env-file .env python research/phase3/report_demo.py 2026-10-16 data/live-demo guard_qqq.yaml
uv run --env-file .env qfin live-report --root data/live-demo --day 2026-10-16
```

IEX's minutes stay in the git-ignored `data/raw/intraday_iex/`.
