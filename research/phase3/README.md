# Phase 3: paper trading, the questions to answer before it starts

| Script | Question | Result |
|---|---|---|
| `iex_vs_sip.py` → `results/01_iex_vs_sip.txt` | Can the rule decide on Alpaca's free live feed, IEX's trades alone, when the research used the consolidated tape (SIP)? Decision 0008's plan. | **Yes.** On 2021 – 9 October 2026 (Alpaca keeps IEX's minutes back to 2021), IEX has a trade in 92.4% of the session's minutes, and the rule deciding on IEX holds the same position as on SIP at 98.4% of 17,136 decisions (every decision the same on 85.1% of days). Filled at SIP's prices, it earns 17.3% a year (Sharpe 1.44) against 16.4% (1.36) deciding on SIP: +0.36 bps a day, t +0.78, daily results correlated 0.973. The 99 USD a month SIP plan is not needed. |

| `live_core_check.py` → `results/02_live_core_check.txt` | Does the live engine's core (`live/engine.py`) decide as the backtest's engine does, when bars reach it 0.2 s late and Alpaca answers each order late and in parts? | **Yes, exactly.** The QQQ strategy behind Guard, on IEX's minutes of 2021 – 9 October 2026, 1,448 sessions: the same 18,557 decision notes, 2,985 orders and 707 Guard events, and the books at each close agree within 6e-11 USD. With fills slowed to 0.2% of each bar's volume, so the strategy cancels open orders 2,817 times and the core must wait for each cancel before its replacement goes: again the same 18,557 notes, 5,178 orders and 736 Guard events. Every session, written to its journal as JSON and replayed through a fresh core, gives the same orders and books. |

```bash
uv run --env-file .env python research/phase3/iex_vs_sip.py data/raw
uv run python research/phase3/live_core_check.py data/raw 2021-01-04 2026-10-09 guard_qqq.yaml 0.1
uv run python research/phase3/live_core_check.py data/raw 2021-01-04 2026-10-09 guard_qqq.yaml 0.002
```

IEX's minutes stay in the git-ignored `data/raw/intraday_iex/`.
