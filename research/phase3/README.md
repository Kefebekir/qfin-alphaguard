# Phase 3: paper trading, the questions to answer before it starts

| Script | Question | Result |
|---|---|---|
| `iex_vs_sip.py` → `results/01_iex_vs_sip.txt` | Can the rule decide on Alpaca's free live feed, IEX's trades alone, when the research used the consolidated tape (SIP)? Decision 0008's plan. | **Yes.** On 2021 – 9 October 2026 (Alpaca keeps IEX's minutes back to 2021), IEX has a trade in 92.4% of the session's minutes, and the rule deciding on IEX holds the same position as on SIP at 98.4% of 17,136 decisions (every decision the same on 85.1% of days). Filled at SIP's prices, it earns 17.3% a year (Sharpe 1.44) against 16.4% (1.36) deciding on SIP: +0.36 bps a day, t +0.78, daily results correlated 0.973. The 99 USD a month SIP plan is not needed. |

```bash
uv run --env-file .env python research/phase3/iex_vs_sip.py data/raw
```

IEX's minutes stay in the git-ignored `data/raw/intraday_iex/`.
