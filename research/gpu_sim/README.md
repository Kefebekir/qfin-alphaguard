# The rule on the GPU

The first piece of a faster engine: the QQQ rule simulated on the GPU, every session and
thousands of variants at once, with the research's and the engine's fills (the next minute's
open, everything closed at the session's last price).

| Script | What it does |
|---|---|
| `prepare.py` | QQQ's sessions as arrays, one row a session and one column a minute (project environment) |
| `gym.py` | A batched environment for learned policies: every clean session of the 16 ETFs side by side, ten observations at each decision (known at the minute before), an exposure in [-2, 2], the engine's fills and costs. The rule as a policy reproduces qqq_momentum.py exactly (gross return to 6e-16, turnover to 2e-13, on all 2,693 days) |
| `test_gym.py` | Its guarantees on made-up markets: an observation never sees its decision's minute or later, no decision after a half day's close, costs, the cap, the close at the end. Eight deliberate breaks all fail a test. Run with `data/llmenv/bin/python -m pytest research/gpu_sim` (the project's CI has no GPU) |
| `sim.py` | The rule on the GPU (`data/llmenv`, PyTorch with CUDA): first checked against qqq_momentum.py in float64, then timed in float32 |

`results/01_check_and_speed.txt`, on Efe's RTX 5070 Laptop GPU (8 GB):

- **The same as the research, exactly.** All 2,693 sessions, the largest difference in a day's
  return 0.0, the turnover equal on every day.
- **About 6,500 times faster than the Python simulation.** 10,000 variants of all 2,693 sessions
  in about half a second (about 52 million sessions a second), against 7,982 sessions a second
  on the CPU: 0.9 hours for the same 10,000. Past about 10,000 variants the 8 GB fill and the
  run spills into system memory, so batches of 10,000 suit this GPU.

A variant here only widens or narrows the noise band: it measures speed. Choosing a band
among thousands on the same days would be overfitting; what the speed is for is training
models on many more simulated days than a CPU allows, and judging them forward.

Speed of the environment: 256 small networks (10-16-1) over all 42,998 clean sessions of the
16 ETFs in about 0.4 seconds; 1,024 at once exceed the 8 GB.

```bash
for t in QQQ SPY IWM DIA TLT GLD HYG XLB XLE XLF XLI XLK XLP XLU XLV XLY; do
  uv run python research/gpu_sim/prepare.py data/raw data/gpu_sim/$t.npz $t
done
data/llmenv/bin/python research/gpu_sim/sim.py data/gpu_sim/QQQ.npz data/intraday_pilot/qqq_days.parquet
data/llmenv/bin/python research/gpu_sim/gym.py data/gpu_sim data/intraday_pilot/qqq_days.parquet
data/llmenv/bin/python -m pytest research/gpu_sim
```
