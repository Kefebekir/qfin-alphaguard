"""The QQQ rule on the GPU: every session, and thousands of variants of the rule, at once.

The same rule and fills as research/intraday_pilot/qqq_momentum.py (and so as momentum.py and
the engine): the noise band from the 14 sessions before, VWAP, decisions every 30 minutes from
the open, each filled at the next minute's open (or the last price if that minute has no bar),
everything closed at the session's last price. Sessions run side by side; within a session
the 12 decisions run in turn, as the rule's position depends on the one before.

A variant here scales the noise band's width (band = 1 is the rule). It exists to measure
speed, not to choose a band: choosing among thousands of variants on the same days is the
overfitting the research warns of.

Usage: sim.py QQQ_NPZ QQQ_DAYS_PARQUET   (data/llmenv/bin/python, on the GPU)
"""

import sys
import time

import numpy as np
import polars as pl
import torch

LOOKBACK, STEP, DECISIONS = 14, 30, 12


def prepare(path: str, device: str, dtype: torch.dtype) -> dict:
    """The arrays of prepare_qqq.py, as tensors with what every variant shares."""
    z = np.load(path)
    t = {
        k: torch.tensor(z[k], device=device, dtype=dtype)
        for k in ("close", "fill_open", "pv", "vol", "first_open", "last_close")
    }
    length = torch.tensor(z["length"], device=device)
    days, width = t["close"].shape
    minute = torch.arange(width, device=device)
    have = torch.isfinite(t["close"])
    # the last bar's index at each minute (-1 before the first), to carry prices forward
    last = torch.where(
        have,
        minute.expand(days, width),
        torch.full_like(minute, -1).expand(days, width),
    )
    last = torch.cummax(last, dim=1).values
    price = torch.gather(torch.nan_to_num(t["close"]), 1, last.clamp(min=0))
    price = torch.where(last >= 0, price, torch.nan)
    inside = minute[None, :] < length[:, None]
    move = torch.where(inside, (price / t["first_open"][:, None] - 1).abs(), torch.nan)
    # VWAP: typical price times volume over volume, carried forward; NaN before the first bar
    vwap = torch.cumsum(t["pv"], 1) / torch.clamp(torch.cumsum(t["vol"], 1), min=1)
    vwap = torch.where(last >= 0, vwap, torch.nan)
    # sigma at each minute: the mean move over the 14 sessions before that reach the minute
    shifted = torch.stack([torch.roll(move, k, dims=0) for k in range(1, LOOKBACK + 1)])
    reach = torch.stack([torch.roll(inside, k, dims=0) for k in range(1, LOOKBACK + 1)])
    total = torch.where(reach, shifted, 0.0).sum(0)
    count = reach.sum(0)
    sigma = total / count
    # daily volatility: the standard deviation of the 14 log returns before the session
    r = torch.log(t["last_close"][1:] / t["last_close"][:-1])
    r = torch.cat([torch.full((1,), torch.nan, device=device, dtype=dtype), r])
    windows = torch.stack([torch.roll(r, k) for k in range(1, LOOKBACK + 1)])
    vol14 = windows.std(0, unbiased=True)
    # what the decisions need: the minute before each decision, and each fill
    steps = torch.arange(1, DECISIONS + 1, device=device) * STEP  # 30, 60, ..., 360
    before = steps - 1
    decide = steps[None, :] < length[:, None]
    nxt = t["fill_open"][:, steps.clamp(max=width - 1)]
    p_before = price[:, before]
    fill = torch.where(decide & torch.isfinite(nxt), nxt, p_before)
    end = torch.gather(price, 1, (length - 1)[:, None]).squeeze(1)
    prev_close = torch.roll(t["last_close"], 1)
    first = t["first_open"]
    return {
        "price": p_before,
        "sigma": sigma[:, before],
        "vwap": vwap[:, before],
        "fill": fill,
        "decide": decide,
        "end": end,
        "hi": torch.maximum(first, prev_close)[:, None],
        "lo": torch.minimum(first, prev_close)[:, None],
        "vol14": vol14,
        "days": z["days"],
    }


def run(s: dict, band: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The gross return and the turnover of a 1x position, [variants, sessions]."""
    k = band[:, None, None]
    upper = s["hi"] * (1 + k * s["sigma"])
    lower = s["lo"] * (1 - k * s["sigma"])
    variants, days = band.shape[0], s["price"].shape[0]
    zeros = torch.zeros(variants, days, device=band.device, dtype=s["price"].dtype)
    held, entry, gross, traded = (
        zeros.clone(),
        zeros.clone(),
        zeros.clone(),
        zeros.clone(),
    )
    pos = zeros.clone()
    for j in range(DECISIONS):
        p, up, lo, vw = (
            s["price"][:, j],
            upper[:, :, j],
            lower[:, :, j],
            s["vwap"][:, j],
        )
        flat_next = torch.where(p > up, 1.0, torch.where(p < lo, -1.0, 0.0))
        long_next = torch.where(
            p < torch.maximum(up, vw), torch.where(p < lo, -1.0, 0.0), 1.0
        )
        short_next = torch.where(
            p > torch.minimum(lo, vw), torch.where(p > up, 1.0, 0.0), -1.0
        )
        want = torch.where(
            pos == 0, flat_next, torch.where(pos == 1, long_next, short_next)
        )
        want = torch.where(
            s["decide"][:, j], want, pos
        )  # no decision past the session's end
        pos = want
        f = s["fill"][:, j]
        change = (want != held) & s["decide"][:, j]
        gross = gross + torch.where(change & (held != 0), held * (f / entry - 1), 0.0)
        traded = traded + torch.where(change, (want - held).abs(), 0.0)
        entry = torch.where(change, f.expand_as(entry), entry)
        held = torch.where(change, want, held)
    gross = gross + torch.where(held != 0, held * (s["end"] / entry - 1), 0.0)
    traded = traded + held.abs()
    return gross, traded


if __name__ == "__main__":
    npz, reference = sys.argv[1], sys.argv[2]
    # 1. Correctness, in float64: the rule (band 1) against qqq_momentum.py's stored days.
    s = prepare(npz, "cuda", torch.float64)
    gross, traded = run(s, torch.tensor([1.0], device="cuda", dtype=torch.float64))
    ours = pl.DataFrame(
        {
            "day": s["days"][LOOKBACK + 1 :],
            "gross": gross[0, LOOKBACK + 1 :].cpu().numpy(),
            "traded": traded[0, LOOKBACK + 1 :].cpu().numpy(),
        }
    ).with_columns(pl.col("day").str.to_date())
    ref = pl.read_parquet(reference).select("day", "gross_1x", "traded_1x")
    both = ours.join(ref, on="day", how="inner")
    diff = np.abs(both["gross"].to_numpy() - both["gross_1x"].to_numpy())
    print(
        f"correctness: {both.height} sessions against qqq_momentum.py; largest difference in a "
        f"day's return {diff.max():.2e}; days differing by more than 1e-9: {(diff > 1e-9).sum()}; "
        f"turnover equal on {(np.abs(both['traded'] - both['traded_1x']) < 1e-9).sum()} days"
    )
    # 2. Speed, in float32: many variants of the band at once.
    s32 = prepare(npz, "cuda", torch.float32)
    for variants in (1, 100, 1_000, 10_000, 50_000):
        band = torch.linspace(0.5, 1.5, variants, device="cuda")
        torch.cuda.synchronize()
        start = time.perf_counter()
        gross, _ = run(s32, band)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        sessions = variants * (gross.shape[1] - LOOKBACK - 1)
        print(
            f"{variants:>6} variants x {gross.shape[1] - LOOKBACK - 1} sessions: {seconds:.3f} s, "
            f"{sessions / seconds:,.0f} sessions a second, peak memory "
            f"{torch.cuda.max_memory_allocated() / 1e9:.1f} GB"
        )
