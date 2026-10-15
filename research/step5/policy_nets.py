"""Intraday pilot: trading policies learned end to end, against the noise-area rule.

Every model here outputs the position itself at each of the rule's decision
moments, a number from -1 (short) to 1 (long) times the day's size
min(2, 2% / 14-day volatility), and is trained to maximise the Sharpe ratio
of its daily results after costs: the loss of Deep Momentum Networks (Lim,
Zohren and Roberts, 2019), and the "direct reinforcement" of Moody and
Saffell (2001). A decision fills at the open of its minute and every day
ends flat, as in the engine (policy_data.py).

Models, each from what is known at the end of the minute before a decision:

- rule: the noise-area rule itself (decision 0011), not learned;
- tuned_rule: the rule with its band widened or narrowed by the factor that
  gave the best Sharpe ratio on the training years;
- knn: the 200 most similar past decisions vote on the direction;
- linear: tanh of a weighted sum of the features, the rule among them;
- mlp: a small neural network, from scratch;
- anchored: the rule plus a learned correction, tanh(a * rule + mlp(x)),
  starting at the rule (a = 2, the correction 0);
- gate: the rule's position scaled by a learned factor, from 0 up to the
  2x cap, starting at 1;
- day_size: the rule with one learned factor for the whole day, set from
  what is known at 10:00;
- gru: a recurrent network reads the day so far in 5-minute steps; its
  state at a decision corrects the rule, as anchored;
- attention: the same with causal self-attention over the steps instead of
  the recurrent network (as the Momentum Transformer of Wood, Giegerich,
  Roberts and Zohren, 2022), each step seeing only those before it.

Each learned model is fitted on QQQ alone, on all 16 ETFs pooled (as Deep
Momentum Networks pooled 88 futures), and on the four broad index ETFs with
the largest option markets (QQQ, SPY, IWM, DIA), where the mechanism behind
intraday momentum, dealers hedging their options, applies. It is judged on
QQQ only. The third pool was added after seeing the rule's result on each
ETF: it works on QQQ, SPY and XLK and loses on most others.

Walk-forward by year: for test year Y, fitted on the years before Y - 1,
stopped early on Y - 1, an average of 5 seeds. Fixed in advance, before
any result: the models, their sizes, the learning rate, the batch, the
patience and the costs. Models are compared on 2019-2022; 2023-2026 is
looked at once, at the end (HOLDOUT=1).

Usage: policy_nets.py DATA_DIR OUT_DIR [HOLDOUT]
"""

import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.neighbors import KNeighborsRegressor
from torch import nn

DATA, OUT = Path(sys.argv[1]), Path(sys.argv[2])
HOLDOUT = len(sys.argv) > 3 and sys.argv[3] == "1"
TEST_YEARS = list(range(2023, 2027)) if HOLDOUT else list(range(2019, 2023))
CAP = 2.0
FEES = 0.0000206 / 2  # the SEC fee on sales: half of all turnover, on average
HALF_SPREAD = {"QQQ": 0.3e-4, "SPY": 0.3e-4}  # measured for QQQ; SPY's is as tight
OTHER_SPREAD = 1.0e-4  # the other ETFs, not measured: 1 bp, as the stocks'
FEATURES = [
    "rule",
    "above_upper",
    "below_lower",
    "from_vwap",
    "from_open",
    "last_30",
    "gap",
    "time",
    "minutes_left",
    "volume_so_far",
    "vol_so_far",
    "sigma",
    "yesterday",
]
SLOTS, STEPS, STEP_INPUTS = 12, 78, 5
SEEDS, BATCH, EPOCHS, PATIENCE, LR, DECAY = 5, 256, 200, 15, 1e-3, 1e-4
MIN_STEPS = 20  # minibatches an epoch at least, drawn with replacement if few days
if os.environ.get("SMOKE"):
    SEEDS, EPOCHS, TEST_YEARS = 1, 2, TEST_YEARS[:1]
HIDDEN, DROPOUT = 32, 0.1


# --- the data, one row a ticker-day ---------------------------------------------


def load():
    d = pl.read_parquet(DATA / "decisions.parquet").sort("ticker", "day", "t")
    paths = np.load(DATA / "paths.npz")
    days = d.select("ticker", "day").unique(maintain_order=True)
    index = {key: i for i, key in enumerate(days.iter_rows())}
    n = days.height
    x = np.zeros((n, SLOTS, len(FEATURES)), np.float32)
    ret = np.zeros((n, SLOTS), np.float32)
    mask = np.zeros((n, SLOTS), np.float32)
    rule = np.zeros((n, SLOTS), np.float32)
    step = np.zeros((n, SLOTS), np.int64)  # the last 5-minute step a decision sees
    size = np.zeros(n, np.float32)
    cost = np.zeros(n, np.float32)
    hi = np.zeros(n, np.float32)  # max(open, last close) / open
    lo = np.zeros(n, np.float32)
    rows = d.select("ticker", "day", "t", "ret", "size", "vol14", *FEATURES)
    slot = np.zeros(n, np.int64)
    for r in rows.iter_rows(named=True):
        i = index[(r["ticker"], r["day"])]
        k = slot[i]
        slot[i] += 1
        x[i, k] = [r[f] for f in FEATURES]
        ret[i, k], mask[i, k], rule[i, k] = r["ret"], 1.0, r["rule"]
        step[i, k] = r["t"] // 5 - 1
        size[i] = r["size"]
        cost[i] = HALF_SPREAD.get(r["ticker"], OTHER_SPREAD) + FEES
        ratio = 1 / (1 + r["gap"] * r["vol14"])  # last close / open
        hi[i], lo[i] = max(1.0, ratio), min(1.0, ratio)
    path = np.zeros((n, STEPS, STEP_INPUTS), np.float32)
    tickers = days["ticker"].to_numpy()
    isodays = [day.isoformat() for day in days["day"]]
    for ticker in np.unique(tickers):
        # An npz unpacks an array at every access: read each one once.
        stored, stored_days = paths[f"{ticker}_paths"], paths[f"{ticker}_days"]
        on = {day: s for s, day in enumerate(stored_days)}  # ISO dates
        for i in np.where(tickers == ticker)[0]:
            path[i] = stored[on[isodays[i]]]
    return {
        "ticker": days["ticker"].to_numpy(),
        "day": days["day"].to_numpy(),
        "year": days["day"].dt.year().to_numpy(),
        "x": x,
        "ret": ret,
        "mask": mask,
        "rule": rule,
        "step": step,
        "size": size,
        "cost": cost,
        "hi": hi,
        "lo": lo,
        "path": path,
    }


def day_pnl(pos, ret, mask, size, cost):
    """Each day's result: the positions' returns less the cost of every change,
    the last change being the close. Works on numpy arrays and on tensors."""
    pos = pos * mask  # a day's decisions come first; the empty slots hold 0
    lib = torch if isinstance(pos, torch.Tensor) else np
    zero = lib.zeros_like(pos[:, :1])
    path = lib.concatenate([zero, pos, zero], 1)  # flat before and after
    traded = lib.abs(path[:, 1:] - path[:, :-1]).sum(1)
    return size * ((pos * ret).sum(1) - cost * traded)


def sharpe(pnl):
    return float(pnl.mean() / pnl.std(ddof=1) * math.sqrt(252)) if len(pnl) > 1 else 0.0


# --- rules that are not neural networks -----------------------------------------


def rule_positions(x, mask, hi, lo, widen):
    """The noise-area rule again, with its band `widen` times as wide."""
    above = x[:, :, 1] - (widen - 1) * hi[:, None]
    below = x[:, :, 2] + (widen - 1) * lo[:, None]
    vwap = x[:, :, 3]
    pos = np.zeros(mask.shape, np.float32)
    for i in range(mask.shape[0]):
        p = 0.0
        for k in range(int(mask[i].sum())):
            if p == 0:
                p = 1.0 if above[i, k] > 0 else (-1.0 if below[i, k] < 0 else 0.0)
            elif p == 1 and (above[i, k] < 0 or vwap[i, k] < 0):
                p = -1.0 if below[i, k] < 0 else 0.0
            elif p == -1 and (below[i, k] > 0 or vwap[i, k] > 0):
                p = 1.0 if above[i, k] > 0 else 0.0
            pos[i, k] = p
    return pos


def tuned_rule(data, train, test):
    best, best_widen = -1e9, 1.0
    for widen in (0.6, 0.8, 1.0, 1.2, 1.5):
        pos = rule_positions(
            data["x"][train],
            data["mask"][train],
            data["hi"][train],
            data["lo"][train],
            widen,
        )
        s = sharpe(
            day_pnl(pos, *(data[k][train] for k in ("ret", "mask", "size", "cost")))
        )
        if s > best:
            best, best_widen = s, widen
    pos = rule_positions(
        data["x"][test],
        data["mask"][test],
        data["hi"][test],
        data["lo"][test],
        best_widen,
    )
    return pos, {"widen": best_widen}


def knn(data, train, test):
    m = data["mask"][train] > 0
    xs = data["x"][train][m]
    ys = (data["ret"][train] / 1.0)[m] * data["size"][train][:, None].repeat(SLOTS, 1)[
        m
    ]
    mean, sd = xs.mean(0), xs.std(0) + 1e-9
    model = KNeighborsRegressor(n_neighbors=200, n_jobs=2)
    model.fit((xs - mean) / sd, ys)
    mt = data["mask"][test] > 0
    pred = np.zeros(data["mask"][test].shape, np.float32)
    pred[mt] = model.predict((data["x"][test][mt] - mean) / sd)
    fitted = model.predict((xs[::20] - mean) / sd)
    strong = np.abs(pred) >= np.median(np.abs(fitted))
    return np.where(strong, np.sign(pred), 0.0).astype(np.float32), {}


# --- neural policies -------------------------------------------------------------


class Policy(nn.Module):
    """Position in [-1, 1] at each decision (the gate up to the cap)."""

    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        f = len(FEATURES)
        if kind == "linear":
            self.head = nn.Linear(f, 1)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
            with torch.no_grad():
                self.head.weight[0, 0] = 2.0  # starts at the rule
            return
        extra = 0
        if kind in ("gru", "attention"):
            extra = HIDDEN
            self.embed = nn.Linear(STEP_INPUTS + 1, HIDDEN)
            if kind == "gru":
                self.seq = nn.GRU(HIDDEN, HIDDEN, batch_first=True)
            else:
                self.position = nn.Parameter(torch.zeros(STEPS, HIDDEN))
                self.seq = nn.TransformerEncoderLayer(
                    HIDDEN, 2, 2 * HIDDEN, DROPOUT, batch_first=True
                )
                causal = torch.triu(torch.ones(STEPS, STEPS, dtype=torch.bool), 1)
                self.register_buffer("causal", causal)
        self.mlp = nn.Sequential(
            nn.Linear(f + extra, HIDDEN),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, HIDDEN),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, 1),
        )
        if kind != "mlp":  # every other model starts at the rule
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)
        self.a = nn.Parameter(torch.tensor(2.0))

    def forward(self, x, rule, path, step, room):
        if self.kind == "linear":
            return torch.tanh(self.head(x).squeeze(-1))
        inputs = x
        if self.kind in ("gru", "attention"):
            clock = torch.linspace(0, 1, STEPS).expand(path.shape[0], STEPS)[..., None]
            h = self.embed(torch.cat([path, clock], -1))
            if self.kind == "gru":
                h, _ = self.seq(h)
            else:
                h = self.seq(h + self.position, src_mask=self.causal, is_causal=True)
            seen = torch.gather(h, 1, step[..., None].expand(-1, -1, HIDDEN))
            inputs = torch.cat([x, seen], -1)
        out = self.mlp(inputs).squeeze(-1)
        if self.kind == "mlp":
            return torch.tanh(out)
        if self.kind == "gate":
            # 1 at the start: softplus(c) = 1; never past the 2x cap
            factor = nn.functional.softplus(out + math.log(math.e - 1))
            return rule * torch.minimum(factor, room[:, None])
        if self.kind == "day_size":
            factor = nn.functional.softplus(out[:, :1] + math.log(math.e - 1))
            return rule * torch.minimum(factor, room[:, None])
        return torch.tanh(self.a * rule + out)  # anchored, gru, attention


def tensors(data, rows, mean, sd):
    x = np.clip((data["x"][rows] - mean) / sd, -5, 5)
    x[:, :, 0] = data["x"][rows][:, :, 0]  # the rule stays -1, 0 or 1
    path = np.clip(data["path"][rows], -10, 10)
    return {
        "x": torch.from_numpy(x.astype(np.float32)),
        "rule": torch.from_numpy(data["rule"][rows]),
        "path": torch.from_numpy(path),
        "step": torch.from_numpy(np.maximum(data["step"][rows], 0)),
        "room": torch.from_numpy(CAP / data["size"][rows]),
        "ret": torch.from_numpy(data["ret"][rows]),
        "mask": torch.from_numpy(data["mask"][rows]),
        "size": torch.from_numpy(data["size"][rows]),
        "cost": torch.from_numpy(data["cost"][rows]),
    }


def positions(model, t):
    with torch.no_grad():
        model.eval()
        return model(t["x"], t["rule"], t["path"], t["step"], t["room"]) * t["mask"]


def fit_one(kind, data, train, valid, test, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = data["mask"][train] > 0
    mean = data["x"][train][m].mean(0)
    sd = data["x"][train][m].std(0) + 1e-9
    tr, va, te = (tensors(data, rows, mean, sd) for rows in (train, valid, test))
    model = Policy(kind)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=DECAY)
    n = len(train)
    best, best_state, waited, epochs = -1e9, None, 0, 0
    pnl_args = ("ret", "mask", "size", "cost")
    rng = np.random.default_rng(seed)
    for epoch in range(EPOCHS):
        model.train()
        steps = max(n // BATCH, MIN_STEPS)
        if n // BATCH >= MIN_STEPS:
            order = rng.permutation(n)[: steps * BATCH].reshape(steps, BATCH)
        else:
            order = rng.integers(0, n, (steps, min(BATCH, n)))
        for batch in order:
            idx = torch.from_numpy(batch)
            part = {k: v[idx] for k, v in tr.items()}
            pos = model(
                part["x"], part["rule"], part["path"], part["step"], part["room"]
            )
            pnl = day_pnl(pos, *(part[k] for k in pnl_args))
            loss = -pnl.mean() / (pnl.std() + 1e-6)
            opt.zero_grad()
            loss.backward()
            opt.step()
        val = day_pnl(positions(model, va), *(va[k] for k in pnl_args)).numpy()
        score = sharpe(val)
        if score > best + 1e-4:
            best, waited, epochs = score, 0, epoch + 1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return positions(model, te).numpy(), epochs, best


def neural(kind, data, train, valid, test):
    runs = [fit_one(kind, data, train, valid, test, seed) for seed in range(SEEDS)]
    pos = np.mean([r[0] for r in runs], 0)
    return pos, {
        "epochs": [r[1] for r in runs],
        "valid_sharpe": round(float(np.mean([r[2] for r in runs])), 2),
    }


# --- the walk-forward -------------------------------------------------------------

MODELS = {
    "rule": None,
    "tuned_rule": tuned_rule,
    "knn": knn,
    **{
        kind: kind
        for kind in (
            "linear",
            "mlp",
            "anchored",
            "gate",
            "day_size",
            "gru",
            "attention",
        )
    },
}


def job(args):
    name, pool, year = args  # pool: "QQQ", "index" or "all"
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    data = DATASET
    qqq = data["ticker"] == "QQQ"
    scope = {
        "QQQ": qqq,
        "index": np.isin(data["ticker"], ["QQQ", "SPY", "IWM", "DIA"]),
        "all": np.ones_like(qqq),
    }[pool]
    train = np.where(scope & (data["year"] < year - 1))[0]
    valid = np.where(scope & (data["year"] == year - 1))[0]
    test = np.where(qqq & (data["year"] == year))[0]
    started = time.time()
    if name == "rule":
        pos, info = data["rule"][test], {}
    elif callable(MODELS[name]):
        pos, info = MODELS[name](data, np.concatenate([train, valid]), test)
    else:
        pos, info = neural(name, data, train, valid, test)
    pnl = day_pnl(pos, *(data[k][test] for k in ("ret", "mask", "size", "cost")))
    held = data["mask"][test] > 0
    info["off_rule"] = round(float(np.abs(pos - data["rule"][test])[held].mean()), 3)
    info["seconds"] = round(time.time() - started)
    return name, pool, year, data["day"][test], pnl, pos, info


DATASET = load()

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [("rule", "QQQ", y) for y in TEST_YEARS]
    for name in MODELS:
        if name == "rule":
            continue
        for pool in ("QQQ", "index", "all"):
            jobs += [(name, pool, y) for y in TEST_YEARS]
    jobs.sort(key=lambda j: j[0] in ("gru", "attention"), reverse=True)  # long first
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    results = []
    with ProcessPoolExecutor(max_workers=10) as executor:
        for r in executor.map(job, jobs):
            results.append(r)
            name, pool, year, *_, info = r
            print(f"done {name} {pool} {year}: {info}", flush=True)
    table = pl.concat(
        [
            pl.DataFrame({"day": days, "pnl": pnl.astype(np.float64)}).with_columns(
                pl.lit(name).alias("model"),
                pl.lit(pool).alias("pooled"),
                pl.lit(year).alias("year"),
                pl.col("day").cast(pl.Date),
            )
            for name, pool, year, days, pnl, pos, info in results
        ]
    ).select("model", "pooled", "year", "day", "pnl")
    tag = "holdout" if HOLDOUT else "selection"
    table.write_parquet(OUT / f"daily_{tag}.parquet")
    rule = table.filter(pl.col("model") == "rule").select(
        "day", pl.col("pnl").alias("r")
    )
    print(
        f"\nQQQ, {TEST_YEARS[0]}-{TEST_YEARS[-1]}, after costs, each against the rule:"
    )
    for (name, pool), g in table.group_by("model", "pooled", maintain_order=True):
        g = g.sort("day").join(rule, on="day")
        r, base = g["pnl"].to_numpy(), g["r"].to_numpy()
        diff = r - base
        t = (
            diff.mean() / diff.std(ddof=1) * math.sqrt(len(diff))
            if diff.std() > 0
            else 0
        )
        growth = np.prod(1 + r) ** (252 / len(r)) - 1
        print(
            f"{name:11} {pool:5} {growth:+6.1%} a year, "
            f"Sharpe {sharpe(r):5.2f}, minus the rule {diff.mean() * 252:+6.2%} a year, "
            f"t {t:+5.2f}"
        )
