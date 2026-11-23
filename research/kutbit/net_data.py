"""The rows of Efe's network (decision 0027): exactly what the skill test's ridge was given.

So that only the model differs from the ridge, the network learns from and is measured on the
skill test's rows: every minute but the last of every usable session of every instrument but
QQQ, on days when QQQ's session is usable too. A row's input is the 120-minute window's 8
channels and the 11 scalars, clipped at ±10. Its targets are `z` for each horizon, clipped at
±5, and beside them the hedged return and the two known patterns, which the measures need.

A row's window is the WINDOW rows of channels up to it. So each minute's channels are stored
once, not in 120 windows, and a window is gathered when a batch is made, with zeros for the
minutes before its session's open, as the inputs have them. A session's rows are stored one
after another, so a window never reaches into another session.

- `build`: every day's rows of 2016-2021, from the skill test's own code (`skill.each_day`),
  on the real prices and on the skill test's random walks, with the same seeds. Every row's
  window is gathered back and must equal, bit for bit, the input the ridge was given; a day
  that differs stops the build. Each year goes to DATA/kutbit/net/<real|random>/<year>.npz,
  with a manifest of its rows and the SHA-256 of the code that built it. A year already built
  by the same code is not built again. 2022-2026 are not read.
- `speed`: on the GPU, how fast batches of BATCH windows of 2016-2018 are gathered and run
  forward and back through a stand-in network of each width decision 0027 allows, on random
  weights; and from that, how long its runs take at the most. It learns nothing and shows no
  score.

    uv run python research/kutbit/net_data.py build [DATA]
    PYTHONPATH=src data/llmenv/bin/python research/kutbit/net_data.py speed [DATA]
"""

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import skill

from qfin_alphaguard.kutbit.features import CHANNELS, SCALARS, WINDOW

DATA = Path("data")
HERE = Path(__file__).parent
YEARS = (*skill.FIT_YEARS, skill.CHOOSE_YEAR, *skill.MEASURE_YEARS)  # 2016-2021
SOURCES = ("real", "random")
WINDOW_INPUTS = WINDOW * len(CHANNELS)  # the window's part of the 971 inputs
FIELDS = {  # each stored array, a row each, and its type
    "channels": np.float32,  # the row's own channels: the last of its window
    "scalars": np.float32,
    "z": np.float64,  # for each horizon, clipped as the ridge's
    "apart": np.float64,  # for each horizon, the hedged return, unclipped
    "patterns": np.float32,  # habit, and the last 30 minutes apart from the market, reversed
    "minute": np.int16,  # the row's minute in its session, from 0
    "instrument": np.int16,  # an index into tickers.json
    "etf": np.bool_,
    "day": np.int32,  # the session's date, as an ordinal
}
CODE = {  # the code whose SHA-256 each year's record keeps
    "net_data.py": HERE / "net_data.py",
    "skill.py": HERE / "skill.py",
    "dataset.py": HERE / "dataset.py",
    "labels.py": HERE / "labels.py",
    "features.py": HERE.parents[1]
    / "src"
    / "qfin_alphaguard"
    / "kutbit"
    / "features.py",
}
# decision 0027: batches, hidden widths, settings of each width, seeds and epochs
BATCH = 4_096
WIDTHS = (128, 512)
SETTINGS_A_WIDTH, SEEDS, EPOCHS = 2, 5, 20
SEED = 27


# --- a window, gathered from the stored rows ----------------------------------------------


def windows_of(
    channels: np.ndarray, minute: np.ndarray, rows: np.ndarray
) -> np.ndarray:
    """The windows of `rows`, [len(rows), WINDOW, len(CHANNELS)]: each row's channels and those
    of the WINDOW - 1 rows before it, oldest first, and zeros for the minutes before its
    session's open."""
    back = np.arange(WINDOW - 1, -1, -1)
    inside = back[None, :] <= minute[rows][:, None]
    at = np.where(inside, rows[:, None] - back[None, :], 0)
    return np.where(inside[..., None], channels[at], np.float32(0))


def inputs_of(stored: dict, rows: np.ndarray) -> np.ndarray:
    """The 971 inputs of `rows`, laid out as the skill test's: the window, then the scalars."""
    windows = windows_of(stored["channels"], stored["minute"], rows)
    return np.concatenate(
        [windows.reshape(len(rows), -1), stored["scalars"][rows]], axis=1
    )


def gathered(channels, minute, rows):
    """`windows_of` for tensors, on the device they are on."""
    import torch

    back = torch.arange(WINDOW - 1, -1, -1, device=channels.device)
    inside = back[None, :] <= minute[rows][:, None]
    at = torch.where(inside, rows[:, None] - back[None, :], 0)
    return channels[at] * inside[..., None]


# --- the build ------------------------------------------------------------------------------


def stored_rows(rows: skill.Rows, day) -> dict[str, np.ndarray]:
    """One day's rows as they are stored: each row's own channels, the last of its window."""
    n = len(rows.x)
    windows = rows.x[:, :WINDOW_INPUTS].reshape(n, WINDOW, len(CHANNELS))
    found = {
        "channels": windows[:, -1, :],
        "scalars": rows.x[:, WINDOW_INPUTS:],
        "z": rows.z,
        "apart": rows.apart,
        "patterns": rows.patterns,
        "minute": rows.minute,
        "instrument": rows.instrument,
        "etf": rows.etf,
        "day": np.full(n, day.toordinal()),
    }
    return {k: np.ascontiguousarray(v, dtype=FIELDS[k]) for k, v in found.items()}


def fingerprint() -> dict[str, str]:
    """The SHA-256 of the code that builds the rows."""
    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in CODE.items()
    }


def folder(data: Path, source: str) -> Path:
    return data / "kutbit" / "net" / source


def manifest(data: Path, source: str) -> dict:
    """Each built year's rows and the fingerprint of the code that built it."""
    path = folder(data, source) / "manifest.json"
    return json.loads(path.read_text()) if path.exists() else {}


def unread(years) -> None:
    if max(years) > YEARS[-1]:
        raise ValueError("the network's rows stop at 2021; 2022-2026 stay unread")


def build(data: Path, source: str, years=YEARS, log=print) -> Path:
    """Every day's rows of `years` from `source`, "real" or "random", one file a year, each
    row's window checked against the skill test's input. A year already built by today's code
    is skipped."""
    if source not in SOURCES:
        raise ValueError(f"the source is one of {SOURCES}")
    unread(years)
    out = folder(data, source)
    out.mkdir(parents=True, exist_ok=True)
    days = pl.read_parquet(data / "kutbit" / "days.parquet")
    # every ticker ever listed, numbered as the skill test numbers them for its walks' seeds
    names = sorted(days["ticker"].unique())
    tickers = {t: i for i, t in enumerate(names)}
    (out.parent / "tickers.json").write_text(json.dumps(names))
    code, record = fingerprint(), manifest(data, source)
    for year in years:
        path = out / f"{year}.npz"
        if path.exists() and record.get(str(year), {}).get("code") == code:
            log(f"{source} {year}: built by today's code already")
            continue
        started = time.time()
        _, arrays = skill.load(data, [year])
        parts = []
        for day, rows in skill.each_day(days, arrays, [year], tickers, source):
            part = stored_rows(rows, day)
            if not np.array_equal(inputs_of(part, np.arange(len(rows.x))), rows.x):
                raise ValueError(
                    f"{source} {day}: a window gathered back is not the skill test's input"
                )
            parts.append(part)
        if not parts:
            raise ValueError(f"{source} {year}: no rows")
        table = {k: np.concatenate([p[k] for p in parts]) for k in FIELDS}
        partial = out / f"{year}.part.npz"
        np.savez(partial, **table)
        partial.replace(path)
        record[str(year)] = {"rows": len(table["day"]), "code": code}
        (out / "manifest.json").write_text(json.dumps(record, indent=1))
        log(
            f"{source} {year}: {len(table['day']):,} rows ({time.time() - started:.0f} s)"
        )
    return out


def load(data: Path, source: str, years) -> dict[str, np.ndarray]:
    """The stored rows of `years` from `source`, oldest first, as one table. Every year must
    have been built by today's code."""
    unread(years)
    code, record = fingerprint(), manifest(data, source)
    stale = [y for y in years if record.get(str(y), {}).get("code") != code]
    if stale:
        raise ValueError(
            f"{source} {stale}: not built by today's code; run net_data.py build"
        )
    parts = []
    for year in years:
        with np.load(folder(data, source) / f"{year}.npz") as stored:
            parts.append({k: stored[k] for k in FIELDS})
    return {k: np.concatenate([p[k] for p in parts]) for k in FIELDS}


# --- the speed ------------------------------------------------------------------------------


def speed(data: Path, batches: int = 200, log=print) -> dict:
    """How long decision 0027's runs take at the most, from batches of 2016-2018 gathered on
    the GPU and run forward and back through a stand-in network of each width, on random
    weights. It learns nothing and shows no score."""
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    table = load(data, "real", skill.FIT_YEARS)
    rows = len(table["day"])
    channels = torch.from_numpy(table["channels"]).to(device)
    scalars = torch.from_numpy(table["scalars"]).to(device)
    minute = torch.from_numpy(table["minute"].astype(np.int64)).to(device)
    z = torch.from_numpy(table["z"].astype(np.float32)).to(device)
    del table
    size = min(BATCH, rows)
    order = torch.randperm(
        rows, device=device, generator=torch.Generator(device).manual_seed(SEED)
    )
    picks = [
        order[i * size : (i + 1) * size]
        for i in range(max(min(batches, rows // size), 1))
    ]

    def batch(pick):
        x = gathered(channels, minute, pick).reshape(len(pick), -1)
        return torch.cat([x, scalars[pick]], dim=1), z[pick]

    def timed(work) -> float:
        work(picks[0])  # the first call warms the device up
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        for pick in picks:
            work(pick)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return (time.perf_counter() - started) / len(picks)

    found = {
        "device": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "cpu",
        "rows": rows,
        "batch": size,
        "gather": timed(batch),
    }

    def stand_in(width: int) -> dict:
        """The seconds a batch takes through a stand-in of decision 0027's network at
        `width`, on random weights: forward and back, and forward alone."""
        torch.manual_seed(SEED)
        inputs, horizons = WINDOW_INPUTS + len(SCALARS), z.shape[1]
        shapes = [
            (inputs, width),
            (width, width),
            (width, horizons),
            (inputs, horizons),
        ]
        weights = [
            (torch.randn(*s, device=device) / math.sqrt(s[0])).requires_grad_()
            for s in shapes
        ]

        def forward(x):
            hidden = torch.relu(torch.relu(x @ weights[0]) @ weights[1])
            return hidden @ weights[2] + x @ weights[3]

        def step(pick):
            x, target = batch(pick)
            ((forward(x) - target) ** 2).mean().backward()
            for w in weights:
                w.grad = None

        def forecast(pick):
            with torch.no_grad():
                forward(batch(pick)[0])

        return {"step": timed(step), "forecast": timed(forecast)}

    for width in WIDTHS:
        found[width] = stand_in(width)
    if device.type == "cuda":
        found["memory"] = torch.cuda.max_memory_allocated(device) / 1e9
        found["of"] = torch.cuda.get_device_properties(device).total_memory / 1e9
    found |= runs(found, manifest(data, "real"))
    log(said(found))
    return found


def runs(found: dict, record: dict) -> dict:
    """Decision 0027's runs at the most, in hours, from the seconds a batch takes: the choice,
    four settings of five seeds for EPOCHS epochs on 2016-2018, each epoch's forecasts of 2019
    beside it; and the measurement, the network and the one on shuffled labels learning again
    on 2016-2019 at the wider width for every epoch. The placebo takes as much again."""
    rows = {y: record[str(y)]["rows"] for y in YEARS}

    def batches(years) -> int:
        return math.ceil(sum(rows[y] for y in years) / found["batch"])

    choice = 0.0
    for w in WIDTHS:
        an_epoch = batches(skill.FIT_YEARS) * found[w]["step"]
        its_forecasts = batches([skill.CHOOSE_YEAR]) * found[w]["forecast"]
        choice += SETTINGS_A_WIDTH * SEEDS * EPOCHS * (an_epoch + its_forecasts)
    wide = found[WIDTHS[-1]]
    learning = EPOCHS * batches((*skill.FIT_YEARS, skill.CHOOSE_YEAR)) * wide["step"]
    forecasts = batches((skill.CHOOSE_YEAR, *skill.MEASURE_YEARS)) * wide["forecast"]
    twins = 2  # the network, and the one on shuffled labels
    measure = twins * SEEDS * (learning + forecasts)
    return {"choice hours": choice / 3600, "measurement hours": measure / 3600}


def said(found: dict) -> str:
    """What `speed` found, in a few lines."""
    lines = [
        f"device: {found['device']}; {found['rows']:,} rows of 2016-2018",
        f"gathering a batch of {found['batch']:,} windows: {found['gather'] * 1e3:.1f} ms",
    ]
    for width in WIDTHS:
        epoch = math.ceil(found["rows"] / found["batch"]) * found[width]["step"]
        lines.append(
            f"width {width}: a batch forward and back {found[width]['step'] * 1e3:.1f} ms, "
            f"an epoch of 2016-2018 {epoch:.0f} s"
        )
    lines.append(
        f"decision 0027 at the most: the choice {found['choice hours']:.1f} h, the "
        f"measurement {found['measurement hours']:.1f} h; the placebo as much again"
    )
    if "memory" in found:
        lines.append(f"GPU memory: {found['memory']:.1f} GB of {found['of']:.1f}")
    return "\n".join(lines)


# --- the commands ---------------------------------------------------------------------------------


def main(argv: list[str]) -> None:
    if not argv or argv[0] not in ("build", "speed"):
        raise SystemExit("usage: net_data.py build|speed [DATA]")
    data = Path(argv[1]) if len(argv) > 1 else DATA
    if argv[0] == "speed":
        speed(data)
        return
    for source in SOURCES:
        build(data, source)
    print(f"build: {folder(data, 'real').parent}")


if __name__ == "__main__":
    main(sys.argv[1:])
