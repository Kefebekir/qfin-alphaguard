"""N6's eyes: every member headline in 202412's scaled reading (method B, align_check.py).

align_check.py's rule chose B: each checkpoint's dimensions scaled by their mean and spread on
20,000 anchor headlines, then each older checkpoint's reading mapped by ridge to 202412's
scaled reading. This freezes those arrays once and writes every headline N3 read (embed_pit's
last-token states, each by the checkpoint of the year before its own) in that one scale:

- data/models/n6-eyes.npz: each checkpoint's mean and spread, and each older one's map and
  its penalty (float32). Fitted again from the anchors on disk, they must be the same arrays,
  or the run stops.
- data/news_reader/n6/headlines.npy (float16 [articles, 4,096]), with ids.npy (int64) and
  checkpoint.npy (int8: the reading checkpoint's place in CHECKPOINTS), checkpoint by
  checkpoint in embed_pit's order. Written once.

Checks, printed: the frozen arrays give align_check's judged results (B's cosine and share of
neighbours for each older checkpoint, as results_align_check.txt rounds them); the store's
rows for the 3,000 judged headlines, from embed_pit's float16 states, against the frozen
arrays applied to eyes_check's float32 reading of them; each checkpoint's rows' spread.

Usage: data/llmenv/bin/python research/news_reader/n6_eyes.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from align_check import REFERENCE, fit
from eyes_check import (
    CHECKPOINTS,
    OUT,
    STORED,
    agreement,
    centred,
    same_cosine,
    sample,
    unit,
)

EYES = Path("data/models/n6-eyes.npz")
STORE = Path("data/news_reader/n6")
RESULTS = Path(__file__).parent / "results_align_check.txt"
CHUNK = 16_384


def freeze(scale: dict, maps: dict) -> dict[str, np.ndarray]:
    """The arrays B needs, by name."""
    arrays = {}
    for name in CHECKPOINTS:
        arrays[f"mean_{name}"] = scale[name].mean.astype(np.float32)
        arrays[f"spread_{name}"] = scale[name].spread.astype(np.float32)
        if name in maps:
            w, penalty = maps[name]
            arrays[f"map_{name}"] = w.astype(np.float32)
            arrays[f"penalty_{name}"] = np.array(penalty, dtype=np.float32)
    return arrays


def aligned(arrays: dict[str, np.ndarray], name: str, x: np.ndarray) -> np.ndarray:
    """Checkpoint `name`'s reading x in 202412's scaled reading (float32)."""
    z = (np.asarray(x, dtype=np.float32) - arrays[f"mean_{name}"]) / arrays[
        f"spread_{name}"
    ]
    return z @ arrays[f"map_{name}"] if f"map_{name}" in arrays else z


def keep(arrays: dict[str, np.ndarray], path: Path = EYES) -> None:
    """Write the arrays once; a file already there must hold the same."""
    if path.exists():
        with np.load(path) as kept:
            same = set(kept.files) == set(arrays) and all(
                np.array_equal(kept[k], arrays[k]) for k in arrays
            )
        if not same:
            raise SystemExit(f"{path} holds other arrays: N6's eyes are frozen once")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part.npz")
    np.savez(partial, **arrays)
    partial.rename(path)


def build(
    arrays: dict[str, np.ndarray],
    source: Path = STORED,
    out: Path = STORE,
    chunk: int = CHUNK,
) -> None:
    """Every checkpoint's headlines in the one scale, in CHECKPOINTS' order; once."""
    if (out / "headlines.npy").exists():
        return
    sizes = {name: len(np.load(source / f"{name}.ids.npy")) for name in CHECKPOINTS}
    width = len(arrays[f"mean_{CHECKPOINTS[0]}"])
    out.mkdir(parents=True, exist_ok=True)
    partial = out / "headlines.part.npy"
    store = np.lib.format.open_memmap(
        partial, mode="w+", dtype=np.float16, shape=(sum(sizes.values()), width)
    )
    ids, which, start = [], [], 0
    for k, name in enumerate(CHECKPOINTS):
        x = np.load(source / f"{name}.last.npy", mmap_mode="r")
        for first in range(0, sizes[name], chunk):
            rows = x[first : first + chunk]
            store[start + first : start + first + len(rows)] = aligned(
                arrays, name, rows
            )
        ids.append(np.load(source / f"{name}.ids.npy"))
        which.append(np.full(sizes[name], k, dtype=np.int8))
        start += sizes[name]
    store.flush()
    del store
    np.save(out / "ids.npy", np.concatenate(ids).astype(np.int64))
    np.save(out / "checkpoint.npy", np.concatenate(which))
    partial.rename(out / "headlines.npy")


def judged_results(path: Path = RESULTS) -> dict[str, tuple[float, float]]:
    """B's cosine and share of neighbours for each older checkpoint, as align_check printed
    them."""
    found = {}
    for line in path.read_text().splitlines():
        cells = line.split()
        if cells and cells[0] in CHECKPOINTS and len(cells) == 11:
            found[cells[0]] = (float(cells[7]), float(cells[8]))
    return found


def main() -> None:
    reads = {name: np.load(OUT / f"anchors_{name}.npy") for name in CHECKPOINTS}
    arrays = freeze(*fit(reads))
    keep(arrays)
    print(f"{EYES}: {len(arrays)} arrays, the same as fitted again from the anchors")

    judged = {name: np.load(OUT / f"{name}.npy") for name in CHECKPOINTS}
    target = centred(aligned(arrays, REFERENCE, judged[REFERENCE]))
    printed = judged_results()
    gap = 0.0
    for name in CHECKPOINTS[:-1]:
        b = centred(aligned(arrays, name, judged[name]))
        mine = (same_cosine(b, target), agreement(b, target))
        gap = max(gap, *(abs(round(m, 3) - p) for m, p in zip(mine, printed[name])))
    print(
        f"the frozen arrays against align_check's judged B, nine checkpoints: largest gap "
        f"{gap:.3f} at three decimals"
    )

    build(arrays)
    store = np.load(STORE / "headlines.npy", mmap_mode="r")
    ids, which = np.load(STORE / "ids.npy"), np.load(STORE / "checkpoint.npy")
    where = {int(i): row for row, i in enumerate(ids)}
    eyes_ids, own = sample()
    worst = 1.0
    for k, name in enumerate(CHECKPOINTS):
        mine = eyes_ids[own == k]
        rows = np.array([where[int(i)] for i in mine])
        assert (which[rows] == k).all()
        stored = np.asarray(store[rows], dtype=np.float32)
        fresh = aligned(arrays, name, judged[name][own == k])
        worst = min(worst, float((unit(stored) * unit(fresh)).sum(1).min()))
    print(
        f"{STORE / 'headlines.npy'}: {len(ids):,} headlines; the 3,000 judged against "
        f"eyes_check's reading in the one scale: cosine min {worst:.6f}"
    )
    print("each checkpoint's headlines, the median spread of a dimension:")
    for k, name in enumerate(CHECKPOINTS):
        rows = np.flatnonzero(which == k)[:20_000]
        spread = np.median(np.asarray(store[rows], dtype=np.float32).std(0))
        print(f"  {name}: {spread:.3f}")


if __name__ == "__main__":
    main()
