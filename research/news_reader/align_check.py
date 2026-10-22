"""Two ways to put PIT-4B's ten checkpoints on one scale, judged on headlines alone (before N6).

eyes_check.py found the checkpoints share one coordinate system that drifts, and that two of
them (202012, 202112) have one dimension carrying a third and a fifth of their variance. N6's
past years are read by nine older checkpoints and its record by 202412, so each older one is
brought to 202412's scale, in one of two ways:

- A: each dimension less its mean, over its spread, both from the anchors as that checkpoint
  reads them; 202412's reading the same way.
- B: A, then a linear map from the checkpoint's scaled reading to 202412's, fitted by ridge on
  the anchors. Its penalty is the one that best fits a fifth of the anchors held out (the
  median cosine), from 10 to 10^6; then it is fitted on them all.

The anchors: 20,000 member headlines, 2,000 from each checkpoint's own, none of eyes_check's.
Each is read by all ten checkpoints. The judges: eyes_check's 3,000 headlines, which no
method saw, as each older checkpoint reads them brought to 202412's scale, against 202412's
own reading: the same headline's median cosine, the share of its ten nearest neighbours they
agree on, and linear CKA. No return is read.

THE RULE, fixed before any of this was run (committed before its first run):
B is chosen if, over the nine older checkpoints, its mean share of neighbours agreed exceeds
A's by at least 0.02, and its median cosine is above A's for at least seven of the nine.
Otherwise A is chosen: simpler, and its map holds nothing 202412 knows of later years.

Output: data/news_reader/eyes/anchors.npy and anchors_<checkpoint>.npy (float16), each
checkpoint read once; the measures and the choice printed. No headline is printed.

Usage: data/llmenv/bin/python research/news_reader/align_check.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eyes_check import (
    CHECKPOINTS,
    OUT,
    STORED,
    agreement,
    centred,
    cka,
    headlines,
    read,
    same_cosine,
)

REFERENCE = "202412"
EACH = 2_000
PENALTIES = [10.0**k for k in range(1, 7)]
MARGIN = 0.02  # the rule's margin in neighbours agreed
WINS = 7  # of nine


def anchors(seed: int = 2) -> np.ndarray:
    """EACH article ids a checkpoint, from those it read, none of eyes_check's sample."""
    rng = np.random.default_rng(seed)
    judged = set(np.load(OUT / "ids.npy").tolist())
    out = []
    for name in CHECKPOINTS:
        read_ids = np.load(STORED / f"{name}.ids.npy")
        free = np.array([i for i in read_ids.tolist() if i not in judged])
        out.append(np.sort(rng.choice(free, EACH, replace=False)))
    return np.concatenate(out)


class Scale:
    """Method A for one checkpoint: each dimension's mean and spread on the anchors."""

    def __init__(self, x: np.ndarray):
        x = x.astype(np.float64)
        self.mean, self.spread = x.mean(0), x.std(0)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x.astype(np.float64) - self.mean) / self.spread


def ridge_map(z: np.ndarray, y: np.ndarray, rng) -> tuple[np.ndarray, float, dict]:
    """Method B's map from z to y (both scaled): the penalty that best fits a held-out
    fifth, then fitted on all rows. Returns the map, its penalty and each penalty's fit."""
    z, y = z.astype(np.float32), y.astype(np.float32)
    order = rng.permutation(len(z))
    held, fit = order[: len(z) // 5], order[len(z) // 5 :]

    def solve(rows):
        u, s, vt = np.linalg.svd(z[rows], full_matrices=False)
        return s, vt, u.T @ y[rows]

    s, vt, uty = solve(fit)
    scores = {}
    for penalty in PENALTIES:
        w = vt.T @ ((s / (s**2 + penalty))[:, None] * uty)
        scores[penalty] = same_cosine(z[held] @ w, y[held])
    best = max(scores, key=scores.get)
    s, vt, uty = solve(np.arange(len(z)))
    return vt.T @ ((s / (s**2 + best))[:, None] * uty), best, scores


def fit(
    reads: dict[str, np.ndarray],
) -> tuple[dict[str, Scale], dict[str, tuple[np.ndarray, float]]]:
    """Each checkpoint's scale on the anchors, and each older one's map to REFERENCE with
    its penalty: B as the rule judged it (the penalties held out in this order, seed 3)."""
    rng = np.random.default_rng(3)
    scale = {name: Scale(x) for name, x in reads.items()}
    y = scale[REFERENCE](reads[REFERENCE])
    maps = {}
    for name in CHECKPOINTS[:-1]:
        w, penalty, _ = ridge_map(scale[name](reads[name]), y, rng)
        maps[name] = (w, penalty)
    return scale, maps


def main() -> None:
    ids = anchors()
    path = OUT / "anchors.npy"
    if path.exists() and not np.array_equal(np.load(path), ids):
        raise SystemExit(f"{path} holds other anchors")
    np.save(path, ids)
    texts = None
    reads = {}
    for name in CHECKPOINTS:
        file = OUT / f"anchors_{name}.npy"
        if not file.exists():
            texts = texts or headlines(set(ids.tolist()))
            np.save(file, read([texts[i] for i in ids], name).astype(np.float16))
            print(f"anchors read by {name}", flush=True)
        reads[name] = np.load(file)
    judged = {name: np.load(OUT / f"{name}.npy") for name in CHECKPOINTS}
    print(f"{len(ids):,} anchors; {len(judged[REFERENCE]):,} headlines judged")

    scale, maps = fit(reads)
    target = scale[REFERENCE](judged[REFERENCE])
    rows, a_knn, b_knn, b_wins = [], [], [], 0
    for name in CHECKPOINTS[:-1]:
        w, penalty = maps[name]
        none = judged[name].astype(np.float64) - scale[name].mean
        a = scale[name](judged[name])
        b = a @ w
        m = {}
        for method, x in (("none", none), ("A", a), ("B", b)):
            cx, ct = centred(x), centred(target)
            m[method] = (same_cosine(cx, ct), agreement(cx, ct), cka(cx, ct))
        a_knn.append(m["A"][1])
        b_knn.append(m["B"][1])
        b_wins += m["B"][0] > m["A"][0]
        rows.append((name, penalty, m))

    print("\nAgainst 202412's own reading, on the 3,000 headlines judged:")
    print(
        f"{'checkpoint':<11}"
        + "".join(
            f"{f'{k} cos':>9}{f'{k} kNN':>9}{f'{k} CKA':>9}" for k in ("none", "A", "B")
        )
        + f"{'penalty':>9}"
    )
    for name, penalty, m in rows:
        cells = "".join(
            f"{c:>9.3f}{n:>9.3f}{k:>9.3f}" for c, n, k in (m["none"], m["A"], m["B"])
        )
        print(f"{name:<11}{cells}{penalty:>9.0e}")
    gain = float(np.mean(b_knn) - np.mean(a_knn))
    chosen = "B" if gain >= MARGIN and b_wins >= WINS else "A"
    print(
        f"\nB less A in neighbours agreed, mean of nine: {gain:+.3f} (needs {MARGIN:+.2f}); "
        f"B's cosine above A's for {b_wins} of 9 (needs {WINS}).\nThe rule chooses {chosen}."
    )


if __name__ == "__main__":
    main()
