"""Do PIT-4B's ten checkpoints see a headline alike? Before N6 learns one map for all years.

N3 and N4 read each year's headlines with the checkpoint of the year before (2016 by 201511,
2026 by 202412) and fitted one model on all the years. N6 will learn one projection of the
4,096 numbers a headline becomes. That is sound only if the checkpoints put a headline in the
same place, or in places one rotation apart. This measures it, on headlines alone (no return
is read, so no test is touched):

- 300 member headlines a checkpoint, 3,000 in all, drawn at random from those it read for N3,
  each read again by all ten checkpoints (the last token's state, as N3's base-last);
- each checkpoint's own 300 against what embed_pit.py stored for them (the reading is the same);
- for each pair of checkpoints, on the vectors less their mean over the 3,000:
  - the same headline's cosine (the median): one space, coordinate for coordinate;
  - after the best rotation, fitted on half the headlines and scored on the other half;
  - linear CKA: how alike the two arrangements of the 3,000 are, whatever the rotation;
  - the share of each headline's ten nearest neighbours the two checkpoints agree on;
  - and, for scale, the cosine of two different headlines (about 0);
- the share of a checkpoint's variance its largest dimensions carry, and the measures again
  with each dimension scaled by its spread within each checkpoint.

Output: data/news_reader/eyes/<checkpoint>.npy (float32 [3,000, 4,096]) and ids.npy, each
checkpoint read once; the measures printed. No headline is printed.

Usage: data/llmenv/bin/python research/news_reader/eyes_check.py
"""

import json
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).parent))
from pit import SHAPE, HiddenStates, batches, check, embeddings, load

NEWS = Path("data/frontier/news_all")
PIT = Path("data/news_reader/pit")
STORED = Path("data/news_reader/pit_emb/PIT-4B")
OUT = Path("data/news_reader/eyes")
CHECKPOINTS = (
    "201511",
    "201612",
    "201712",
    "201812",
    "201912",
    "202012",
    "202112",
    "202212",
    "202312",
    "202412",
)
EACH = 300
MAX_TOKENS = 64
NEIGHBOURS = 10


def sample(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """EACH article ids a checkpoint, from those it read; and the checkpoint of each."""
    rng = np.random.default_rng(seed)
    ids, own = [], []
    for k, name in enumerate(CHECKPOINTS):
        read = np.load(STORED / f"{name}.ids.npy")
        ids.append(np.sort(rng.choice(read, EACH, replace=False)))
        own += [k] * EACH
    return np.concatenate(ids), np.array(own)


def headlines(wanted: set[int]) -> dict[int, str]:
    """Each wanted article's headline, from its last version on disk (as embed_pit.py)."""
    out = {}
    for path in sorted(NEWS.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            a = json.loads(line)
            if a["id"] in wanted:
                out[a["id"]] = a["headline"]
    return out


def read(texts: list[str], name: str) -> np.ndarray:
    """The checkpoint's last-token states, as embed_pit.py reads a headline."""
    folder = PIT / f"PIT-4B-{name}"
    check(json.loads((folder / "config.json").read_text()), SHAPE)
    tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
    encoded = [e.ids[:MAX_TOKENS] or [50256] for e in tokenizer.encode_batch(texts)]
    lengths = np.array([len(e) for e in encoded])
    wte, blocks = load(folder / "model.safetensors")
    states = HiddenStates(wte, blocks)
    out = np.empty((len(texts), SHAPE.width), dtype=np.float32)
    for rows in batches(np.argsort(lengths, kind="stable"), lengths, 16_384):
        batch = torch.zeros((len(rows), int(lengths[rows].max())), dtype=torch.long)
        for i, row in enumerate(rows):
            batch[i, : lengths[row]] = torch.tensor(encoded[row])
        last, _ = embeddings(
            states(batch), torch.as_tensor(lengths[rows], device="cuda")
        )
        out[rows] = last.cpu().numpy()
    del wte, blocks, states
    torch.cuda.empty_cache()
    return out


def centred(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(0)).astype(np.float64)


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def same_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """The median cosine of each row of a with the same row of b."""
    return float(np.median((unit(a) * unit(b)).sum(1)))


def rotated_cosine(a: np.ndarray, b: np.ndarray, rng) -> float:
    """same_cosine on held-out rows after the rotation of a that best fits b on the others."""
    order = rng.permutation(len(a))
    fit, test = order[: len(a) // 2], order[len(a) // 2 :]
    u, _, vt = np.linalg.svd(a[fit].T @ b[fit], full_matrices=False)
    return same_cosine(a[test] @ (u @ vt), b[test])


def cka(a: np.ndarray, b: np.ndarray) -> float:
    """Linear CKA (Kornblith et al. 2019) of two centred arrays, rows matched."""
    k, m = a @ a.T, b @ b.T
    return float((k * m).sum() / np.sqrt((k * k).sum() * (m * m).sum()))


def neighbours(x: np.ndarray) -> np.ndarray:
    s = unit(x) @ unit(x).T
    np.fill_diagonal(s, -np.inf)
    return np.argsort(-s, axis=1)[:, :NEIGHBOURS]


def agreement(a: np.ndarray, b: np.ndarray) -> float:
    """The mean share of a row's nearest neighbours that a and b agree on."""
    na, nb = neighbours(a), neighbours(b)
    return float(
        np.mean(
            [len(set(r) & set(s)) / NEIGHBOURS for r, s in zip(na, nb, strict=True)]
        )
    )


def scaled(x: np.ndarray) -> np.ndarray:
    """Each dimension less its mean, over its spread."""
    return centred(x) / x.std(0)


def measures(a: np.ndarray, b: np.ndarray, rng) -> dict:
    a, b = centred(a), centred(b)
    return {
        "same": same_cosine(a, b),
        "rotated": rotated_cosine(a, b, rng),
        "cka": cka(a, b),
        "neighbours": agreement(a, b),
        "different": same_cosine(a, b[rng.permutation(len(b))]),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ids, own = sample()
    if (OUT / "ids.npy").exists() and not np.array_equal(np.load(OUT / "ids.npy"), ids):
        raise SystemExit(f"{OUT}/ids.npy holds another sample")
    np.save(OUT / "ids.npy", ids)
    texts = headlines(set(ids.tolist()))
    print(f"{len(ids):,} headlines, {EACH} a checkpoint; {len(texts):,} found")
    vectors = {}
    for name in CHECKPOINTS:
        path = OUT / f"{name}.npy"
        if not path.exists():
            np.save(path, read([texts[i] for i in ids], name))
            print(f"read by {name}", flush=True)
        vectors[name] = np.load(path)

    worst = 1.0
    for k, name in enumerate(CHECKPOINTS):
        stored_ids = np.load(STORED / f"{name}.ids.npy")
        where = {int(a): i for i, a in enumerate(stored_ids)}
        mine = ids[own == k]
        stored = np.load(STORED / f"{name}.last.npy", mmap_mode="r")
        theirs = np.array([stored[where[int(i)]] for i in mine], dtype=np.float32)
        ours = vectors[name][own == k]
        worst = min(worst, float((unit(ours) * unit(theirs)).sum(1).min()))
    print(
        f"each checkpoint's own headlines against embed_pit.py's: cosine min {worst:.6f}"
    )

    print("\nThe variance a checkpoint's largest dimension and five largest carry:")
    for name, x in vectors.items():
        var = np.sort(x.astype(np.float64).var(0))[::-1]
        top = np.argsort(x.var(0))[::-1][:3].tolist()
        print(
            f"{name}: {var[0] / var.sum():5.1%} {var[:5].sum() / var.sum():5.1%}  "
            f"(dimensions {top})"
        )

    rng = np.random.default_rng(1)
    pairs = list(pairwise(CHECKPOINTS))
    pairs += [(a, "202412") for a in CHECKPOINTS[:-2]]
    print("\nEach pair, on all 3,000 headlines (vectors less their mean):")
    print(
        f"{'pair':<16} {'same':>7} {'rotated':>8} {'CKA':>6} {'kNN@10':>7} {'other':>7}"
    )
    for a, b in pairs:
        m = measures(vectors[a], vectors[b], rng)
        print(
            f"{a + ' ' + b:<16} {m['same']:>7.3f} {m['rotated']:>8.3f} {m['cka']:>6.3f} "
            f"{m['neighbours']:>7.3f} {m['different']:>7.3f}"
        )
    print("\nThe same, each dimension scaled by its spread within each checkpoint:")
    print(f"{'pair':<16} {'same':>7} {'CKA':>6} {'kNN@10':>7}")
    for a, b in pairs:
        sa, sb = scaled(vectors[a]), scaled(vectors[b])
        print(
            f"{a + ' ' + b:<16} {same_cosine(sa, sb):>7.3f} {cka(sa, sb):>6.3f} "
            f"{agreement(sa, sb):>7.3f}"
        )


if __name__ == "__main__":
    main()
