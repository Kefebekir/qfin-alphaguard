"""N3's features: each member-day's mean headline embedding, in each of the four feature sets.

As N2 (features.py): a member-day's articles are those members.py finds for it, and its
features the mean of their embeddings and log(1 + their number), the log added by the harness.
Here an article's embedding is the one embed_pit.py wrote, by the checkpoint of the year
before the article's (pre-registration N3), so one set of features serves every test year.

The four sets: base-last and base-mean (PIT-4B, the state at the last token or the mean over
the tokens), ft-last and ft-mean (PIT-4B-FT). Sums are float32, as features.py's.

Output, under PIT_EMB: n3_rows.parquet (ticker, day, count), one row a member-day, sorted by
day and ticker, in the order of n3_<set>.npy (float16 [member-days, 4096]).

Usage: features_n3.py EMB_DIR PRICES PIT_EMB   (EMB_DIR holds N2's index.parquet)
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import sparse

sys.path.insert(0, str(Path(__file__).parent))
from members import member_articles

SETS = {
    "base-last": ("PIT-4B", "last"),
    "base-mean": ("PIT-4B", "mean"),
    "ft-last": ("PIT-4B-FT", "last"),
    "ft-mean": ("PIT-4B-FT", "mean"),
}
COLUMNS = 512  # columns averaged at a time, to bound the memory


def member_days(pairs: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The member-days (ticker, day, count), sorted, and each pair's (id, group). An article
    whose tags name one member twice (BRK.B and BRK/B) counts twice, as in N2."""
    rows = pairs.group_by("ticker", "day").agg(count=pl.len()).sort("day", "ticker")
    rows = rows.with_row_index("group")
    pairs = pairs.join(rows, on=["ticker", "day"]).select("id", "group")
    return rows.drop("group"), pairs


def means(
    pairs: pl.DataFrame, groups: int, ids: np.ndarray, vectors: np.ndarray
) -> np.ndarray:
    """Each group's mean of its articles' vectors; `ids` names the rows of `vectors`."""
    order = np.argsort(ids)
    article = pairs["id"].to_numpy()
    at = np.searchsorted(ids[order], article)
    found = at < len(ids)
    found[found] = ids[order][at[found]] == article[found]
    if not found.all():
        raise ValueError(f"{int((~found).sum())} articles have no embedding")
    rows = order[at]
    group = pairs["group"].to_numpy()
    count = np.bincount(group, minlength=groups).astype(np.float32)
    weights = sparse.csr_matrix(
        (1.0 / count[group], (group, rows)), shape=(groups, len(ids)), dtype=np.float32
    )
    out = np.empty((groups, vectors.shape[1]), dtype=np.float16)
    for start in range(0, vectors.shape[1], COLUMNS):
        part = np.asarray(vectors[:, start : start + COLUMNS], dtype=np.float32)
        out[:, start : start + COLUMNS] = weights @ part
    return out


def stacked(model_dir: Path, pool: str) -> tuple[np.ndarray, np.ndarray]:
    """Every checkpoint's ids and vectors of one model and pooling, one after another."""
    codes = sorted(p.name.split(".")[0] for p in model_dir.glob("*.mean.npy"))
    ids = np.concatenate([np.load(model_dir / f"{c}.ids.npy") for c in codes])
    vectors = np.concatenate([np.load(model_dir / f"{c}.{pool}.npy") for c in codes])
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"{model_dir}: an article read by two checkpoints")
    return ids, vectors


if __name__ == "__main__":
    EMB, PRICES, PIT_EMB = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    index = pl.read_parquet(EMB / "index.parquet")
    prices = pl.read_parquet(PRICES, columns=["date", "ticker", "sp500"])
    rows, pairs = member_days(member_articles(index, prices))
    rows.write_parquet(PIT_EMB / "n3_rows.parquet")
    print(f"{rows.height:,} member-days, {pairs.height:,} member articles", flush=True)
    for name, (model, pool) in SETS.items():
        ids, vectors = stacked(PIT_EMB / model, pool)
        out = means(pairs, rows.height, ids, vectors)
        if not np.isfinite(out).all():
            sys.exit(f"{name}: features that are not finite")
        np.save(PIT_EMB / f"n3_{name}.npy", out)
        print(f"{name}: {out.shape[0]:,} x {out.shape[1]:,}", flush=True)
