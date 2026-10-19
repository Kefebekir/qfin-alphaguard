"""N3's embeddings: each member headline read by PIT-4B of the year before.

Pre-registration N3: an article known in year Y (New York's calendar) is read by the
checkpoint of December Y-1, an article of 2016 by November 2015, and articles of 2025 and 2026
by December 2024. Each article is read once by each model, as in the papers behind N3 (Kelly
et al. embed each year's articles with the model of the year before; so does the ChronoBERT paper).
N2 differs: each of its test years reads every year with one vintage.

The articles: those N2 counts for a member's session (members.py), 481,228 of them. A
headline is its first 64 tokens (GPT-2's tokenizer, as the models were trained with), with
no token before them, as the model card reads text. Two embeddings from one pass: the final
state at the last token, and the mean of the final states over the tokens (pit.py). Headlines
go in order of length, so that a batch carries little padding. The checkpoints are those
pit_download.py fetched and checked; before use, each one's SHA-256 is checked again and its
config against the architecture pit.py copies.

Output, under OUT_DIR/<model>/: <checkpoint>.ids.npy (int64), <checkpoint>.last.npy and
<checkpoint>.mean.npy (float16 [articles, 4096]). A checkpoint already done is skipped. A
lock keeps two runs from sharing the GPU.

Usage: embed_pit.py NEWS_DIR EMB_DIR PRICES PIT_DIR OUT_DIR   (data/llmenv/bin/python)
"""

import fcntl
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).parent))
from members import checkpoint, member_articles
from pit import SHAPE, HiddenStates, batches, check, embeddings, load

NEWS, EMB, PRICES, PIT, OUT = (Path(a) for a in sys.argv[1:6])
MODELS = ("PIT-4B",)  # amendment 4 dropped PIT-4B-FT
MAX_TOKENS = 64
BATCH_TOKENS = 16_384  # tokens a batch, padding included


def headlines(wanted: set[int]) -> dict[int, str]:
    """Each wanted article's headline, from its last version on disk."""
    out = {}
    for path in sorted(NEWS.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            a = json.loads(line)
            if a["id"] in wanted:
                out[a["id"]] = a["headline"]
    return out


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


OUT.mkdir(parents=True, exist_ok=True)
lock = (OUT / ".lock").open("w")
fcntl.flock(lock, fcntl.LOCK_EX)

index = pl.read_parquet(EMB / "index.parquet")
prices = pl.read_parquet(PRICES, columns=["date", "ticker", "sp500"])
articles = member_articles(index, prices).select("id").unique()
known = index.select("id", "known").join(articles, on="id", how="inner")
known = known.with_columns(
    pl.col("known").map_elements(checkpoint, return_dtype=pl.String).alias("checkpoint")
)
text = headlines(set(known["id"].to_list()))
print(
    f"{known.height:,} member articles; by checkpoint:",
    dict(known.group_by("checkpoint").len().sort("checkpoint").iter_rows()),
    flush=True,
)
manifest = json.loads((PIT / "manifest.json").read_text())

for model in MODELS:
    (OUT / model).mkdir(exist_ok=True)
    for code in sorted(
        known["checkpoint"].unique()
    ):  # the order pit_download.py fetches
        part = known.filter(pl.col("checkpoint") == code).sort("id")
        done = OUT / model / f"{code}.mean.npy"
        if done.exists():
            print(f"{model}-{code}: done", flush=True)
            continue
        name = f"{model}-{code}"
        weights = PIT / name / "model.safetensors"
        while name not in manifest or not weights.exists():
            time.sleep(60)  # pit_download.py is still fetching it
            manifest = json.loads((PIT / "manifest.json").read_text())
        if sha256(weights) != manifest[name]["sha256"]:
            sys.exit(f"{name}: the weights are not the ones downloaded and checked")
        check(json.loads((PIT / name / "config.json").read_text()), SHAPE)
        started = time.monotonic()
        tokenizer = Tokenizer.from_file(str(PIT / name / "tokenizer.json"))
        ids = part["id"].to_numpy()
        encoded = [
            e.ids[:MAX_TOKENS] for e in tokenizer.encode_batch([text[i] for i in ids])
        ]
        lengths = np.array([max(len(e), 1) for e in encoded])
        order = np.argsort(lengths, kind="stable")
        wte, blocks = load(weights)
        states = HiddenStates(wte, blocks)
        last = np.empty((len(ids), SHAPE.width), dtype=np.float16)
        mean = np.empty((len(ids), SHAPE.width), dtype=np.float16)
        groups = batches(order, lengths, BATCH_TOKENS)
        for g, rows in enumerate(groups, 1):
            if g % max(len(groups) // 10, 1) == 0:
                print(
                    f"{name}: batch {g:,} of {len(groups):,}, at {datetime.now(UTC):%H:%M} UTC",
                    flush=True,
                )
            width = int(lengths[rows].max())
            batch = torch.zeros((len(rows), width), dtype=torch.long)
            for r, row in enumerate(rows):
                tokens = encoded[row] or [
                    50256
                ]  # an empty headline: the end-of-text token
                batch[r, : len(tokens)] = torch.tensor(tokens)
            hidden = states(batch)
            a, b = embeddings(
                hidden, torch.as_tensor(lengths[rows], device=hidden.device)
            )
            last[rows] = a.cpu().numpy().astype(np.float16)
            mean[rows] = b.cpu().numpy().astype(np.float16)
        if not (np.isfinite(last).all() and np.isfinite(mean).all()):
            sys.exit(f"{name}: embeddings that are not finite")
        np.save(OUT / model / f"{code}.ids.npy", ids.astype(np.int64))
        np.save(OUT / model / f"{code}.last.npy", last)
        np.save(OUT / model / f"{code}.mean.part.npy", mean)
        (OUT / model / f"{code}.mean.part.npy").rename(done)
        del wte, blocks, states
        torch.cuda.empty_cache()
        took = time.monotonic() - started
        print(
            f"{name}: {len(ids):,} headlines in {took / 60:.1f} min ({len(ids) / took:,.0f} a second), "
            f"month {manifest[name]['month']}, at {datetime.now(UTC):%H:%M} UTC",
            flush=True,
        )
