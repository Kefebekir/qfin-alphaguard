"""Every headline as numbers, by the ChronoBERT vintage each test year may use (N1, N2).

For test year Y the pre-registrations (and amendment 1) allow ChronoBERT vintage
min(Y - 1, 2024): manelalab/chrono-bert-v1-<vintage>1231, trained only on text from before
the vintage's end. A vintage embeds every article known by the end of the last test year it
serves (vintage 2017 serves 2018 and so on; vintage 2024 serves 2025 and 2026), so that a
test year's training and test features come from the same model.

A headline's embedding is the mean of the model's last hidden layer over its tokens (at most
64), in float16. An article is known at created_at, or at updated_at when that is more than
5 minutes later.

It runs beside news_all.py: a vintage starts once the months it needs are on disk. Alpaca
files an article under the month of its last edit, so an article known by 31 December can
sit in January's file (an edit within 5 minutes); a vintage therefore waits for the January
after its last test year, and vintage 2024 for every month. Embeddings are stored with their
article ids, not by position: a later month can hold a later edit of an article, which moves
it later, and features.py looks each article up by id. Headlines are embedded in order of
length, so that a batch carries little padding. A lock keeps two runs from sharing the GPU.

Output, under OUT_DIR:
- v<vintage>.npy: float16 [articles, 768], and v<vintage>.ids.npy: their ids;
- index.parquet, once every month is on disk: id, created and known (UTC), symbols, in the
  order the articles became known.

Runs on the GPU (data/llmenv: PyTorch with CUDA and transformers). The weights are
safetensors from Hugging Face, loaded without remote code.

Usage: embed.py NEWS_DIR OUT_DIR   (data/llmenv/bin/python)
"""

import fcntl
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer

NEWS, OUT = Path(sys.argv[1]), Path(sys.argv[2])
VINTAGES = range(2017, 2025)  # test years 2018-2026
FIRST_MONTH, LAST_MONTH = (2016, 1), (2026, 10)  # as news_all.py
EDIT = timedelta(minutes=5)
BATCH = 512


def last_test_year(vintage: int) -> int:
    return 2026 if vintage == 2024 else vintage + 1


def months_through(year: int, month: int) -> list[str]:
    out, (y, m) = [], FIRST_MONTH
    while (y, m) <= min((year, month), LAST_MONTH):
        out.append(f"{y}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def needed(vintage: int) -> list[str]:
    """The monthly files that hold every article known by the vintage's last test year."""
    return months_through(last_test_year(vintage) + 1, 1)


def articles() -> pl.DataFrame:
    """Every article on disk, its last version, in the order the articles became known."""
    rows = []
    for path in sorted(NEWS.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            a = json.loads(line)
            created = datetime.fromisoformat(a["created_at"])
            updated = datetime.fromisoformat(a["updated_at"])
            rows.append(
                {
                    "id": a["id"],
                    "created": created,
                    "known": updated if updated - created > EDIT else created,
                    "symbols": a["symbols"],
                    "headline": a["headline"],
                }
            )
    return (
        pl.DataFrame(rows)
        .with_columns(pl.col("created", "known").dt.convert_time_zone("UTC"))
        .unique("id", keep="last")
        .sort("known", "id")
    )


def embed(vintage: int, heads: list[str]) -> np.ndarray:
    path = snapshot_download(
        f"manelalab/chrono-bert-v1-{vintage}1231",
        local_dir=f"data/models/chrono-bert-v1-{vintage}1231",
        allow_patterns=["*.json", "*.safetensors"],
    )
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path, dtype=torch.float16).cuda().eval()
    out = np.empty((len(heads), model.config.hidden_size), dtype=np.float16)
    order = np.argsort([len(h) for h in heads], kind="stable")
    with torch.inference_mode():
        for i in range(0, len(heads), BATCH):
            rows = order[i : i + BATCH]
            batch = tokenizer(
                [heads[r] for r in rows],
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt",
            ).to("cuda")
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            mean = (hidden * mask).sum(1) / mask.sum(1)
            out[rows] = mean.float().cpu().numpy().astype(np.float16)
    del model
    torch.cuda.empty_cache()
    return out


OUT.mkdir(parents=True, exist_ok=True)
lock = (OUT / ".lock").open("w")
fcntl.flock(lock, fcntl.LOCK_EX)  # one run at a time on the GPU

for vintage in VINTAGES:
    target = OUT / f"v{vintage}.npy"
    if target.exists():
        print(f"vintage {vintage}: done", flush=True)
        continue
    while not all((NEWS / f"{m}.jsonl").exists() for m in needed(vintage)):
        time.sleep(30)
    index = articles()
    known = index.filter(pl.col("known").dt.year() <= last_test_year(vintage))
    start = time.time()
    vectors = embed(vintage, known["headline"].to_list())
    if not np.isfinite(vectors).all():
        sys.exit(f"vintage {vintage}: embeddings that are not finite")
    np.save(OUT / f"v{vintage}.ids.npy", known["id"].to_numpy().astype(np.int64))
    part = target.with_suffix(".part.npy")
    np.save(part, vectors)
    part.rename(target)
    print(
        f"vintage {vintage}: {known.height:,} articles to {last_test_year(vintage)} "
        f"in {time.time() - start:.0f} s ({known.height / (time.time() - start):,.0f} a second)",
        flush=True,
    )

if all((NEWS / f"{m}.jsonl").exists() for m in months_through(*LAST_MONTH)):
    index = articles()
    index.drop("headline").write_parquet(OUT / "index.parquet")
    print(
        f"index: {index.height:,} articles, {index['known'].min()} to {index['known'].max()}"
    )
