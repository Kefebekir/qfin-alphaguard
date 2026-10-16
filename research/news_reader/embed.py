"""Every headline as numbers, by the ChronoBERT vintage each test year may use (N1, N2).

For test year Y the pre-registrations (and amendment 1) allow ChronoBERT vintage
min(Y - 1, 2024): manelalab/chrono-bert-v1-<vintage>1231, trained only on text from before
the vintage's end. A vintage embeds every article from 2016 to the last test year it serves
(vintage 2017 serves 2018 and so on; vintage 2024 serves 2025 and 2026), so that a test
year's training and test features come from the same model.

A headline's embedding is the mean of the model's last hidden layer over its tokens (at most
64), in float16. Articles are taken in the order in which they became known; the index
records each one's id, the time it is known (created_at, or updated_at when that is more
than 5 minutes later) and the stocks it tags.

Output, under OUT_DIR:
- index.parquet: id, created and known (UTC), symbols, in the order they became known;
- v<vintage>.npy: float16 [articles, 768] for the articles up to the vintage's last test year,
  in index order (the first rows of the index: those known by the end of that year).

Runs on the GPU (data/llmenv: PyTorch with CUDA and transformers). The weights are
safetensors from Hugging Face, loaded without remote code.

Usage: embed.py NEWS_DIR OUT_DIR   (data/llmenv/bin/python)
"""

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
EDIT = timedelta(minutes=5)
BATCH = 256


def last_test_year(vintage: int) -> int:
    return 2026 if vintage == 2024 else vintage + 1


articles = []
for path in sorted(NEWS.glob("*.jsonl")):
    for line in path.read_text().splitlines():
        a = json.loads(line)
        created = datetime.fromisoformat(a["created_at"])
        updated = datetime.fromisoformat(a["updated_at"])
        articles.append(
            {
                "id": a["id"],
                "created": created,
                "known": updated if updated - created > EDIT else created,
                "symbols": a["symbols"],
                "headline": a["headline"],
            }
        )
# The API files articles by their last edit (updated_at), so they are put in the order in
# which they became known; a year's articles then end at one row.
index = (
    pl.DataFrame(articles)
    .with_columns(pl.col("created", "known").dt.convert_time_zone("UTC"))
    .unique("id", keep="last")
    .sort("known", "id")
)
known_years = index["known"].dt.year().to_numpy()
OUT.mkdir(parents=True, exist_ok=True)
index.drop("headline").write_parquet(OUT / "index.parquet")
print(f"{index.height:,} articles, {index['known'].min()} to {index['known'].max()}")

for vintage in VINTAGES:
    target = OUT / f"v{vintage}.npy"
    count = int(np.searchsorted(known_years, last_test_year(vintage), "right"))
    if target.exists():
        print(f"vintage {vintage}: done")
        continue
    name = f"manelalab/chrono-bert-v1-{vintage}1231"
    path = snapshot_download(
        name,
        local_dir=f"data/models/chrono-bert-v1-{vintage}1231",
        allow_patterns=["*.json", "*.safetensors"],
    )
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path, dtype=torch.float16).cuda().eval()
    heads = index["headline"][:count].to_list()
    out = np.empty((count, model.config.hidden_size), dtype=np.float16)
    start = time.time()
    with torch.inference_mode():
        for i in range(0, count, BATCH):
            batch = tokenizer(
                heads[i : i + BATCH],
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt",
            ).to("cuda")
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            mean = (hidden * mask).sum(1) / mask.sum(1)
            out[i : i + len(mean)] = mean.float().cpu().numpy().astype(np.float16)
    if not np.isfinite(out).all():
        sys.exit(f"vintage {vintage}: embeddings that are not finite")
    part = target.with_suffix(".part.npy")
    np.save(part, out)
    part.rename(target)
    print(
        f"vintage {vintage}: {count:,} articles to {last_test_year(vintage)} "
        f"in {time.time() - start:.0f} s",
        flush=True,
    )
    del model
    torch.cuda.empty_cache()
