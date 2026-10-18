"""N3's models: the PIT-4B checkpoints, each pinned and checked (pre-registration N3).

Year Y's headlines are read by the checkpoint of December Y-1, 2016's by November 2015, and
2025's and 2026's by December 2024, the latest published. For each checkpoint of PIT-4B and
PIT-4B-FT this downloads, at the repository's current commit, the weights, the config and the
tokenizer, and checks the weights' SHA-256 against the one Hugging Face records. The
repository's own code (modeling_pit.py) is not downloaded: embed_pit.py has its own copy of
the architecture. Each checkpoint's repository, commit, SHA-256 and month go to
OUT/manifest.json, for the ledger; a checkpoint already there and whole is not fetched again.

The weights are about 17.75 GB a checkpoint, 355 GB for all twenty.

Usage: data/llmenv/bin/python pit_download.py OUT   (OUT: data/news_reader/pit)
"""

import hashlib
import json
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

OWNER = (
    "Diamegs"  # the same files as anonymneurips2027, the NeurIPS submission's account
)
MODELS = ("PIT-4B", "PIT-4B-FT")
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
FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors")

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
MANIFEST = OUT / "manifest.json"
manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
api = HfApi()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


for model in MODELS:
    for checkpoint in CHECKPOINTS:
        name = f"{model}-{checkpoint}"
        repo = f"{OWNER}/{name}"
        here = OUT / name
        weights = here / "model.safetensors"
        if name in manifest and weights.exists():
            print(f"{name}: already here", flush=True)
            continue
        info = api.model_info(repo, files_metadata=True)
        expected = next(
            s.lfs.sha256 for s in info.siblings if s.rfilename == "model.safetensors"
        )
        started = time.monotonic()
        for filename in FILES:
            hf_hub_download(repo, filename, revision=info.sha, local_dir=here)
        took = time.monotonic() - started
        got = sha256(weights)
        if got != expected:
            raise SystemExit(f"{name}: SHA-256 {got} is not Hugging Face's {expected}")
        config = json.loads((here / "config.json").read_text())
        manifest[name] = {
            "repository": repo,
            "commit": info.sha,
            "sha256": got,
            "month": config["month"],
            "bytes": weights.stat().st_size,
        }
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
        print(
            f"{name}: month {config['month']}, {weights.stat().st_size / 1e9:.2f} GB in "
            f"{took / 60:.1f} min ({weights.stat().st_size / took / 1e6:.0f} MB/s), SHA-256 checked",
            flush=True,
        )
print(f"{len(manifest)} checkpoints in {MANIFEST}")
