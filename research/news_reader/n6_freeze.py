"""N6's challenger, frozen once: the chosen variant fitted on every member-day (pre-registration N6).

After the second look (data/news_reader/n6/second_look.json) is above 0, Efe runs:

    PYTHONPATH=src data/llmenv/bin/python research/news_reader/n6_freeze.py

It fits the chosen variant (variants/N6_chosen.json), one model a seed, on every member-day
from 2016 to 9 October 2026 with a next return, and writes data/models/news-n6-forward.npz
once: each seed's parameters as plain float32 arrays (no pickle), the settings, the seeds, the
first and last day trained on, and the SHA-256 of N6's eyes (n6-eyes.npz), with which the
record must read its headlines. Loaded again, the models must give the fitted ones' forecasts
exactly, or nothing is written. It prints the file's SHA-256 for the card.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import n6_harness as nh
import n6_model as nm

WEIGHTS = Path("data/models/news-n6-forward.npz")
EYES = Path("data/models/n6-eyes.npz")
CHOSEN = Path("research/news_reader/variants/N6_chosen.json")
SECOND_LOOK = nh.N6 / "second_look.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def state(models: list[nm.N6]) -> dict[str, np.ndarray]:
    """Every seed's parameters, as plain arrays named seed<k>.<parameter>."""
    return {
        f"seed{k}.{name}": tensor.detach().cpu().numpy().astype(np.float32)
        for k, model in enumerate(models)
        for name, tensor in model.state_dict().items()
    }


def models_from(arrays: dict, device="cpu") -> list[nm.N6]:
    """The frozen models, rebuilt from their arrays."""
    settings = nm.Settings(**json.loads(str(arrays["settings"])))
    out = []
    for k in range(len(arrays["seeds"])):
        model = nm.N6(settings, len(nh.NUMBERS), len(nh.MARKET), len(nh.EVENTS))
        prefix = f"seed{k}."
        model.load_state_dict(
            {
                name[len(prefix) :]: torch.as_tensor(value)
                for name, value in arrays.items()
                if name.startswith(prefix)
            }
        )
        out.append(model.to(device).eval())
    return out


@torch.no_grad()
def forecast(models: list[nm.N6], store, x: dict) -> np.ndarray:
    """The mean of the models' forecasts for the member-days in x."""
    for model in models:
        model.eval()
    return np.mean(
        [model(store, x)[0].double().cpu().numpy() for model in models], axis=0
    )


def freeze(
    a: nh.Arrays,
    store,
    chosen: dict,
    eyes_sha256: str,
    weights: Path = WEIGHTS,
    check_rows: int = 2_000,
) -> str:
    """Fit, check and write the frozen models once; the file's SHA-256."""
    if weights.exists():
        raise SystemExit(
            f"{weights} exists: N6's challenger is frozen once, never again"
        )
    settings = nm.Settings(**chosen["settings"])
    when = a.days["day"].to_numpy()
    trained = ~np.isnan(a.y) & (when <= np.datetime64(nh.LAST_DAY))
    x = nm.inputs(a, trained, store.device)
    models = [nm.fit(a, trained, settings, seed, store, x) for seed in chosen["seeds"]]
    arrays = state(models) | {
        "settings": np.array(json.dumps(chosen["settings"], sort_keys=True)),
        "seeds": np.array(chosen["seeds"]),
        "trained_from": np.array(str(when[trained].min())[:10]),
        "trained_to": np.array(str(when[trained].max())[:10]),
        "eyes_sha256": np.array(eyes_sha256),
    }
    rows = np.flatnonzero(trained)[:check_rows]
    sample = nm.take(x, torch.as_tensor(rows, device=store.device))
    again = models_from(arrays, store.device)
    gap = np.abs(forecast(models, store, sample) - forecast(again, store, sample)).max()
    if gap != 0:
        raise SystemExit(
            f"the arrays' forecasts differ from the fitted ones by {gap:.2e}"
        )
    weights.parent.mkdir(parents=True, exist_ok=True)
    partial = weights.with_suffix(".part.npz")
    np.savez(partial, **arrays)
    partial.rename(weights)
    print(
        f"fitted {len(models)} models on {trained.sum():,} member-days, "
        f"{arrays['trained_from']} to {arrays['trained_to']}; the arrays give their forecasts "
        "exactly"
    )
    return sha256(weights)


def main() -> None:
    import n6_run as nr

    if not SECOND_LOOK.exists():
        raise SystemExit(f"{SECOND_LOOK} is missing: the second look comes first")
    lock = json.loads(SECOND_LOOK.read_text())
    if not lock["report"]["passes"]:
        raise SystemExit("the second look was not above 0: no card, nothing is frozen")
    chosen = json.loads(CHOSEN.read_text())
    if chosen != lock["chosen"]:
        raise SystemExit(f"{CHOSEN} is not the choice the second look ran")
    digest = freeze(nh.load(), nr.gpu_store(), chosen, sha256(EYES))
    print(f"{WEIGHTS}: SHA-256 {digest}")


if __name__ == "__main__":
    main()
