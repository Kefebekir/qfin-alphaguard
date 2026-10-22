"""N5's challenger: N4's variant 2, frozen once, then run forward (pre-registration N5).

Freezing, by Efe, once:

    uv run python research/news_reader/forward.py freeze

It fits N4's chosen variant (variants/N4_chosen.json, through efe_model.make) on every
member-day from 2016 to the last with a next close, the training rows of a test year 2027,
and writes its weights as plain arrays to data/models/news-n4-forward.npz: the principal
components and their mean, the scaler's mean and scale, the ridge's coefficients and
intercept. No pickle: forecast() computes the pipeline's forecast from the arrays, and the
freeze stops unless it gives the fitted model's on every training row, to a hundred-thousandth
of their spread (the model computes in float32, the arrays in float64). A weights file that
exists is never written again. The SHA-256 it prints goes on the challenger card.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

WEIGHTS = Path("data/models/news-n4-forward.npz")
TRAIN_THROUGH = 2027  # the training rows of test year 2027: every year before it


def freeze_arrays(model) -> dict[str, np.ndarray]:
    """The fitted pipeline (efe_model.make) as plain arrays."""
    pca = model.named_steps["features"].named_transformers_["pca"]
    scale = model.named_steps["scale"]
    ridge = model.named_steps["ridge"]
    if getattr(pca, "whiten", False):
        raise ValueError("a whitened PCA is not the model N4 chose")
    return {
        "pca_mean": pca.mean_,
        "pca_components": pca.components_,
        "scale_mean": scale.mean_,
        "scale_scale": scale.scale_,
        "ridge_coef": np.atleast_1d(ridge.coef_),
        "ridge_intercept": np.atleast_1d(ridge.intercept_),
    }


def forecast(arrays: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    """The pipeline's forecast from its arrays: x is the embedding's columns, then the count."""
    a = {
        k: np.asarray(v, dtype=np.float64)
        for k, v in arrays.items()
        if v.dtype.kind == "f"
    }
    x = np.asarray(x, dtype=np.float64)
    embedding, count = x[:, :-1], x[:, -1:]
    components = (embedding - a["pca_mean"]) @ a["pca_components"].T
    features = np.hstack([components, count])
    scaled = (features - a["scale_mean"]) / a["scale_scale"]
    return scaled @ a["ridge_coef"] + a["ridge_intercept"][0]


TOLERANCE = 1e-5  # of the forecasts' standard deviation


def disagreement(arrays: dict[str, np.ndarray], model, x: np.ndarray) -> float:
    """The largest gap between the arrays' forecasts and the model's, over the standard
    deviation of the model's. The model computes in float32 and the arrays in float64, so
    they differ by rounding; the ranking that trades turns on the spread, far larger."""
    ours = forecast(arrays, x.astype(np.float64))
    theirs = model.predict(x)
    return float(np.abs(ours - theirs).max() / theirs.std())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def freeze(harness, efe_model, weights: Path = WEIGHTS) -> str:
    """Fit the chosen variant on every training row, check the arrays, write them once."""
    if weights.exists():
        raise SystemExit(
            f"{weights} exists: the challenger is frozen once, never again"
        )
    chosen = json.loads((harness.VARIANTS / "N4_chosen.json").read_text())
    levers = {lever: chosen[lever] for lever in harness.N4_LEVERS}
    data = harness.n4_year(
        TRAIN_THROUGH, harness.next_day_returns(), chosen["features"], levers
    )
    days = data.train["day"]
    print(
        f"fitting on {len(data.train_y):,} member-days, {days.min()} to {days.max()} ..."
    )
    model = efe_model.make(chosen["components"], chosen["penalty"])().fit(
        data.train_x, data.train_y
    )
    arrays = freeze_arrays(model)
    gap = disagreement(arrays, model, data.train_x)
    if gap > TOLERANCE:
        raise SystemExit(
            f"the arrays' forecasts differ from the fitted model's by {gap:.2e} of their "
            "spread; nothing written"
        )
    print(f"the arrays give the model's forecasts to {gap:.1e} of their spread")
    weights.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        weights,
        **arrays,
        trained_from=np.array(str(days.min())),
        trained_to=np.array(str(days.max())),
        settings=np.array(json.dumps(chosen, sort_keys=True)),
    )
    digest = sha256(weights)
    print(f"frozen: {weights}, trained {days.min()} to {days.max()}")
    print(f"weights_sha256: {digest}")
    return digest


def load(weights: Path = WEIGHTS) -> dict[str, np.ndarray]:
    with np.load(weights) as stored:
        return {k: stored[k] for k in stored.files}


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    import efe_model
    import harness

    if sys.argv[1:] != ["freeze"]:
        raise SystemExit("usage: forward.py freeze")
    freeze(harness, efe_model)
