"""Does pit.py compute PIT-4B? Checks on a downloaded checkpoint, before N3 uses it.

1. The weights: every name pit.py reads, with its shape, and nothing else; the output layer
   the same as the embedding table (they are tied).
2. A plain reference: the published computation written again in float32 on the processor,
   with explicit attention, one block at a time from the file. pit.py's states on the GPU
   (bfloat16 products, float32 stream) are compared with it on real headlines; so are the
   states of a model run wholly in bfloat16, as the repository's own example loads it.
3. Language: the next-token loss on sentences written for this check and on headlines of
   the year after the checkpoint, against 10.8 nats for a model that knows nothing; and
   greedy completions of a few prompts. Text goes in as N3 reads a headline and as the
   model card does: its tokens alone, with no token before them.

Headlines are read from disk but never printed: they are licensed. Usage:
pit_check.py CHECKPOINT_DIR NEWS_DIR YEAR   (data/llmenv/bin/python)
"""

import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).parent))
from pit import (
    NAMES,
    SHAPE,
    HiddenStates,
    check,
    embeddings,
    load,
    norm,
    rotary,
    rotate,
)

HERE, NEWS, YEAR = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
WEIGHTS = HERE / "model.safetensors"
torch.manual_seed(0)
check(json.loads((HERE / "config.json").read_text()), SHAPE)
tokenizer = Tokenizer.from_file(str(HERE / "tokenizer.json"))

print("1. The weights")
expected = {"transformer.wte.weight", "lm_head.weight"} | {
    name.format(i=i) for i in range(SHAPE.blocks) for name in NAMES.values()
}
with safe_open(str(WEIGHTS), framework="pt") as f:
    keys = set(f.keys())
    tied = torch.equal(
        f.get_tensor("lm_head.weight"), f.get_tensor("transformer.wte.weight")
    )
print(
    f"   {len(keys)} tensors; missing {sorted(expected - keys)}, extra {sorted(keys - expected)}"
)
print(f"   output layer equal to the embedding table: {tied}")


def reference(ids: list[int]) -> torch.Tensor:
    """Final states of one headline, float32 throughout, written apart from pit.py."""
    d, half, t = SHAPE.head, SHAPE.head // 2, len(ids)
    angle = torch.arange(t, dtype=torch.float32)[:, None] / SHAPE.theta ** (
        torch.arange(0, d, 2, dtype=torch.float32) / d
    )
    cos, sin = angle.cos()[:, None], angle.sin()[:, None]

    def rms(v):
        return v * torch.rsqrt((v * v).mean(-1, keepdim=True) + SHAPE.eps)

    def turn(v):
        first, second = v[..., :half], v[..., half:]
        return torch.cat([first * cos + second * sin, second * cos - first * sin], -1)

    future = ~torch.ones(t, t, dtype=torch.bool).tril()
    with safe_open(str(WEIGHTS), framework="pt") as f:
        x = f.get_tensor("transformer.wte.weight")[torch.tensor(ids)]
        for i in range(SHAPE.blocks):
            w = {k: f.get_tensor(name.format(i=i)) for k, name in NAMES.items()}
            h = rms(x)
            q, k, v = (
                (h @ w[n].T).view(t, SHAPE.heads, d) for n in ("c_q", "c_k", "c_v")
            )
            q, k = turn(rms(q)), turn(rms(k))
            scores = torch.einsum("qhd,khd->hqk", q, k) / math.sqrt(d)
            attention = scores.masked_fill(future, float("-inf")).softmax(-1)
            x = (
                x
                + torch.einsum("hqk,khd->qhd", attention, v).reshape(t, -1)
                @ w["c_proj"].T
            )
            x = x + F.relu(rms(x) @ w["fc"].T).square() @ w["proj"].T
    return rms(x)


@torch.no_grad()
def all_bf16(ids: torch.Tensor) -> torch.Tensor:
    """The model run wholly in bfloat16, the stream too, as the repository's example loads it."""
    b, t = ids.shape
    x = wte[ids].bfloat16().cuda()
    cos, sin = rotary(t, SHAPE.head, "cuda", SHAPE.theta)
    for weights in blocks:
        w = {k: v.cuda() for k, v in weights.items()}
        h = norm(x, SHAPE)
        q, k, v = (
            (h @ w[n].T).view(b, t, SHAPE.heads, SHAPE.head)
            for n in ("c_q", "c_k", "c_v")
        )
        q, k = rotate(norm(q, SHAPE), cos, sin), rotate(norm(k, SHAPE), cos, sin)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )
        x = x + y.transpose(1, 2).reshape(b, t, -1) @ w["c_proj"].T
        x = x + F.relu(norm(x, SHAPE) @ w["fc"].T).square() @ w["proj"].T
    return norm(x, SHAPE).float()


def headlines(year: int, n: int) -> list[str]:
    out = []
    for path in sorted(NEWS.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            a = json.loads(line)
            if a["created_at"].startswith(str(year)) and len(a["headline"].split()) > 2:
                out.append(a["headline"])
    g = torch.Generator().manual_seed(year)
    return [out[i] for i in torch.randperm(len(out), generator=g)[:n].tolist()]


sample = headlines(YEAR, 256)
wte, blocks = load(WEIGHTS)
states = HiddenStates(wte, blocks)

print("2. Against the float32 reference, 12 headlines of", YEAR)
encoded = [e.ids[:64] for e in tokenizer.encode_batch(sample[:12])]
lengths = torch.tensor([len(e) for e in encoded], device="cuda")
batch = torch.zeros((len(encoded), int(lengths.max())), dtype=torch.long)
for r, e in enumerate(encoded):
    batch[r, : len(e)] = torch.tensor(e)
last, mean = embeddings(states(batch), lengths)
plain = {"bf16": [], "ours": []}
for r, e in enumerate(encoded):
    ref = reference(e)
    ref_last, ref_mean = ref[-1], ref.mean(0)
    alone = all_bf16(torch.tensor([e]))[0]
    for label, (a, m) in {
        "ours": (last[r].cpu(), mean[r].cpu()),
        "bf16": (alone[-1].cpu(), alone.mean(0).cpu()),
    }.items():
        plain[label].append(
            (
                F.cosine_similarity(a, ref_last, dim=0).item(),
                F.cosine_similarity(m, ref_mean, dim=0).item(),
            )
        )
for label, rows in plain.items():
    lasts, means = zip(*rows, strict=True)
    print(
        f"   {label}: cosine with the reference, last token min {min(lasts):.5f} "
        f"mean {sum(lasts) / len(lasts):.5f}; mean of tokens min {min(means):.5f}"
    )


def loss(texts: list[str]) -> float:
    total, count = 0.0, 0
    for text in texts:
        ids = tokenizer.encode(text).ids[
            :64
        ]  # as N3 reads a headline: no token before it
        x = torch.tensor([ids])
        logits = states.logits(x)[0, :-1]
        total += F.cross_entropy(logits, x[0, 1:].cuda(), reduction="sum").item()
        count += len(ids) - 1
    return total / count


written = [
    "The United States declared its independence from Great Britain in 1776.",
    "Water boils at one hundred degrees Celsius at sea level.",
    "The company reported quarterly earnings that beat analysts' expectations.",
    "Shares of the bank fell after the Federal Reserve left interest rates unchanged.",
    "The stock market closed higher on Friday as oil prices rose.",
]
print("3. Language")
print(
    f"   loss on written sentences: {loss(written):.2f} nats a token (knowing nothing: 10.83)"
)
print(f"   loss on 256 headlines of {YEAR}: {loss(sample):.2f} nats a token")
for prompt in [
    "The capital of France is",
    "Apple Inc. is a company that makes",
    "The Federal Reserve raised interest rates, and stocks",
]:
    ids = tokenizer.encode(prompt).ids
    start = len(ids)
    for _ in range(8):
        ids.append(int(states.logits(torch.tensor([ids]))[0, -1].argmax()))
    print(f"   {prompt!r} -> {tokenizer.decode(ids[start:])!r}")
