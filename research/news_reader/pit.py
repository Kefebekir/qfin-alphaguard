"""PIT-4B, as the project runs it: the published architecture, rewritten, with no remote code.

The point-in-time models of Kelly, Malamud, Schwab and Xu ("Scaling Point-in-Time Language
Models", NeurIPS 2026; Apache-2.0) load on Hugging Face with `trust_remote_code`, which would
run the repository's modeling_pit.py. That file was read in full on 18 October 2026, at
commit 93c8f9a5 of Diamegs/PIT-4B-201511: plain PyTorch, a decoder-only transformer. This
module is the same computation, written here, so nothing from the repository runs; only its
weights (safetensors) and config are read.

The architecture, as published:
- token embeddings (vocabulary 50,304, width 4,096), tied to the output layer;
- 20 blocks, each x + attention(rms_norm(x)), then x + mlp(rms_norm(x));
- attention: 32 heads; queries and keys RMS-normed per head, then rotated (RoPE, base
  10,000); causal scaled dot-product attention;
- mlp: 4 times wider, the square of the ReLU;
- a final RMS norm, whose output the authors name as the one to embed. No norm has a weight,
  no layer a bias. Every norm's epsilon is the config's, 2^-23: the repository pins it
  because torch's default changed between versions, which moves the output by about 24%.

The precision is the training's: float32 weights under bfloat16 autocast. So the residual
stream and the norms of it are float32, while the products, the per-head norms, the rotation
(cos and sin in bfloat16, as the training code had them) and attention run in bfloat16.
Weights in bfloat16 are what autocast multiplies by; the embedding table stays float32.

With headlines padded on the right no padding mask is needed: causal attention keeps every
real token from seeing the padding after it, so its state is the one it would have alone.

The weights take 8.9 GB in bfloat16, more than an 8 GB GPU holds with the activations, so
`HiddenStates` keeps them in the computer's memory and moves one block at a time to the GPU.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Shape:
    vocab: int = 50_304
    width: int = 4_096
    heads: int = 32
    blocks: int = 20
    theta: float = 10_000.0
    eps: float = 2**-23  # float32's machine epsilon, as each published config pins it

    @property
    def head(self) -> int:
        return self.width // self.heads


def check(config: dict, shape: Shape) -> None:
    """Refuse a checkpoint whose published config is not the architecture copied here."""
    published = {
        "model_type": config.get("model_type"),
        "vocab": config.get("vocab_size"),
        "width": config.get("n_embd"),
        "heads": config.get("n_head"),
        "blocks": config.get("n_layer"),
        "theta": config.get("rope_theta"),
        "eps": config.get("rms_norm_eps"),
        "tied": config.get("tie_word_embeddings"),
    }
    ours = {
        "model_type": "pit",
        "vocab": shape.vocab,
        "width": shape.width,
        "heads": shape.heads,
        "blocks": shape.blocks,
        "theta": shape.theta,
        "eps": shape.eps,
        "tied": True,
    }
    differ = {k: (published[k], ours[k]) for k in ours if published[k] != ours[k]}
    if differ:
        raise ValueError(
            f"not the architecture copied here (published, ours): {differ}"
        )


def rotary(length: int, dim: int, device, theta: float = 10_000.0):
    """cos and sin for positions 0..length-1, in bfloat16, as the model trained with them."""
    inv = 1.0 / (
        theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
    )
    freqs = torch.outer(torch.arange(length, device=device, dtype=torch.float32), inv)
    cos, sin = freqs.cos().bfloat16(), freqs.sin().bfloat16()
    return cos[None, :, None, :], sin[None, :, None, :]


def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[3] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], dim=3).type_as(x)


def norm(x: torch.Tensor, shape: Shape) -> torch.Tensor:
    return F.rms_norm(x, (x.shape[-1],), eps=shape.eps)


def block(x: torch.Tensor, w: dict[str, torch.Tensor], shape: Shape) -> torch.Tensor:
    """One block on the float32 stream `x`; `w` holds its bfloat16 weights: c_q, c_k, c_v,
    c_proj, fc and proj."""
    b, t, _ = x.shape
    h = norm(x, shape).bfloat16()
    q = (h @ w["c_q"].T).view(b, t, shape.heads, shape.head)
    k = (h @ w["c_k"].T).view(b, t, shape.heads, shape.head)
    v = (h @ w["c_v"].T).view(b, t, shape.heads, shape.head)
    cos, sin = rotary(t, shape.head, x.device, shape.theta)
    q = rotate(norm(q, shape), cos, sin)
    k = rotate(norm(k, shape), cos, sin)
    y = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
    )
    x = x + (y.transpose(1, 2).reshape(b, t, shape.width) @ w["c_proj"].T).float()
    h = norm(x, shape).bfloat16()
    return x + (F.relu(h @ w["fc"].T).square() @ w["proj"].T).float()


# The published weights' names, block by block.
NAMES = {
    "c_q": "transformer.h.{i}.attn.c_q.weight",
    "c_k": "transformer.h.{i}.attn.c_k.weight",
    "c_v": "transformer.h.{i}.attn.c_v.weight",
    "c_proj": "transformer.h.{i}.attn.c_proj.weight",
    "fc": "transformer.h.{i}.mlp.c_fc.weight",
    "proj": "transformer.h.{i}.mlp.c_proj.weight",
}


SHAPE = Shape()


def load(
    path: Path, shape: Shape = SHAPE
) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
    """The embedding table (float32) and each block's weights (bfloat16, pinned), in the
    computer's memory."""
    from safetensors import safe_open

    blocks: list[dict[str, torch.Tensor]] = [{} for _ in range(shape.blocks)]
    with safe_open(str(path), framework="pt") as f:
        keys = set(f.keys())
        table = (
            "transformer.wte.weight"
            if "transformer.wte.weight" in keys
            else "lm_head.weight"
        )
        wte = f.get_tensor(table).float()
        if wte.shape != (shape.vocab, shape.width):
            raise ValueError(
                f"{table} is {tuple(wte.shape)}, not the shape copied here"
            )
        for i in range(shape.blocks):
            for short, name in NAMES.items():
                blocks[i][short] = (
                    f.get_tensor(name.format(i=i)).to(torch.bfloat16).pin_memory()
                )
    return wte, blocks


class HiddenStates:
    """The final hidden states of PIT-4B for a batch of token ids, one block on the GPU at a time."""

    def __init__(
        self,
        wte: torch.Tensor,
        blocks: list[dict[str, torch.Tensor]],
        shape: Shape = SHAPE,
        device="cuda",
    ):
        self.wte, self.blocks, self.shape, self.device = wte, blocks, shape, device

    @torch.no_grad()
    def __call__(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: [batch, tokens], padded on the right; returns float32 [batch, tokens, width]."""
        x = self.wte[ids].to(self.device)
        for weights in self.blocks:
            on_gpu = {
                k: v.to(self.device, non_blocking=True) for k, v in weights.items()
            }
            x = block(x, on_gpu, self.shape)
            del on_gpu
        return norm(x, self.shape)

    @torch.no_grad()
    def logits(self, ids: torch.Tensor) -> torch.Tensor:
        """Next-token scores, through the tied output layer: for checking the copy."""
        table = self.wte.to(self.device, torch.bfloat16)
        return (self(ids).bfloat16() @ table.T).float()


def embeddings(
    states: torch.Tensor, lengths: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Each row's state at its last real token, and the mean over its real tokens."""
    rows = torch.arange(states.shape[0], device=states.device)
    last = states[rows, lengths - 1]
    mask = (
        torch.arange(states.shape[1], device=states.device)[None, :] < lengths[:, None]
    )
    mean = (states.float() * mask[..., None]).sum(1) / lengths[:, None]
    return last.float(), mean


def batches(order, lengths, budget: int):
    """Rows in `order` (ascending length), grouped so that rows times the longest stays
    within `budget` tokens; a row longer than the budget goes alone."""
    out, start = [], 0
    while start < len(order):
        end = start + 1
        while end < len(order) and (end + 1 - start) * lengths[order[end]] <= budget:
            end += 1
        out.append(order[start:end])
        start = end
    return out
