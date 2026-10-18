"""Tests for N3's model copy and batching (pit.py); members.py's are in tests/.

They need PyTorch: data/llmenv/bin/python -m pytest research/news_reader/test_pit.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).parent))
from pit import (
    NAMES,
    HiddenStates,
    Shape,
    batches,
    check,
    embeddings,
    load,
    rotary,
    rotate,
)

TINY = Shape(vocab=50, width=16, heads=2, blocks=2)


def tiny_weights(seed=0):
    """A float32 embedding table and bfloat16 blocks, as load() returns them."""
    g = torch.Generator().manual_seed(seed)
    w = TINY.width

    def r(*shape):
        return (torch.randn(*shape, generator=g) * 0.2).to(torch.bfloat16)

    wte = torch.randn(TINY.vocab, w, generator=g)
    blocks = [
        {
            "c_q": r(w, w),
            "c_k": r(w, w),
            "c_v": r(w, w),
            "c_proj": r(w, w),
            "fc": r(4 * w, w),
            "proj": r(w, 4 * w),
        }
        for _ in range(TINY.blocks)
    ]
    return wte, blocks


def reference(ids, wte, blocks, shape):
    """The published computation, written plainly in float64, to check the copy against."""
    d, half, t = shape.head, shape.head // 2, ids.shape[1]
    angle = torch.arange(t, dtype=torch.float64)[:, None] / shape.theta ** (
        torch.arange(0, d, 2, dtype=torch.float64) / d
    )
    cos, sin = angle.cos()[None, :, None], angle.sin()[None, :, None]

    def rms(v):
        return v / torch.sqrt((v * v).mean(-1, keepdim=True) + shape.eps)

    def turn(v):
        first, second = v[..., :half], v[..., half:]
        return torch.cat([first * cos + second * sin, second * cos - first * sin], -1)

    future = ~torch.ones(t, t, dtype=torch.bool).tril()
    x = wte.double()[ids]
    for weights in blocks:
        w = {k: v.double() for k, v in weights.items()}
        h = rms(x)
        q, k, v = (
            (h @ w[n].T).unflatten(-1, (shape.heads, d)) for n in ("c_q", "c_k", "c_v")
        )
        q, k = turn(rms(q)), turn(rms(k))
        scores = torch.einsum("bqhd,bkhd->bhqk", q, k) / d**0.5
        attention = scores.masked_fill(future, float("-inf")).softmax(-1)
        x = x + torch.einsum("bhqk,bkhd->bqhd", attention, v).flatten(2) @ w["c_proj"].T
        x = x + torch.relu(rms(x) @ w["fc"].T) ** 2 @ w["proj"].T
    return rms(x)


def test_batches_stay_within_the_budget_and_take_every_row_once():
    lengths = np.array([5, 1, 9, 3, 30, 7, 2])
    order = np.argsort(lengths, kind="stable")
    groups = batches(order, lengths, budget=12)
    assert sorted(np.concatenate(groups).tolist()) == list(range(7))
    for g in groups:
        assert len(g) * lengths[g].max() <= 12 or len(g) == 1
    assert [lengths[g].tolist() for g in groups] == [[1, 2, 3], [5], [7], [9], [30]]
    # A batch exactly at the budget is allowed.
    even = np.array([3, 3, 3, 3, 5])
    assert [even[g].tolist() for g in batches(np.arange(5), even, 12)] == [
        [3, 3, 3, 3],
        [5],
    ]


def test_embeddings_read_the_last_real_token_and_the_mean_of_the_real_ones():
    states = torch.arange(2 * 3 * 2, dtype=torch.float32).view(2, 3, 2)
    last, mean = embeddings(states, torch.tensor([2, 3]))
    assert last.tolist() == [[2.0, 3.0], [10.0, 11.0]]
    assert mean.tolist() == [[1.0, 2.0], [8.0, 9.0]]


def test_padding_on_the_right_changes_no_real_token():
    # The published model ignores the attention mask; causal attention cannot look ahead,
    # so a headline read alone or beside a longer one gets the same embeddings.
    wte, blocks = tiny_weights()
    states = HiddenStates(wte, blocks, TINY, device="cpu")
    short, long_ = [3, 7, 11], [5, 9, 2, 8, 1, 4]
    alone = states(torch.tensor([short]))
    batch = torch.zeros((2, 6), dtype=torch.long)
    batch[0, :3] = torch.tensor(short)
    batch[1] = torch.tensor(long_)
    together = states(batch)
    assert torch.equal(alone[0], together[0, :3])
    a1, m1 = embeddings(alone, torch.tensor([3]))
    a2, m2 = embeddings(together, torch.tensor([3, 6]))
    assert torch.equal(a1[0], a2[0]) and torch.allclose(m1[0], m2[0])


def test_the_rotation_keeps_lengths_and_leaves_the_first_position_alone():
    x = torch.randn(1, 5, 2, 8).to(torch.bfloat16)
    cos, sin = rotary(5, 8, "cpu")
    y = rotate(x, cos, sin)
    assert torch.equal(y[:, 0], x[:, 0])
    assert torch.allclose(y.float().norm(dim=-1), x.float().norm(dim=-1), rtol=2e-2)


def test_a_state_depends_only_on_the_tokens_before_it():
    wte, blocks = tiny_weights(1)
    states = HiddenStates(wte, blocks, TINY, device="cpu")
    a = states(torch.tensor([[4, 8, 15, 16]]))
    b = states(torch.tensor([[4, 8, 15, 23]]))
    assert torch.equal(a[0, :3], b[0, :3]) and not torch.equal(a[0, 3], b[0, 3])


def test_the_copy_computes_the_published_model():
    wte, blocks = tiny_weights(3)
    ids = torch.randint(
        0, TINY.vocab, (3, 11), generator=torch.Generator().manual_seed(4)
    )
    ours = HiddenStates(wte, blocks, TINY, device="cpu")(ids)
    plain = reference(ids, wte, blocks, TINY)
    assert ours.dtype == torch.float32
    cosine = torch.nn.functional.cosine_similarity(ours.double(), plain, dim=-1)
    assert cosine.min() > 0.999  # bfloat16 products; a wrong wire gives far less
    size = (ours.double() - plain).norm(dim=-1) / plain.norm(dim=-1)
    assert size.max() < 0.05  # and the same length, which cosines alone would not see


def test_the_norms_use_the_pinned_epsilon():
    wte, blocks = tiny_weights(5)
    ids = torch.tensor([[1, 2, 3, 4]])
    pinned = HiddenStates(wte, blocks, TINY, device="cpu")(ids)
    loose = HiddenStates(wte, blocks, Shape(50, 16, 2, 2, eps=1.0), device="cpu")(ids)
    assert not torch.allclose(pinned, loose, atol=1e-2)


PUBLISHED = {
    "model_type": "pit",
    "vocab_size": 50304,
    "n_embd": 4096,
    "n_head": 32,
    "n_layer": 20,
    "rope_theta": 10000.0,
    "rms_norm_eps": 1.1920928955078125e-07,  # as PIT-4B-201511's config.json has it
    "tie_word_embeddings": True,
}


def test_a_checkpoint_must_have_the_architecture_copied_here():
    check(PUBLISHED, Shape())
    for key, value in [
        ("rms_norm_eps", None),
        ("rope_theta", 500000.0),
        ("n_layer", 24),
        ("tie_word_embeddings", False),
        ("model_type", "llama"),
    ]:
        with pytest.raises(ValueError, match="not the architecture"):
            check({**PUBLISHED, key: value}, Shape())


def test_weights_load_by_the_published_names(tmp_path):
    from safetensors.torch import save_file

    wte, blocks = tiny_weights(2)
    tensors = {"transformer.wte.weight": wte.float()}
    for i, block in enumerate(blocks):
        for short, name in NAMES.items():
            tensors[name.format(i=i)] = block[short].float()
    save_file(tensors, tmp_path / "model.safetensors")
    got_wte, got_blocks = load(tmp_path / "model.safetensors", TINY)
    assert got_wte.dtype == torch.float32 and torch.equal(got_wte, wte)
    assert all(
        torch.equal(got_blocks[i][k], blocks[i][k]) for i in range(2) for k in NAMES
    )
