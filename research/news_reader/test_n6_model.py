"""N6's model (n6_model.py). They need PyTorch:
data/llmenv/bin/python -m pytest research/news_reader/test_n6_model.py
"""

import sys
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).parent))
import n6_harness as nh
import n6_model as nm

LATENT = 8
SMALL = nm.Settings(dim=8, attention=4, context=8, hidden=8, dropout=0.0, epochs=1)


def world(days=60, per_day=30, seed=0, signal=0.0, first=date(2018, 1, 1)):
    """Arrays for `days` days of `per_day` member-days, 0 to 4 headlines each, and a store
    of headlines that, like a language model's, lie near a few directions (LATENT, of
    falling spread). With `signal`, the third of them tells the member-day's target."""
    g = np.random.default_rng(seed)
    n = days * per_day
    heads = np.full((n, nh.MAX_HEADLINES), -1, dtype=np.int32)
    counts = g.integers(0, 5, n)
    y = g.normal(size=n).astype(np.float32)
    spread = np.linspace(3.0, 1.0, LATENT)
    latent = g.normal(size=(int(counts.sum()) + 1, LATENT)) * spread
    row = 0
    for i, c in enumerate(counts):
        heads[i, :c] = np.arange(row, row + c)
        if signal and c:
            latent[row : row + c, 2] = signal * y[i] * spread[2]
        row += c
    basis = g.normal(size=(LATENT, nm.WIDTH)).astype(np.float32)
    store = latent.astype(np.float32) @ basis
    store += 0.5 * g.standard_normal(store.shape, dtype=np.float32)
    store = store.astype(np.float16)
    day = np.repeat(np.arange(days), per_day)
    dates = [first + timedelta(days=int(d)) for d in day]
    a = nh.Arrays(
        days=pl.DataFrame(
            {"day": dates, "ticker": [f"T{i % per_day}" for i in range(n)]}
        ).with_columns(
            next_return=pl.Series(y * 0.01), vol=pl.lit(0.01), sector=pl.lit(0)
        ),
        day=day,
        member=g.normal(size=(n, len(nh.NUMBERS))).astype(np.float32),
        market=np.repeat(g.normal(size=(days, len(nh.MARKET))), per_day, 0).astype(
            np.float32
        ),
        events=np.zeros((n, len(nh.EVENTS)), np.float32),
        weekday=day % 5,
        sector=g.integers(0, 10, n),
        y=y,
        count=np.log1p(counts).astype(np.float32),
        heads=heads,
        head_info=g.normal(size=(n, nh.MAX_HEADLINES, 3)).astype(np.float32),
    )
    return a, torch.as_tensor(store)


def forward(model, a, store, rows):
    x = nm.inputs(a, np.ones(len(a.day), bool), "cpu")
    model.eval()
    with torch.no_grad():
        return model(store, nm.take(x, torch.as_tensor(rows)))[0].numpy()


def model_for(a, s=SMALL, seed=0):
    torch.manual_seed(seed)
    return nm.N6(s, a.member.shape[1], a.market.shape[1], a.events.shape[1])


def test_the_recommended_model_fits_the_hardware_cap():
    a, _ = world(days=2)
    model = model_for(a, nm.Settings())
    assert nm.WIDTH * 32 < nm.parameters(model) < nm.MAX_PARAMETERS
    assert nm.parameters(model_for(a, nm.Settings(dim=256))) < nm.MAX_PARAMETERS
    big = nm.Settings(dim=300)  # 4,096 x 300 = 1.23 million: above amendment 8's cap
    with pytest.raises(ValueError, match="amendment 8"):
        nm.fit(a, np.ones(len(a.day), bool), big, 0, world(days=2)[1])


def test_padding_and_the_order_of_headlines_do_not_change_a_forecast():
    a, store = world(days=2)
    model = model_for(a)
    rows = np.flatnonzero(a.count > np.log1p(2))[:5]
    before = forward(model, a, store, rows)
    shuffled = a.heads.copy()
    info = a.head_info.copy()
    for i in rows:
        k = int((a.heads[i] >= 0).sum())
        order = np.random.default_rng(i).permutation(k)
        shuffled[i, :k] = a.heads[i, order]
        info[i, :k] = a.head_info[i, order]
    info[rows, -1] = 99.0  # the padding's info is never read
    after = forward(model, replace(a, heads=shuffled, head_info=info), store, rows)
    np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-6)


def test_a_member_day_without_headlines_has_a_finite_forecast():
    a, store = world(days=2)
    quiet = nh.silent(a)
    out = forward(model_for(a), quiet, store, np.arange(10))
    assert np.isfinite(out).all()


def test_the_sharpe_loss_falls_as_the_forecasts_line_up_with_the_returns():
    g = np.random.default_rng(0)
    ret = torch.as_tensor(g.normal(size=200), dtype=torch.float32)
    day = torch.as_tensor(np.repeat(np.arange(10), 20))
    noise = torch.as_tensor(g.normal(size=200), dtype=torch.float32)
    good = nm.sharpe_loss(ret + 0.1 * noise, ret, day)
    bad = nm.sharpe_loss(-ret + 0.1 * noise, ret, day)
    assert good < 0 < bad
    shifted = nm.sharpe_loss(
        ret + 0.1 * noise + 5.0, ret, day
    )  # a day's level is no bet
    assert shifted == pytest.approx(good.item(), rel=1e-4)


def test_whole_days_go_together_into_batches():
    day = np.repeat(np.arange(7), 3)
    rows = np.ones(len(day), bool)
    rows[[0, 4]] = False
    groups = nm.batches(day, rows, 2, np.random.default_rng(0))
    assert sorted(np.concatenate(groups).tolist()) == np.flatnonzero(rows).tolist()
    for g in groups:
        assert len(np.unique(day[g])) <= 2
        assert all(
            set(np.flatnonzero(rows & (day == d))) <= set(g) for d in np.unique(day[g])
        )


@pytest.mark.parametrize("start", nm.STARTS)
@pytest.mark.parametrize("loss", nm.LOSSES)
def test_the_model_learns_a_signal_in_the_headlines_and_not_without_it(loss, start):
    a, store = world(days=200, signal=1.0)
    s = nm.Settings(
        dim=8,
        attention=4,
        context=8,
        hidden=16,
        dropout=0.0,
        epochs=8,
        days_per_batch=8,
        loss=loss,
        lr=3e-3,
        start=start,
    )
    trained = a.day < 150
    validated = (a.day >= 150) & (a.count > 0)
    forecast = nm.ensemble(a, trained, validated, s, [0], store)
    assert np.corrcoef(forecast, a.y[validated])[0, 1] > 0.8
    quiet = nm.ensemble(nh.silent(a), trained, validated, s, [0], store)
    assert abs(np.corrcoef(quiet, a.y[validated])[0, 1]) < 0.2


def test_the_pca_start_projects_on_the_first_components_at_unit_spread():
    a, store = world(days=20)
    rows = a.heads[a.heads >= 0]
    weight, bias = nm.pca_start(store, rows, 4, seed=0)
    x = store[torch.as_tensor(np.unique(rows)).long()].double().numpy()
    z = x @ weight.double().numpy().T + bias.double().numpy()
    np.testing.assert_allclose(z.mean(0), 0, atol=1e-4)
    np.testing.assert_allclose(np.cov(z.T), np.eye(4), atol=1e-3)
    _, _, vt = np.linalg.svd(x - x.mean(0), full_matrices=False)
    directions = weight.double().numpy()
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    np.testing.assert_allclose(np.abs((directions * vt[:4]).sum(1)), 1, atol=1e-4)


def test_a_seed_gives_the_same_model():
    a, store = world(days=10)
    rows = np.ones(len(a.day), bool)
    one = nm.ensemble(a, rows, rows, SMALL, [3], store)
    two = nm.ensemble(a, rows, rows, SMALL, [3], store)
    np.testing.assert_array_equal(one, two)


def test_settings_are_checked():
    with pytest.raises(ValueError, match="loss is one of"):
        nm.Settings(loss="mae")
    with pytest.raises(ValueError, match="at least 1"):
        nm.Settings(dim=0)
    with pytest.raises(ValueError, match="dropout"):
        nm.Settings(dropout=1.0)
    with pytest.raises(ValueError, match="start is one of"):
        nm.Settings(start="zeros")
    with pytest.raises(ValueError, match="inputs is one of"):
        nm.Settings(inputs="context")


def test_a_model_on_the_news_alone_reads_no_context():
    a, store = world(days=2)
    s = replace(SMALL, inputs="news")
    model = model_for(a, s)
    assert not hasattr(model, "context")
    assert nm.parameters(model) < nm.parameters(model_for(a))
    rows = np.arange(20)
    before = forward(model, a, store, rows)
    g = np.random.default_rng(9)
    other = replace(
        a,
        member=g.normal(size=a.member.shape).astype(np.float32),
        market=g.normal(size=a.market.shape).astype(np.float32),
        sector=(a.sector + 1) % 10,
        weekday=(a.weekday + 1) % 5,
    )
    np.testing.assert_array_equal(before, forward(model, other, store, rows))
    quiet = forward(model, nh.silent(a), store, rows)
    assert np.ptp(quiet) == 0  # nothing to read: one forecast for all


def test_the_news_alone_still_learns_the_headlines_signal():
    a, store = world(days=200, signal=1.0)
    s = nm.Settings(
        dim=8, attention=4, hidden=16, dropout=0.0, epochs=8, lr=3e-3, inputs="news"
    )
    trained = a.day < 150
    validated = (a.day >= 150) & (a.count > 0)
    forecast = nm.ensemble(a, trained, validated, s, [0], store)
    assert np.corrcoef(forecast, a.y[validated])[0, 1] > 0.8


def test_the_sharpe_loss_skips_a_batch_of_one_day():
    a, store = world(days=17)  # 17 days in batches of 8: the last batch is one day
    groups = nm.batches(a.day, np.ones(len(a.day), bool), 8, np.random.default_rng(0))
    assert sorted(len(np.unique(a.day[g])) for g in groups) == [1, 8, 8]
    one = torch.tensor([0.5, -0.5])
    assert torch.isnan(
        nm.sharpe_loss(one, one, torch.tensor([3, 3]))
    )  # why it is skipped
    s = replace(SMALL, loss="sharpe", days_per_batch=8, epochs=2)
    rows = np.ones(len(a.day), bool)
    forecast = nm.ensemble(a, rows, rows, s, [0], store)
    assert np.isfinite(forecast).all()
