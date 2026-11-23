"""The harness of Efe's network (research/kutbit/net_harness.py), with a stand-in network.

The tests that learn need PyTorch, so they run in data/llmenv and are skipped elsewhere:

    PYTHONPATH=src data/llmenv/bin/python -m pytest tests/test_kutbit_net_harness.py
"""

import importlib.util
import json
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from qfin_alphaguard.kutbit.features import SCALARS

HERE = Path(__file__).parents[1] / "research" / "kutbit"
sys.path.insert(0, str(HERE))


def loaded(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nh = loaded("kutbit_net_harness", HERE / "net_harness.py")
made = loaded(
    "kutbit_net_data_tests", Path(__file__).parent / "test_kutbit_net_data.py"
)
import costs
import net_data
import skill
from labels import HORIZONS


def quiet(*_):
    pass


@pytest.fixture(autouse=True)
def few_threads():
    """Small solves on many threads run a hundred times slower here; the skill test's own
    commands run on 8."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        yield
        return
    with threadpool_limits(4):
        yield


# --- without PyTorch ----------------------------------------------------------------------


def test_the_constants_are_decision_0027_s():
    assert nh.SETTINGS == ((128, 0.1), (128, 0.01), (512, 0.1), (512, 0.01))
    assert nh.SEEDS == (1, 2, 3, 4, 5) and nh.EPOCHS == 20
    assert nh.SCORED == (15, 30, 60) and nh.PROOF == 2.0 and nh.PLACEBO_T == 3.0
    # the speed check's estimate counts the same runs
    assert sorted({w for w, _ in nh.SETTINGS}) == list(net_data.WIDTHS)
    assert len(nh.SETTINGS) == len(net_data.WIDTHS) * net_data.SETTINGS_A_WIDTH
    assert len(nh.SEEDS) == net_data.SEEDS and nh.EPOCHS == net_data.EPOCHS


def test_shuffled_deals_each_minute_s_targets_between_its_instruments():
    rng = np.random.default_rng(0)
    day = np.repeat([700_000, 700_001], 600)
    minute = np.tile(np.repeat(np.arange(60), 10), 2)  # 10 instruments a minute
    z = rng.normal(size=(1200, 5))
    dealt = nh.shuffled(z, day, minute, seed=3)
    for d in (700_000, 700_001):
        for m in (0, 17, 59):
            at = (day == d) & (minute == m)
            assert sorted(dealt[at, 0]) == sorted(z[at, 0])  # the same targets, dealt
            assert np.array_equal(  # a row's horizons travel together
                np.sort(dealt[at], axis=0), np.sort(z[at], axis=0)
            )
    assert (dealt[:, 0] != z[:, 0]).mean() > 0.8
    assert np.array_equal(dealt, nh.shuffled(z, day, minute, seed=3))
    assert not np.array_equal(dealt, nh.shuffled(z, day, minute, seed=4))


def test_the_choice_keeps_the_best_and_a_tie_the_earlier():
    scores = np.full((4, 20), 0.001)
    scores[2, 6] = 0.004
    assert nh.kept(scores) == (2, 7)
    scores[1, 3] = 0.004  # an earlier setting ties
    assert nh.kept(scores) == (1, 4)
    scores[1, 3] = np.nan
    assert nh.kept(scores) == (2, 7)
    assert nh.kept(np.full((4, 20), np.nan)) == (0, 1)


def test_a_score_averages_each_day_s_correlation_over_the_horizons():
    rng = np.random.default_rng(1)
    day = np.repeat(np.arange(6), 200)
    z = rng.normal(size=(1200, 3))
    p = np.stack([z[:, 0], -z[:, 1], rng.normal(size=1200)], axis=1)
    expected = np.mean(
        [skill.daily_correlation(p[:, k], z[:, k], day)["mean"] for k in range(3)]
    )
    assert nh.score(p, z, day) == pytest.approx(expected)
    assert nh.score(p[:, :1], z[:, :1], day) == pytest.approx(1.0)


def test_the_margin_is_day_by_day_and_averaged_over_15_to_60_minutes():
    rng = np.random.default_rng(2)
    day = np.repeat(np.arange(40), 300)
    z = rng.normal(size=(12_000, len(HORIZONS)))
    noise = rng.normal(size=z.shape)
    better = z + 3 * noise
    worse = z + 6 * noise
    found = nh.margin(better, worse, z, day)
    assert found["mean"] > 0 and found["t"] > 2 and found["passed"]
    assert found["n"] == 40
    same = nh.margin(better, better, z, day)
    assert same["mean"] == 0 and not same["passed"]
    # ahead by 2 on two days, behind by 2 on the third: a lead, but no proof
    three = np.repeat([0, 1, 2], 300)
    lead = np.where((three < 2)[:, None], z[:900], -z[:900])
    small = nh.margin(lead, -lead, z[:900], three)
    assert small["mean"] > 0 and abs(small["t"]) < 2 and not small["passed"]
    # by hand: each day's correlations about the means of all the rows, 15-60 minutes
    by_day = []
    for d in range(40):
        diffs = []
        for j in nh.SCORED_AT:
            _, a = skill.grouped_correlations(
                better[:, j], z[:, j], day, about_all=True
            )
            _, b = skill.grouped_correlations(worse[:, j], z[:, j], day, about_all=True)
            diffs.append(a[d] - b[d])
        by_day.append(np.mean(diffs))
    assert found["mean"] == pytest.approx(np.mean(by_day))
    # a horizon outside 15-60 minutes does not count
    planted = better.copy()
    planted[:, HORIZONS.index(1)] = z[:, HORIZONS.index(1)]
    assert nh.margin(planted, better, z, day)["mean"] == pytest.approx(0)


def made_up_costs(data: Path) -> None:
    """Moments of the cost study for AAA, BBB, QQQ and SPY, one in each part of the day."""
    spreads = {
        "AAA": (6.0, 2.0, 1.5),
        "BBB": (4.0, 3.0, 2.5),
        "QQQ": (0.8, 0.6, 0.5),
        "SPY": (0.5, 0.4, 0.3),
    }
    rows = []
    for ticker, by_part in spreads.items():
        for minute, spread in zip((1, 101, 381), by_part):
            half = 100 * spread / 2e4
            rows.append(
                {
                    "day": date(2016, 3, 1),
                    "minute": minute,
                    "ticker": ticker,
                    "spread": spread,
                    "bid": 100 - half,
                    "ask": 100 + half,
                    "bid_size": 5,
                    "ask_size": 5,
                    "quotes": 3,
                    "invalid": 0,
                    "symbol": ticker,
                    "window": costs.WINDOW,
                }
            )
    folder = data / "kutbit" / "costs" / "moments"
    folder.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, schema=costs.MOMENT_SCHEMA).write_parquet(
        folder / "2016-03-01.parquet"
    )


def test_a_signal_pays_its_instrument_s_cost_in_its_part_of_the_day_and_the_hedge_s(
    tmp_path,
):
    made_up_costs(tmp_path)
    (tmp_path / "kutbit" / "net").mkdir(parents=True)
    tickers = [
        "AAA",
        "QQQ",
        "SPY",
        "ZZZ",
    ]  # ZZZ, a stock the cost study did not measure
    (tmp_path / "kutbit" / "net" / "tickers.json").write_text(json.dumps(tickers))
    pl.DataFrame(
        {
            "ticker": ["AAA", "BBB", "QQQ", "SPY", "ZZZ"],
            "etf": [False, False, True, True, False],
        }
    ).write_parquet(tmp_path / "kutbit" / "days.parquet")
    scalars = np.zeros((5, len(SCALARS)), dtype=np.float32)
    scalars[:, SCALARS.index("beta")] = [1.0, -0.5, 0.0, 0.9, 1.0]
    table = {
        "instrument": np.array([0, 0, 3, 2, 1]),
        "minute": np.array([5, 200, 370, 29, 30]),
        "scalars": scalars,
    }
    found = nh.signal_costs(tmp_path, table)
    stocks_median = {
        "first": (6 + 4) / 2,
        "between": (2 + 3) / 2,
        "last": (1.5 + 2.5) / 2,
    }
    assert found == pytest.approx(
        [
            6.0 + 1.0 * 0.8,  # AAA, the first 30 minutes, with the hedge's
            2.0 + 0.5 * 0.6,  # AAA, the day between: a short beta hedges too
            stocks_median["last"],  # ZZZ, unmeasured: the median stock, no beta
            0.5 + 0.9 * 0.8,  # SPY, minute 29 is still in the first 30 minutes
            0.6 + 1.0 * 0.6,  # QQQ, minute 30 is the day between
        ]
    )


def test_own_cost_by_hand():
    p = np.array([2.0, -3.0, 0.1, 4.0])
    apart_bps = np.array([5.0, -1.0, 9.0, -2.0])
    cost = np.array([1.0, 2.0, 3.0, 4.0])
    day = np.array([1, 1, 2, 2])
    found = nh.own_cost(p, apart_bps, cost, day, threshold=1.0)
    # chosen: rows 0, 1 and 3, trading in the forecast's direction: +5, +1, -2
    assert found["cost"] == pytest.approx((1 + 2 + 4) / 3)
    assert found["own"] == pytest.approx(((5 - 1) + (1 - 2) + (-2 - 4)) / 3)
    assert found["twice"] == pytest.approx(((5 - 2) + (1 - 4) + (-2 - 8)) / 3)
    assert math.isfinite(found["own_t"])
    none = nh.own_cost(p, apart_bps, cost, day, threshold=10.0)
    assert math.isnan(none["cost"]) and math.isnan(none["own"])


def results_with(own: dict) -> dict:
    """A network's results whose strongest 1% earn `own` = {horizon: (mean, t)}."""
    return {
        "network": {
            str(h): {
                "all": {
                    "top0.01": {
                        "own": own.get(h, (0, 0))[0],
                        "own_t": own.get(h, (0, 0))[1],
                    }
                }
            }
            for h in HORIZONS
        }
    }


def test_the_gates_ask_the_ridge_the_shuffled_network_and_the_cost():
    rng = np.random.default_rng(4)
    day = np.repeat(np.arange(50), 200)
    z = rng.normal(size=(10_000, len(HORIZONS)))
    noise = rng.normal(size=z.shape)
    p = {"network": z + 2 * noise, "ridge": z + 5 * noise, "shuffled": noise}
    table = {"z": z, "day": day}
    found = nh.gates(p, table, results_with({30: (0.8, 2.4), 60: (1.5, 1.9)}))
    assert (
        found["over the ridge"]["passed"]
        and found["over the shuffled network"]["passed"]
    )
    assert found["its own cost"]["by_horizon"] == {"15": False, "30": True, "60": False}
    assert found["variant passed"] and found["bank and simulator next"]
    found = nh.gates(p, table, results_with({60: (1.5, 1.9), 5: (9.0, 9.0)}))
    assert found["variant passed"] and not found["bank and simulator next"]
    beaten = dict(p, shuffled=p["network"])
    found = nh.gates(beaten, table, results_with({30: (0.8, 2.4)}))
    assert found["over the ridge"]["passed"]
    assert not found["over the shuffled network"]["passed"]
    assert not found["variant passed"] and not found["bank and simulator next"]
    p["ridge"] = p["network"]
    found = nh.gates(p, table, results_with({30: (0.8, 2.4)}))
    assert not found["variant passed"] and not found["bank and simulator next"]


def test_each_measure_adds_the_strongest_after_their_own_cost():
    rng = np.random.default_rng(6)
    n = 2_000
    table = {
        "day": np.repeat(np.arange(10), n // 10),
        "minute": np.tile(np.arange(200), 10),
        "z": rng.normal(size=(n, len(HORIZONS))),
        "apart": rng.normal(0, 1e-3, size=(n, len(HORIZONS))),
        "etf": np.arange(n) % 4 == 0,
    }
    p = table["z"] + rng.normal(size=table["z"].shape)
    cost = rng.uniform(1, 3, n)
    limits = {h: skill.thresholds_of(p[:, j]) for j, h in enumerate(HORIZONS)}
    found = nh.measures_of(p, table, lambda h: limits[h], cost)
    j, s = HORIZONS.index(30), ~table["etf"]
    stocks = found["30"]["stocks"]
    expected = skill.measures(
        p[s, j],
        table["z"][s, j],
        table["apart"][s, j],
        table["day"][s],
        table["minute"][s],
        limits[30],
    )
    assert stocks["daily"] == expected["daily"]
    for share in skill.STRONGEST:
        top = stocks[f"top{share:g}"]
        assert top["gross"] == expected[f"top{share:g}"]["gross"]
        own = nh.own_cost(
            p[s, j],
            table["apart"][s, j] * 1e4,
            cost[s],
            table["day"][s],
            limits[30][share],
        )
        assert {k: top[k] for k in own} == own
    assert set(found) == {str(h) for h in HORIZONS}
    assert set(found["30"]) == {"all", "stocks", "etfs"}


def test_by_day_gives_each_day_s_rows():
    table = {"day": np.array([5, 5, 5, 6, 9, 9])}
    assert [r.tolist() for r in nh.by_day(table)] == [[0, 1, 2], [3], [4, 5]]


def test_runs_wait_for_their_turn(tmp_path):
    code = {"net.py": "a", "net_harness.py": "b"}
    assert nh.refusal(tmp_path, "choose", code) == "run the placebo first"
    record = {"passed": False, "code": code}
    (tmp_path / "placebo.json").write_text(json.dumps(record))
    assert "the placebo failed" in nh.refusal(tmp_path, "choose", code)
    (tmp_path / "placebo.json").write_text(json.dumps(record | {"passed": True}))
    assert nh.refusal(tmp_path, "choose", code) is None
    assert "net.py changed" in nh.refusal(tmp_path, "choose", code | {"net.py": "c"})
    assert nh.refusal(tmp_path, "measure", code) == "choose the setting on 2019 first"
    (tmp_path / "choice.json").write_text(json.dumps({"code": code | {"net.py": "c"}}))
    assert "the choice ran on other code" in nh.refusal(tmp_path, "measure", code)
    (tmp_path / "choice.json").write_text(json.dumps({"code": code}))
    assert nh.refusal(tmp_path, "measure", code) is None
    (tmp_path / "measure.json").write_text("{}")
    assert (
        nh.refusal(tmp_path, "measure", code)
        == "the measurement has run; it runs only once"
    )


def test_the_fingerprint_marks_a_file_not_written_yet(tmp_path):
    for name in nh.CODE[1:]:
        (tmp_path / name).write_text(name)
    found = nh.fingerprint(tmp_path)
    assert found["net.py"] is None and all(found[n] for n in nh.CODE[1:])
    (tmp_path / "net.py").write_text("efe")
    assert nh.fingerprint(tmp_path)["net.py"] is not None


def store_with_skill_runs(tmp_path: Path) -> Path:
    """The made-up rows of the data tests, built, with the skill test run on them."""
    data = made.store(tmp_path)
    for source in net_data.SOURCES:
        net_data.build(data, source, log=quiet)
        skill.write(data, skill.run(data, source, log=quiet))
    made_up_costs(data)
    tickers = json.loads((data / "kutbit" / "net" / "tickers.json").read_text())
    moments = pl.read_parquet(
        data / "kutbit" / "costs" / "moments" / "2016-03-01.parquet"
    )
    extra = moments.filter(pl.col("ticker") == "QQQ").with_columns(
        pl.lit("XLF").alias("ticker")
    )
    pl.concat([moments, extra]).write_parquet(
        data / "kutbit" / "costs" / "moments" / "2016-03-01.parquet"
    )
    assert "QQQ" in tickers
    return data


def test_the_ridge_fitted_again_gives_the_skill_test_s_numbers(tmp_path):
    data = store_with_skill_runs(tmp_path)
    for source in net_data.SOURCES:
        ridge = nh.ridge_again(data, source, log=quiet)
        measured = net_data.load(data, source, skill.MEASURE_YEARS)
        cost = nh.signal_costs(data, measured)
        results = {
            "ridge": nh.measures_of(
                ridge["forecast"], measured, lambda h, r=ridge: r["thresholds"][h], cost
            )
        }
        for i, name in enumerate(skill.PATTERNS):
            p = np.repeat(measured["patterns"][:, i : i + 1], len(HORIZONS), axis=1)
            results[name] = nh.measures_of(
                p, measured, lambda h, n=name, r=ridge: r["thresholds"][n], cost
            )
        assert nh.as_skill_test(data, source, results, ridge["choice"]) == []
        theirs = json.loads((data / "kutbit" / "skill" / f"{source}.json").read_text())
        mine = results["ridge"]["15"]["all"]["daily"]["mean"]
        assert mine == theirs["results"]["ridge"]["15"]["all"]["daily"]["mean"]
    # a number moved beyond rounding is found, and so is a missing run
    theirs["results"]["ridge"]["15"]["all"]["daily"]["mean"] *= 1.01
    (data / "kutbit" / "skill" / "random.json").write_text(json.dumps(theirs))
    found = nh.as_skill_test(data, "random", results, ridge["choice"])
    assert found and found[0].startswith("ridge/15/all/daily/mean")
    (data / "kutbit" / "skill" / "real.json").unlink()
    assert "missing" in nh.as_skill_test(data, "real", results, ridge["choice"])[0]


def test_rounding_is_not_a_difference_but_a_change_is():
    assert (
        nh.differences(
            {"a": 1.0, "b": {"c": float("nan")}},
            {"a": 1.0 + 1e-9, "b": {"c": float("nan")}},
        )
        == []
    )
    assert nh.differences({"a": 1.0}, {"a": 1.001}) == ["/a: 1.0 against 1.001"]
    assert nh.differences({"a": 1.0}, {}) == ["/a: 1.0 against None"]


def test_the_page_shows_each_run(tmp_path):
    result = {
        str(h): {
            g: {
                "daily": {"mean": 0.01, "t": 2.5},
                "across": {"mean": float("nan"), "t": float("nan")},
                "top0.01": {
                    "gross": 1.9,
                    "net": -0.1,
                    "t": 2.4,
                    "cost": 1.2,
                    "own": 0.7,
                    "own_t": 1.1,
                    "twice": -0.5,
                    "twice_t": -0.8,
                },
            }
            for g in ("all", "stocks", "etfs")
        }
        for h in HORIZONS
    }
    results = {name: result for name in nh.NAMES}
    margin = {"mean": 0.002, "t": 2.6, "n": 505, "passed": True}
    gates = {
        "over the ridge": margin,
        "over the shuffled network": margin | {"t": 1.2, "passed": False},
        "its own cost": {
            "by_horizon": {"15": False, "30": True, "60": False},
            "passed": True,
        },
        "variant passed": False,
        "bank and simulator next": False,
    }
    nh.write(
        tmp_path,
        "placebo",
        {
            "passed": True,
            "checks": {"failed": []},
            "results": results,
            "run_utc": "2026-11-24 10:00",
        },
    )
    nh.write(
        tmp_path,
        "choice",
        {
            "setting": 3,
            "width": 512,
            "decay": 0.1,
            "epochs": 7,
            "scores": [[0.001] * 20] * 4,
            "run_utc": "2026-11-24 12:00",
        },
    )
    path = nh.write(
        tmp_path,
        "measure",
        {"results": results, "gates": gates, "run_utc": "2026-11-25 09:00"},
    )
    text = path.read_text()
    for part in (
        "Efe&#x27;s network",
        "Variant 2 failed, and is written down as it is.",
        "The placebo passed",
        "Kept: setting 3 (width 512, decay 0.1) after 7 epochs.",
        "After their own cost (t)",
        "<td>30 min</td>",
        "+0.70 (+1.1)",
    ):
        assert part in text, part


# --- with PyTorch, on the GPU when there is one -------------------------------------------


class StandIn:
    """A stand-in for Efe's network: a linear forecast learned by plain gradient steps. It
    keeps what it was asked."""

    def __init__(self, skip_epochs: bool = False, no_number: bool = False):
        self.asked = []
        self.targets = []  # the first targets it learned from, in their order
        self.skip_epochs = skip_epochs
        self.no_number = no_number

    def learn(self, rows, width, decay, seed, epochs, after_epoch):
        import torch

        self.asked.append((width, decay, seed, epochs, rows.count))
        self.targets.append(rows.z[:200, 0].cpu().numpy().copy())
        generator = torch.Generator(device=rows.device).manual_seed(seed)
        w = torch.zeros(
            rows.inputs, rows.horizons, device=rows.device, requires_grad=True
        )
        for epoch in range(1, epochs + 1):
            order = torch.randperm(rows.count, generator=generator, device=rows.device)
            for start in range(0, rows.count, 256):
                x, z = rows.batch(order[start : start + 256])
                ((x @ w - z) ** 2).mean().backward()
                with torch.no_grad():
                    w -= 0.002 * w.grad
                    w.grad = None
            if not self.skip_epochs:
                after_epoch(epoch, {"w": w.detach().clone()})
        return {"w": w.detach()}

    def forecast(self, params, x):
        return x @ params["w"] * (float("nan") if self.no_number else 1.0)


def test_rows_on_the_device_are_the_standardised_inputs_and_z(tmp_path):
    torch = pytest.importorskip("torch")
    data = made.store(tmp_path)
    net_data.build(data, "real", years=(2016, 2017), log=quiet)
    table = net_data.load(data, "real", [2016, 2017])
    on = nh.device()
    mean, scale = nh.standardisation(table, on)
    x = net_data.inputs_of(table, np.arange(len(table["day"]))).astype(np.float64)
    assert mean == pytest.approx(x.mean(axis=0), abs=1e-6)
    sd = x.std(axis=0)
    assert scale == pytest.approx(np.where(sd > 1e-9, sd, 1.0), rel=1e-5, abs=1e-6)
    assert (scale == 1.0).any()  # no headline here, so that channel never varies
    rows = nh.Rows(table, mean, scale, on)
    index = torch.tensor([0, 7, 300, len(table["day"]) - 1], device=on)
    got_x, got_z = rows.batch(index)
    expected = (x[index.cpu().numpy()] - mean) / scale
    assert got_x.cpu().numpy() == pytest.approx(expected, abs=1e-4)
    assert got_z.cpu().numpy() == pytest.approx(
        table["z"][index.cpu().numpy()], abs=1e-6
    )
    shuffled_z = nh.shuffled(table["z"], table["day"], table["minute"])
    dealt = nh.Rows(table, mean, scale, on, z=shuffled_z)
    assert dealt.batch(index)[1].cpu().numpy() == pytest.approx(
        shuffled_z[index.cpu().numpy()], abs=1e-6
    )
    net = StandIn()
    params = net.learn(rows, 128, 0.1, 1, 1, lambda e, p: None)
    found = nh.forecasts(net, params, rows)
    assert found.shape == (rows.count, len(HORIZONS)) and np.isfinite(found).all()


def test_the_choice_learns_each_setting_s_seeds_and_scores_each_epoch(
    tmp_path, monkeypatch
):
    pytest.importorskip("torch")
    data = made.store(tmp_path)
    net_data.build(data, "real", years=(2016, 2017, 2018, 2019), log=quiet)
    monkeypatch.setattr(nh, "EPOCHS", 3)
    scored, honest = [], nh.score
    monkeypatch.setattr(
        nh, "score", lambda p, z, day: scored.append(z) or honest(p, z, day)
    )
    net = StandIn()
    chosen = net_data.load(data, "real", [skill.CHOOSE_YEAR])
    found = nh.choice_on(
        net, net_data.load(data, "real", skill.FIT_YEARS), chosen, log=quiet
    )
    assert len(scored) == 4 * 3
    assert all(np.array_equal(z, chosen["z"][:, nh.SCORED_AT]) for z in scored)
    learned = net_data.load(data, "real", skill.FIT_YEARS)["day"].size
    assert net.asked == [
        (w, d, seed, 3, learned) for w, d in nh.SETTINGS for seed in nh.SEEDS
    ]
    scores = np.array(found["scores"])
    assert scores.shape == (4, 3) and np.isfinite(scores).all()
    i, epochs = nh.kept(scores)
    assert (found["setting"], found["epochs"]) == (i + 1, epochs)
    assert (found["width"], found["decay"]) == nh.SETTINGS[i]
    # the stand-in ignores width and decay, so each setting scores the same
    assert np.allclose(scores, scores[0]) and found["setting"] == 1
    with pytest.raises(ValueError, match="after_epoch once after each"):
        nh.choice_on(
            StandIn(skip_epochs=True),
            net_data.load(data, "real", skill.FIT_YEARS),
            net_data.load(data, "real", [skill.CHOOSE_YEAR]),
            log=quiet,
        )


def test_the_placebo_and_the_measurement_from_start_to_end(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    data = store_with_skill_runs(tmp_path)
    monkeypatch.setattr(nh, "EPOCHS", 2)
    monkeypatch.setattr(nh, "SEEDS", (1, 2))
    standardised, honest = [], nh.standardisation
    monkeypatch.setattr(
        nh,
        "standardisation",
        lambda t, on: standardised.append(len(t["day"])) or honest(t, on),
    )
    limited, thresholds = [], nh.thresholds_by_horizon
    monkeypatch.setattr(
        nh, "thresholds_by_horizon", lambda p: limited.append(len(p)) or thresholds(p)
    )
    net = StandIn()
    found = nh.placebo(net, data, log=quiet)
    rows = {
        y: net_data.manifest(data, "random")[str(y)]["rows"] for y in net_data.YEARS
    }
    early = sum(rows[y] for y in skill.FIT_YEARS)
    assert standardised == [early, early + rows[skill.CHOOSE_YEAR]]
    assert (
        limited == [rows[skill.CHOOSE_YEAR]] * 2
    )  # the network's and its twin's, on 2019
    network, twin = net.targets[-4], net.targets[-2]
    assert not np.array_equal(
        network, twin
    )  # the twin learned from targets dealt again
    assert np.array_equal(net.targets[-3], network) and np.array_equal(
        net.targets[-1], twin
    )
    assert found["source"] == "random" and isinstance(found["passed"], bool)
    assert set(found["results"]) == {
        "network",
        "ridge",
        "shuffled",
        "habit",
        "reversal",
    }
    choice = found["choice"]
    learned = sum(
        net_data.manifest(data, "random")[str(y)]["rows"]
        for y in (*skill.FIT_YEARS, skill.CHOOSE_YEAR)
    )
    # the measurement learns twice, the network and its shuffled twin, on 2016-2019
    tail = net.asked[-4:]
    assert (
        tail
        == [
            (choice["width"], choice["decay"], s, choice["epochs"], learned)
            for s in (1, 2)
        ]
        * 2
    )
    gates = found["gates"]
    assert set(gates) == {
        "over the ridge",
        "over the shuffled network",
        "its own cost",
        "variant passed",
        "bank and simulator next",
    }
    record = nh.plain(found | {"run_utc": "now", "code": {}})
    json.dumps(record)  # every number is plain
    # the real measurement stops when the ridge does not give the skill test's numbers
    real = json.loads((data / "kutbit" / "skill" / "real.json").read_text())
    real["choice"]["15"]["penalty"] = 99.0
    (data / "kutbit" / "skill" / "real.json").write_text(json.dumps(real))
    with pytest.raises(SystemExit, match="does not give the skill test's numbers"):
        nh.measurement(net, data, "real", choice, log=quiet)


def test_smoke_times_each_width_and_shows_no_score(tmp_path):
    pytest.importorskip("torch")
    data = made.store(tmp_path)
    for source in net_data.SOURCES:
        net_data.build(data, source, log=quiet)
    said = []
    found = nh.smoke(StandIn(), data, log=said.append)
    assert all(found[w]["step"] > 0 and found[w]["forecast"] > 0 for w in (128, 512))
    assert found["choice hours"] > 0 and "Adam included" in said[-1]
    assert "corr" not in said[-1].lower()
    with pytest.raises(ValueError, match="after_epoch"):
        nh.smoke(StandIn(skip_epochs=True), data, log=quiet)
    with pytest.raises(ValueError, match="a finite number for every row"):
        nh.smoke(StandIn(no_number=True), data, log=quiet)
