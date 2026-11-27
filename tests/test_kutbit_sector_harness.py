"""The harness of KUTBIT's third variant (research/kutbit/sector_harness.py, decision 0028):
the skill test's ridge given its sector's move, judged as the decision wrote."""

import html
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).parents[1] / "research" / "kutbit"
sys.path.insert(0, str(HERE))


def loaded(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sh = loaded("kutbit_sector_harness", HERE / "sector_harness.py")
made = loaded(
    "kutbit_net_harness_tests", Path(__file__).parent / "test_kutbit_net_harness.py"
)
import net_data
import net_harness
import sector_data
import skill
from labels import HORIZONS

FIFTEEN = HORIZONS.index(15)


def quiet(*_):
    pass


@pytest.fixture(autouse=True)
def few_threads():
    """Small solves on many threads run a hundred times slower here."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        yield
        return
    with threadpool_limits(4):
        yield


def channels(data: Path, make) -> None:
    """Channels written for every year of the network's rows, as sector_data stores them:
    make(table, source) gives {"sector": ..., "wrong": ...} for a year's table."""
    found = {"instruments": {}, "sectors": {}}
    out = data / "kutbit" / "sector"
    out.mkdir(parents=True, exist_ok=True)
    (out / "match.json").write_text(json.dumps(found))
    for source in net_data.SOURCES:
        sector_data.folder(data, source).mkdir(parents=True, exist_ok=True)
        record = {}
        for year in net_data.YEARS:
            table = net_data.load(data, source, [year])
            values = {
                k: np.asarray(v, dtype=np.float32)
                for k, v in make(table, source).items()
            }
            np.savez(sector_data.folder(data, source) / f"{year}.npz", **values)
            record[str(year)] = {
                "rows": len(table["day"]),
                "made": sector_data.made_of(data, source, year, found),
            }
        (sector_data.folder(data, source) / "manifest.json").write_text(
            json.dumps(record)
        )


def planted(table: dict, source: str) -> dict:
    """On the real prices, the sector channel is the 15-minute target itself, an edge no
    ridge could miss; the wrong sector, and everything on the walks, is noise."""
    rng = np.random.default_rng(len(table["day"]))
    noise = rng.normal(size=len(table["day"]))
    edge = table["z"][:, FIFTEEN] if source == "real" else noise
    return {"sector": edge, "wrong": rng.normal(size=len(table["day"]))}


def noise(table: dict, source: str) -> dict:
    rng = np.random.default_rng(len(table["day"]) + 1)
    return {k: rng.normal(size=len(table["day"])) for k in sector_data.KINDS}


@pytest.fixture(scope="module")
def store(tmp_path_factory) -> Path:
    """The made-up rows of the data tests, built, with the skill test run on them, the cost
    study's made-up moments and planted channels."""
    data = made.store_with_skill_runs(tmp_path_factory.mktemp("sector"))
    channels(data, planted)
    return data


@pytest.fixture(scope="module")
def measured(store) -> dict:
    return sh.run(store, "real", log=quiet)


# --- the constants and the inputs -----------------------------------------------------------


def test_the_constants_are_decision_0028_s():
    assert sh.SCORED == (15, 30, 60) and sh.PROOF == 2.0 and sh.PLACEBO_T == 3.0
    assert sh.LEARN_YEARS == (2016, 2017, 2018, 2019)
    assert sh.BASE == 971 and sh.WIDE == 1_211
    assert len(sh.COLUMNS["sector"]) == sector_data.INPUTS == 1_091
    assert np.array_equal(sh.COLUMNS["sector"], np.arange(1_091))
    assert np.array_equal(sh.COLUMNS["wrong"][:971], np.arange(971))
    assert np.array_equal(sh.COLUMNS["wrong"][971:], np.arange(1_091, 1_211))
    assert np.array_equal(sh.TARGETS["shuffled"], np.arange(5, 10))
    assert set(sh.CODE) == {
        "sector_harness.py",
        "sector_data.py",
        "net_harness.py",
        "net_data.py",
        "skill.py",
        "costs.py",
    }


def test_the_wide_inputs_hold_each_ridge_s_own_inputs(store):
    net = net_data.load(store, "real", [2017])
    sector = sector_data.load(store, "real", [2017])
    rows = np.arange(len(net["day"]))
    wide = sh.inputs(net, sector, rows)
    for kind in sector_data.KINDS:
        own = sector_data.inputs_of(net, sector, rows, kind)
        assert np.array_equal(wide[:, sh.COLUMNS[kind]], own)


def test_the_sums_give_each_ridge_the_moments_of_its_inputs_and_targets(store):
    net = net_data.load(store, "real", sh.LEARN_YEARS)
    sector = sector_data.load(store, "real", sh.LEARN_YEARS)
    by_year = sh.sums(net, sector)
    assert sorted(by_year) == list(sh.LEARN_YEARS)
    dealt = net_harness.shuffled(net["z"], net["day"], net["minute"])
    for kind, target in (("sector", "z"), ("wrong", "z"), ("sector", "shuffled")):
        for year in sh.LEARN_YEARS:
            rows = sh.rows_in(net, year)
            mine = sh.part(by_year[year], sh.COLUMNS[kind], sh.TARGETS[target])
            theirs = skill.Moments(
                0,
                np.zeros(1_091),
                np.zeros((1_091, 1_091)),
                np.zeros((1_091, 5)),
                np.zeros(5),
            )
            z = net["z"] if target == "z" else dealt
            for day in net_harness.by_day({"day": net["day"][rows]}):
                at = rows[day]
                theirs.add(sector_data.inputs_of(net, sector, at, kind), z[at])
            assert mine.n == theirs.n == len(rows)
            for field in ("s", "r", "t"):  # summed in float64
                assert np.allclose(
                    getattr(mine, field), getattr(theirs, field), rtol=1e-9, atol=1e-9
                )
            # q is summed in float32 a day at a time, as the skill test sums it, so another
            # shape of the same product may round its last digits otherwise: by up to about
            # 3e-6 of its largest entry here, and by more on another machine's BLAS
            assert np.abs(mine.q - theirs.q).max() <= 1e-4 * np.abs(theirs.q).max()


def test_the_rows_of_a_year_are_its_days():
    from datetime import date

    days = [date(2018, 12, 31), date(2019, 1, 1), date(2019, 12, 31), date(2020, 1, 1)]
    table = {"day": np.array([d.toordinal() for d in days])}
    assert sh.rows_in(table, 2019).tolist() == [1, 2]
    assert sh.year_of(date(2019, 6, 3).toordinal()) == 2019


def test_each_horizon_keeps_the_penalty_chosen_for_it():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4_000, 30)).astype(np.float32)
    z = x[:, :10] @ rng.normal(size=(10, 10)) * 0.1 + rng.normal(size=(4_000, 10))
    m = skill.Moments(
        0, np.zeros(30), np.zeros((30, 30)), np.zeros((30, 10)), np.zeros(10)
    )
    m.add(x, z)
    columns, targets = np.arange(25), np.arange(5, 10)
    choice = {h: {"penalty": skill.PENALTIES[j]} for j, h in enumerate(HORIZONS)}
    ridge = sh.fitted(m, columns, targets, choice)
    for j, h in enumerate(HORIZONS):
        alone = skill.fit(sh.part(m, columns, targets), choice[h]["penalty"])
        assert np.array_equal(ridge.beta[:, j], alone.beta[:, j])
        assert ridge.intercept[j] == alone.intercept[j]


# --- a run ----------------------------------------------------------------------------------


def test_a_planted_edge_passes_the_gates(measured, store):
    assert list(measured["results"]) == [
        "sector",
        "ridge",
        "shuffled",
        "wrong",
        "habit",
        "reversal",
    ]
    sector = measured["results"]["sector"]["15"]["all"]["daily"]["mean"]
    ridge = measured["results"]["ridge"]["15"]["all"]["daily"]["mean"]
    assert sector > 0.5 > abs(ridge)
    g = measured["gates"]
    assert g["over the ridge"]["mean"] > 0.1 and g["over the ridge"]["passed"]
    assert g["over the shuffled ridge"]["passed"]
    assert g["over the wrong sector, shown"]["mean"] > 0.1
    assert g["variant passed"]
    assert g["bank and simulator next"] == (
        g["variant passed"] and g["its own cost"]["passed"]
    )
    rows = {y: net_data.manifest(store, "real")[str(y)]["rows"] for y in net_data.YEARS}
    assert measured["learned_rows"] == sum(rows[y] for y in sh.LEARN_YEARS)
    assert measured["measured_rows"] == sum(rows[y] for y in skill.MEASURE_YEARS)
    assert set(measured["penalties"]) == {str(h) for h in HORIZONS}
    theirs = json.loads((store / "kutbit" / "skill" / "real.json").read_text())
    assert measured["ridge_penalties"] == {
        h: c["penalty"] for h, c in theirs["choice"].items()
    }
    json.dumps(net_harness.plain(measured))  # every number is plain


def test_the_choice_on_2019_is_the_skill_test_s_rule_on_the_new_inputs(measured, store):
    net = net_data.load(store, "real", sh.LEARN_YEARS)
    sector = sector_data.load(store, "real", sh.LEARN_YEARS)
    by_year = sh.sums(net, sector)
    early = sum((by_year[y] for y in skill.FIT_YEARS), sh.empty())
    chosen = sh.rows_in(net, skill.CHOOSE_YEAR)
    x = sh.inputs(net, sector, chosen)[:, sh.COLUMNS["sector"]]
    for j, h in enumerate(HORIZONS):
        scores = [
            skill.correlation(
                skill.fit(sh.part(early, sh.COLUMNS["sector"], sh.TARGETS["z"]), p)(x)[
                    :, j
                ],
                net["z"][chosen, j],
            )
            for p in skill.PENALTIES
        ]
        assert measured["choice_scores"][str(h)] == pytest.approx(scores, abs=1e-12)
        assert (
            measured["penalties"][str(h)] == skill.PENALTIES[int(np.nanargmax(scores))]
        )


def test_the_run_stops_when_the_ridge_is_not_the_skill_test_s(tmp_path):
    data = made.store_with_skill_runs(tmp_path)
    channels(data, noise)
    path = data / "kutbit" / "skill" / "real.json"
    theirs = json.loads(path.read_text())
    theirs["choice"]["15"]["penalty"] = 99.0
    path.write_text(json.dumps(theirs))
    with pytest.raises(SystemExit, match="does not give the skill test's numbers"):
        sh.run(data, "real", log=quiet)


def test_the_placebo_on_the_walks_and_its_page(store, tmp_path):
    found = sh.run(store, "random", log=quiet)
    assert found["source"] == "random"
    assert found["passed"] is found["checks"]["passed"]
    assert found["checks"] == skill.checks(
        found["results"]
    )  # every forecaster is checked
    assert found["passed"] == (not found["checks"]["failed"])
    folder = tmp_path / "records"
    path = sh.write(
        folder, "placebo", found | {"run_utc": "2026-11-27 10:00", "code": {}}
    )
    shown = path.read_text()
    assert "The placebo, on random walks, 2026-11-27 10:00 UTC" in shown
    verdict = "The placebo passed" if found["passed"] else "The placebo failed"
    assert verdict in shown and "Measured once" not in shown


# --- the records and the page ---------------------------------------------------------------


def test_the_measurement_waits_for_a_placebo_that_passed_on_today_s_code_and_runs_once(
    tmp_path,
):
    code = {"sector_harness.py": "a", "sector_data.py": "b"}
    assert sh.refusal(tmp_path, code) == "run the placebo first"
    placebo = tmp_path / "placebo.json"
    placebo.write_text(json.dumps({"passed": False, "code": code}))
    assert "the placebo failed" in sh.refusal(tmp_path, code)
    placebo.write_text(
        json.dumps({"passed": True, "code": code | {"sector_data.py": "c"}})
    )
    assert "sector_data.py changed since" in sh.refusal(tmp_path, code)
    placebo.write_text(json.dumps({"passed": True, "code": code}))
    assert sh.refusal(tmp_path, code) is None
    (tmp_path / "measure.json").write_text("{}")
    assert "runs only once" in sh.refusal(tmp_path, code)


def test_the_fingerprint_is_each_file_s_sha_256():
    import hashlib

    found = sh.fingerprint()
    assert set(found) == set(sh.CODE)
    assert (
        found["sector_harness.py"]
        == hashlib.sha256((HERE / "sector_harness.py").read_bytes()).hexdigest()
    )


def test_the_page_shows_the_measurement_s_gates_and_verdict(measured, tmp_path):
    folder = tmp_path / "records"
    sh.write(folder, "measure", measured | {"run_utc": "2026-11-27 11:00", "code": {}})
    shown = (folder / "sector.html").read_text()
    assert "Measured once on 2020-2021, 2026-11-27 11:00 UTC" in shown
    assert "Variant 3 passed." in shown
    assert "Shown, not a gate: its margin over the wrong sector" in shown
    for name in sh.NAMES.values():
        assert html.escape(name) in shown
    assert "Penalties chosen on 2019: 1 min" in shown
    failed = json.loads(json.dumps(net_harness.plain(measured)))
    failed["gates"]["variant passed"] = False
    sh.write(folder, "measure", failed | {"run_utc": "2026-11-27 11:00", "code": {}})
    assert (
        "Variant 3 failed, and is written down as it is."
        in (folder / "sector.html").read_text()
    )


def test_the_commands_refuse_what_is_not_their_turn(tmp_path):
    with pytest.raises(SystemExit, match="usage"):
        sh.main(["choose"])
    with pytest.raises(SystemExit, match="run the placebo first"):
        sh.main(["measure", str(tmp_path)])


def test_each_forecaster_is_its_ridge_on_2016_2019_with_2019_s_thresholds(
    measured, store
):
    net = net_data.load(store, "real", sh.LEARN_YEARS)
    sector = sector_data.load(store, "real", sh.LEARN_YEARS)
    by_year = sh.sums(net, sector)
    final = sum((by_year[y] for y in sh.LEARN_YEARS), sh.empty())
    choice = {int(h): {"penalty": p} for h, p in measured["penalties"].items()}
    on_2019 = sh.inputs(net, sector, sh.rows_in(net, skill.CHOOSE_YEAR))
    table = net_data.load(store, "real", skill.MEASURE_YEARS)
    later = sector_data.load(store, "real", skill.MEASURE_YEARS)
    x = sh.inputs(table, later, np.arange(len(table["day"])))
    cost = net_harness.signal_costs(store, table)
    for name, kind, target in (
        ("sector", "sector", "z"),
        ("wrong", "wrong", "z"),
        ("shuffled", "sector", "shuffled"),
    ):
        ridge = sh.fitted(final, sh.COLUMNS[kind], sh.TARGETS[target], choice)
        limits = net_harness.thresholds_by_horizon(ridge(on_2019[:, sh.COLUMNS[kind]]))
        mine = net_harness.measures_of(
            ridge(x[:, sh.COLUMNS[kind]]), table, lambda h, lim=limits: lim[h], cost
        )
        assert net_harness.differences(mine, measured["results"][name], name) == []


def test_the_gates_ask_the_ridge_the_shuffled_ridge_and_the_cost():
    rng = np.random.default_rng(3)
    day = np.repeat(np.arange(30), 400)
    z = rng.normal(size=(12_000, len(HORIZONS)))
    noise = rng.normal(size=z.shape)
    p = {
        "sector": z + 2 * noise,
        "ridge": z + 4 * noise,
        "shuffled": noise,
        "wrong": -z,
    }
    results = {
        "sector": {
            str(h): {
                "all": {
                    "top0.01": {"own": own, "own_t": t},
                    "top0.1": {"own": 9.0, "own_t": 9.0},
                }
            }
            for h, own, t in ((15, 1.0, 1.9), (30, -1.0, 5.0), (60, 2.0, 2.0))
        }
    }
    table = {"day": day, "z": z}
    g = sh.gates(p, table, results)
    for key, other in (
        ("over the ridge", "ridge"),
        ("over the shuffled ridge", "shuffled"),
        ("over the wrong sector, shown", "wrong"),
    ):
        assert g[key] == net_harness.margin(p["sector"], p[other], z, day)
    assert g["over the ridge"]["passed"] and g["over the shuffled ridge"]["passed"]
    assert g["its own cost"]["by_horizon"] == {"15": False, "30": False, "60": True}
    assert g["variant passed"] and g["bank and simulator next"]
    p["ridge"] = p["sector"]  # no margin over the skill test's ridge
    g = sh.gates(p, table, results)
    assert not g["variant passed"] and not g["bank and simulator next"]
    p["ridge"], p["shuffled"] = z + 4 * noise, p["sector"]  # nor over shuffled labels
    assert not sh.gates(p, table, results)["variant passed"]
