"""The network's rows (research/kutbit/net_data.py): exactly the skill test's.

The tests that time the GPU need PyTorch, so they run in data/llmenv and are skipped elsewhere:

    PYTHONPATH=src data/llmenv/bin/python -m pytest tests/test_kutbit_net_data.py
"""

import importlib.util
import json
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from qfin_alphaguard.kutbit.features import CHANNELS, SCALARS, WINDOW

HERE = Path(__file__).parents[1] / "research" / "kutbit"
sys.path.insert(0, str(HERE))
_spec = importlib.util.spec_from_file_location("kutbit_net_data", HERE / "net_data.py")
nd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nd)
import skill

TICKERS = ("QQQ", "S00", "S01", "XLF")
LATE = "S02"  # listed in 2021 only, so a year's own list would number XLF otherwise
LENGTHS = (
    130,
    50,
)  # each year's two sessions: one whose late windows are full, one shorter
FIELDS = ("open", "high", "low", "close", "volume", "profile", "swing", "habit")


def quiet(*_):
    pass


def store(tmp_path: Path, seed: int = 5) -> Path:
    """A data folder with two sessions a year of 2016-2021, of LENGTHS minutes, for four
    instruments whose moves differ by instrument and day, and a fifth in 2021."""
    rng = np.random.default_rng(seed)
    rows, arrays = [], {}
    for year in nd.YEARS:
        fields = {f: [] for f in (*FIELDS, "seen", "news")}
        for d, minutes in enumerate(LENGTHS):
            for ticker in (*TICKERS, LATE) if year == 2021 else TICKERS:
                close = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, minutes)))
                open_ = np.concatenate([[100.0], close[:-1]])
                values = {
                    "open": open_,
                    "high": np.maximum(open_, close) * 1.0003,
                    "low": np.minimum(open_, close) * 0.9997,
                    "close": close,
                    "volume": rng.uniform(500, 1500, minutes),
                    "profile": np.full(minutes, 1000.0),
                    "swing": np.full(minutes, 1e-3),
                    "habit": rng.normal(0, 1e-3, minutes),
                }
                for f in FIELDS:
                    fields[f].append(
                        np.pad(values[f], (0, 390 - minutes), mode="edge").astype(
                            np.float32
                        )
                    )
                fields["seen"].append(np.ones(390, dtype=bool))
                fields["news"].append(np.zeros(390, dtype=np.uint16))
                rows.append(
                    {
                        "day": date(year, 3, 1 + d),
                        "year": year,
                        "row": len(fields["seen"]) - 1,
                        "ticker": ticker,
                        "etf": ticker in ("QQQ", "XLF"),
                        "minutes": minutes,
                        "prev_close": 100.0,
                        "vol": 0.02,
                        "beta": 0.0 if ticker == "QQQ" else 0.8,
                        "vix": 20.0,
                        "news_before": 0,
                        "split": 1.0,
                        "usable": True,
                    }
                )
        arrays[year] = {f: np.stack(v) for f, v in fields.items()}
    folder = tmp_path / "kutbit"
    folder.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(folder / "days.parquet")
    for year, fields in arrays.items():
        np.savez_compressed(folder / f"{year}.npz", **fields)
    return tmp_path


def skill_rows(data: Path, year: int, source: str) -> dict:
    """The skill test's rows of `year`, from its own code, as one table."""
    days, arrays = skill.load(data, [year])
    tickers = {t: i for i, t in enumerate(sorted(days["ticker"].unique()))}
    found = list(skill.each_day(days, arrays, [year], tickers, source))
    pick = {
        k: np.concatenate([getattr(r, k) for _, r in found])
        for k in ("x", "z", "apart", "patterns", "minute", "instrument", "etf")
    }
    pick["day"] = np.concatenate([np.full(len(r.x), d.toordinal()) for d, r in found])
    return pick


def test_every_row_is_the_skill_test_s_row_real_and_random(tmp_path):
    data = store(tmp_path)
    for source in nd.SOURCES:
        nd.build(data, source, log=quiet)
    for year in nd.YEARS:
        real = nd.load(data, "real", [year])
        for source in nd.SOURCES:
            table = nd.load(data, source, [year])
            expected = skill_rows(data, year, source)
            n = len(expected["x"])
            stocks = 4 if year == 2021 else 3  # all but QQQ, every minute but the last
            assert n == stocks * sum(m - 1 for m in LENGTHS)
            assert np.array_equal(nd.inputs_of(table, np.arange(n)), expected["x"])
            for k in ("z", "apart", "patterns", "minute", "instrument", "etf", "day"):
                assert np.array_equal(table[k], expected[k]), k
            assert table["z"].dtype == np.float64
        assert not np.array_equal(table["channels"], real["channels"])  # walks differ


def test_a_window_starts_at_its_own_session_s_open():
    # a session of 5 rows of -1, then one of 130 rows, each row's channels its index
    channels = np.concatenate(
        [
            np.full((5, len(CHANNELS)), -1.0),
            np.repeat(np.arange(5.0, 135.0)[:, None], 8, 1),
        ]
    ).astype(np.float32)
    minute = np.concatenate([np.arange(5), np.arange(130)])
    windows = nd.windows_of(channels, minute, np.array([5, 10, 134]))
    assert windows.shape == (3, WINDOW, len(CHANNELS))
    assert (windows[0, :-1] == 0).all() and (windows[0, -1] == 5).all()
    assert (windows[1, :-6] == 0).all()
    assert np.array_equal(windows[1, -6:, 0], np.arange(5.0, 11.0))
    assert np.array_equal(windows[2, :, 0], np.arange(15.0, 135.0))  # full, no zeros
    assert not (windows == -1).any()  # never the session before
    x = nd.inputs_of(
        {"channels": channels, "minute": minute, "scalars": np.ones((135, 11))},
        np.array([10]),
    )
    assert x.shape == (1, WINDOW * len(CHANNELS) + len(SCALARS))
    assert np.array_equal(x[0, : WINDOW * len(CHANNELS)], windows[1].reshape(-1))


def test_each_year_and_its_code_are_recorded_and_not_built_again(tmp_path, monkeypatch):
    data = store(tmp_path)
    said = []
    nd.build(data, "real", years=(2016,), log=said.append)
    path = nd.folder(data, "real") / "2016.npz"
    record = nd.manifest(data, "real")
    assert record["2016"] == {
        "rows": 3 * sum(m - 1 for m in LENGTHS),
        "code": nd.fingerprint(),
    }
    assert json.loads((data / "kutbit" / "net" / "tickers.json").read_text()) == sorted(
        (*TICKERS, LATE)
    )
    built = path.stat().st_mtime_ns
    nd.build(data, "real", years=(2016,), log=said.append)
    assert path.stat().st_mtime_ns == built and "already" in said[-1]
    assert not list(nd.folder(data, "real").glob("*.part.npz"))
    monkeypatch.setattr(nd, "fingerprint", lambda: {"net_data.py": "changed"})
    with pytest.raises(ValueError, match="not built by today's code"):
        nd.load(data, "real", [2016])
    nd.build(data, "real", years=(2016,), log=said.append)
    assert "rows" in said[-1] and nd.load(data, "real", [2016])["day"].size


def test_years_after_2021_are_never_read(tmp_path):
    data = store(tmp_path)
    with pytest.raises(ValueError, match="2022-2026 stay unread"):
        nd.build(data, "real", years=(2021, 2022), log=quiet)
    assert not (data / "kutbit" / "net").exists()
    with pytest.raises(ValueError, match="2022-2026 stay unread"):
        nd.load(data, "real", [2022])
    with pytest.raises(ValueError, match="the source is one of"):
        nd.build(data, "shuffled", log=quiet)


def test_a_window_that_is_not_the_ridge_s_input_stops_the_build(tmp_path, monkeypatch):
    data = store(tmp_path)
    honest = nd.stored_rows

    def off_by_a_little(rows, day):
        part = honest(rows, day)
        part["channels"][-1, 0] += np.float32(1e-3)
        return part

    monkeypatch.setattr(nd, "stored_rows", off_by_a_little)
    with pytest.raises(ValueError, match="not the skill test's input"):
        nd.build(data, "real", years=(2016,), log=quiet)
    assert not (nd.folder(data, "real") / "2016.npz").exists()


def test_the_years_load_as_one_table_in_order(tmp_path):
    data = store(tmp_path)
    nd.build(data, "real", years=(2016, 2017), log=quiet)
    both = nd.load(data, "real", [2016, 2017])
    one = nd.load(data, "real", [2016])
    assert len(both["day"]) == 2 * len(one["day"])
    assert np.all(np.diff(both["day"]) >= 0)
    rows = np.arange(len(both["day"]))
    expected = np.concatenate(
        [skill_rows(data, 2016, "real")["x"], skill_rows(data, 2017, "real")["x"]]
    )
    assert np.array_equal(nd.inputs_of(both, rows), expected)


# --- on the GPU, with PyTorch -------------------------------------------------------------


def stand_in(data: Path, rows_a_year: int = 600) -> None:
    """Made-up stored years of 2016-2021, as if built by today's code."""
    rng = np.random.default_rng(1)
    out = nd.folder(data, "real")
    out.mkdir(parents=True)
    record = {}
    for year in nd.YEARS:
        minute = np.arange(rows_a_year) % 200
        table = {
            "channels": rng.normal(size=(rows_a_year, len(CHANNELS))),
            "scalars": rng.normal(size=(rows_a_year, len(SCALARS))),
            "z": rng.normal(size=(rows_a_year, 5)),
            "apart": rng.normal(size=(rows_a_year, 5)),
            "patterns": rng.normal(size=(rows_a_year, 2)),
            "minute": minute,
            "instrument": np.zeros(rows_a_year),
            "etf": np.zeros(rows_a_year),
            "day": np.full(rows_a_year, date(year, 3, 1).toordinal()),
        }
        np.savez(
            out / f"{year}.npz",
            **{k: v.astype(nd.FIELDS[k]) for k, v in table.items()},
        )
        record[str(year)] = {"rows": rows_a_year, "code": nd.fingerprint()}
    (out / "manifest.json").write_text(json.dumps(record))


def test_the_gpu_gathers_the_same_windows():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(2)
    minute = np.concatenate([np.arange(40), np.arange(150), np.arange(7)])
    channels = rng.normal(size=(len(minute), len(CHANNELS))).astype(np.float32)
    rows = rng.permutation(len(minute))[:64]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    found = nd.gathered(
        torch.from_numpy(channels).to(device),
        torch.from_numpy(minute).to(device),
        torch.from_numpy(rows).to(device),
    )
    assert np.array_equal(found.cpu().numpy(), nd.windows_of(channels, minute, rows))


def test_speed_times_the_runs_and_shows_no_score(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    stand_in(tmp_path)
    waits = []
    if torch.cuda.is_available():  # a GPU's work is timed only once it has finished
        finish = torch.cuda.synchronize
        monkeypatch.setattr(
            torch.cuda, "synchronize", lambda *a: waits.append(1) or finish(*a)
        )
    said = []
    found = nd.speed(tmp_path, batches=2, log=said.append)
    if torch.cuda.is_available():  # before and after the gathering and each width's two
        assert len(waits) == 2 * (1 + 2 * len(nd.WIDTHS))
    assert found["rows"] == 3 * 600 and found["batch"] == 3 * 600
    assert found["gather"] > 0
    assert all(found[w]["step"] > 0 and found[w]["forecast"] > 0 for w in nd.WIDTHS)
    assert found["choice hours"] > 0 and found["measurement hours"] > 0
    text = said[-1]
    assert "an epoch of 2016-2018" in text and "the placebo as much again" in text
    for word in ("corr", "score", "bps"):
        assert word not in text.lower()
    assert not re.search(r"\bt [+-]?\d", text)  # no t of anything


def test_the_runs_add_up_from_the_seconds_a_batch_takes():
    found = {
        "batch": 100,
        128: {"step": 1.0, "forecast": 0.5},
        512: {"step": 2.0, "forecast": 1.0},
    }
    record = {str(y): {"rows": 1000} for y in nd.YEARS}  # 10 batches a year
    hours = nd.runs(found, record)
    # each width: 2 settings x 5 seeds x 20 epochs x (30 batches a step + 10 a forecast)
    choice = 2 * 5 * 20 * ((30 * 1.0 + 10 * 0.5) + (30 * 2.0 + 10 * 1.0))
    # the network and its shuffled twin: 5 seeds x (20 epochs x 40 batches + 30 forecasts)
    measure = 2 * 5 * (20 * 40 * 2.0 + 30 * 1.0)
    assert hours == pytest.approx(
        {"choice hours": choice / 3600, "measurement hours": measure / 3600}
    )
