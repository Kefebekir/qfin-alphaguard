"""The sector channel of KUTBIT's third variant (research/kutbit/sector_data.py, decision
0028): the match on 2016-2018, the channel from the minute before, and its build."""

import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from qfin_alphaguard.kutbit.features import CHANNELS, SCALARS, WINDOW

HERE = Path(__file__).parents[1] / "research" / "kutbit"
sys.path.insert(0, str(HERE))
_spec = importlib.util.spec_from_file_location(
    "kutbit_sector_data", HERE / "sector_data.py"
)
sd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sd)
import net_data as nd

APART, SEEN = CHANNELS.index("apart"), CHANNELS.index("seen")
STOCKS = ("S00", "S01", "S02", "S03")
NAMES = sorted(["QQQ", *sd.SECTORS, *STOCKS])


def quiet(*_):
    pass


def table_of(
    apart: dict, seen: dict | None = None, present: dict | None = None, first=0
):
    """The network's stored rows for `apart`, each ticker's [sessions, minutes]: a session's
    rows one instrument after another, as net_data stores them. `seen` marks the minutes that
    traded (all by default); `present` the sessions a ticker has rows in (all by default)."""
    sessions, minutes = next(iter(apart.values())).shape
    parts = []
    for d in range(sessions):
        for t in sorted(apart, key=NAMES.index):
            if present is not None and d not in present.get(t, range(sessions)):
                continue
            channels = np.zeros((minutes, len(CHANNELS)), np.float32)
            channels[:, APART] = apart[t][d]
            channels[:, SEEN] = 1.0 if seen is None else seen[t][d]
            parts.append(
                {
                    "channels": channels,
                    "scalars": np.full((minutes, len(SCALARS)), 0.5, np.float32),
                    "z": np.zeros((minutes, 5)),
                    "apart": np.zeros((minutes, 5)),
                    "patterns": np.zeros((minutes, 2), np.float32),
                    "minute": np.arange(minutes, dtype=np.int16),
                    "instrument": np.full(minutes, NAMES.index(t), np.int16),
                    "etf": np.full(minutes, t not in STOCKS),
                    "day": np.full(minutes, first + d, np.int32),
                }
            )
    return {k: np.concatenate([p[k] for p in parts]) for k in nd.FIELDS}


def series(rng, sessions: int, minutes: int) -> dict:
    """Moves apart from QQQ: each sector ETF its own; S00 follows XLE closely, S01 XLF less,
    S03 XLK a little; S02 none."""
    move = {
        t: rng.normal(size=(sessions, minutes)) for t in (*sd.SECTORS, "QQQ", "S02")
    }
    move["S00"] = move["XLE"] + 0.5 * rng.normal(size=(sessions, minutes))
    move["S01"] = move["XLF"] + 1.2 * rng.normal(size=(sessions, minutes))
    move["S03"] = 0.3 * move["XLK"] + rng.normal(size=(sessions, minutes))
    return move


def direct(x: np.ndarray, y: np.ndarray) -> float:
    x, y = x.astype(np.float32).ravel(), y.astype(np.float32).ravel()
    return float(np.corrcoef(x.astype(float), y.astype(float))[0, 1])


def hand_match(**instruments) -> dict:
    """A match written by hand: each instrument's (sector, r, wrong); each sector ETF's least
    related sector is the next one in SECTORS."""
    return {
        "instruments": {
            name: {"sector": s, "r": r, "wrong": w}
            for name, (s, r, w) in instruments.items()
        },
        "sectors": {
            s: {"wrong": sd.SECTORS[(i + 1) % len(sd.SECTORS)]}
            for i, s in enumerate(sd.SECTORS)
        },
    }


def rows_of(table: dict, ticker: str, day: int = 0) -> np.ndarray:
    return np.flatnonzero(
        (table["instrument"] == NAMES.index(ticker)) & (table["day"] == day)
    )


# --- the match ------------------------------------------------------------------------------


def test_each_instrument_is_matched_with_the_sector_it_moves_with():
    move = series(np.random.default_rng(3), 120, 10)
    found = sd.match(table_of(move), NAMES)
    got = found["instruments"]
    assert {n: m["sector"] for n, m in got.items()} == {
        "S00": "XLE",
        "S01": "XLF",
        "S02": got["S02"]["sector"],
        "S03": "XLK",
    }
    for name in ("S00", "S01", "S03"):
        m = got[name]
        assert m["r"] == pytest.approx(direct(move[name], move[m["sector"]]), abs=1e-12)
        rs = {s: direct(move[name], move[s]) for s in sd.SECTORS}
        assert m["correlations"] == pytest.approx(rs, abs=1e-12)
        assert m["wrong"] == min(rs, key=lambda s: abs(rs[s]))
        assert m["sessions"] == 120
    assert got["S00"]["r"] > 0.85 and 0.3 < got["S01"]["r"] < 0.8
    assert "QQQ" not in got and not set(sd.SECTORS) & set(got)
    for s in sd.SECTORS:
        rs = {o: direct(move[s], move[o]) for o in sd.SECTORS if o != s}
        assert found["sectors"][s]["wrong"] == min(rs, key=lambda o: abs(rs[o]))


def test_an_instrument_needs_100_sessions_for_a_match():
    move = series(
        np.random.default_rng(4), 120, 20
    )  # 99 sessions still give 1,980 minutes
    few = sd.match(table_of(move, present={"S00": range(99)}), NAMES)
    enough = sd.match(table_of(move, present={"S00": range(100)}), NAMES)
    assert "S00" not in few["instruments"]
    assert enough["instruments"]["S00"]["sessions"] == 100
    assert enough["instruments"]["S00"]["sector"] == "XLE"


def test_minutes_without_a_trade_count_for_nothing():
    move = series(np.random.default_rng(5), 120, 20)  # 1,560 minutes traded
    seen = {t: np.ones_like(v) for t, v in move.items()}
    seen["S00"][:, ::3] = 0.0
    planted = {t: v.copy() for t, v in move.items()}
    planted["S00"][:, ::3] = 9.0  # what an untraded minute carries must not count
    found = sd.match(table_of(planted, seen=seen), NAMES)
    traded = seen["S00"] > 0
    want = direct(move["S00"][traded], move["XLE"][traded])
    assert found["instruments"]["S00"]["r"] == pytest.approx(want, abs=1e-12)


def test_a_correlation_needs_1000_minutes_both_traded():
    move = series(np.random.default_rng(6), 120, 10)
    seen = {t: np.ones_like(v) for t, v in move.items()}
    for kept, matched in ((999, False), (1000, True)):
        seen["S00"] = np.zeros_like(move["S00"])
        seen["S00"].ravel()[:kept] = 1.0
        found = sd.match(table_of(move, seen=seen), NAMES)
        assert ("S00" in found["instruments"]) is matched


# --- the channel ----------------------------------------------------------------------------


def small(minutes: int = 6) -> tuple[dict, dict]:
    """Two sessions of known moves: each ticker's minute m of session d is a distinct value."""
    move = {
        t: np.array(
            [[(k + 1) * 0.1 + d + m * 0.01 for m in range(minutes)] for d in (0, 1)]
        )
        for k, t in enumerate(NAMES)
    }
    return move, {t: np.ones_like(v) for t, v in move.items()}


def test_a_stock_sees_its_sector_a_minute_late_scaled_by_r():
    move, seen = small()
    table = table_of(move, seen)
    found = hand_match(S00=("XLE", 0.5, "XLU"))
    got = sd.channels_of(table, found, NAMES)
    for d in (0, 1):
        at = rows_of(table, "S00", d)
        xle, xlu = (
            move[t][d][:-1].astype(np.float32).astype(float) for t in ("XLE", "XLU")
        )
        want = np.concatenate([[0.0], 0.5 * xle])
        wrong = np.concatenate([[0.0], 0.5 * xlu])
        assert np.array_equal(got["sector"][at], want.astype(np.float32))
        assert np.array_equal(got["wrong"][at], wrong.astype(np.float32))


def test_a_sector_etf_sees_the_mean_of_its_instruments_that_traded():
    move, seen = small()
    seen["S01"][0, 2] = 0.0  # S01 did not trade at minute 2 of session 0
    seen["S00"][0, 4] = seen["S01"][0, 4] = 0.0  # neither at minute 4
    table = table_of(move, seen)
    found = hand_match(S00=("XLE", 0.5, "XLU"), S01=("XLE", 0.25, "XLU"))
    got = sd.channels_of(table, found, NAMES)["sector"][rows_of(table, "XLE")]
    a, b = (move[t][0].astype(np.float32).astype(float) for t in ("S00", "S01"))
    want = [
        0.0,
        (0.5 * a[0] + 0.25 * b[0]) / 2,
        (0.5 * a[1] + 0.25 * b[1]) / 2,
        0.5 * a[2],
    ]
    want += [(0.5 * a[3] + 0.25 * b[3]) / 2, 0.0]
    assert np.array_equal(got, np.array(want, dtype=np.float32))


def test_the_wrong_sector_is_the_least_related_one_still_scaled_by_its_own_r():
    move, seen = small()
    table = table_of(move, seen)
    found = hand_match(S00=("XLE", 0.5, "XLU"), S01=("XLF", 0.4, "XLB"))
    got = sd.channels_of(table, found, NAMES)["wrong"]
    xle_wrong = found["sectors"]["XLE"]["wrong"]  # XLF, whose only instrument is S01
    want = 0.4 * move["S01"][0][:-1].astype(np.float32).astype(float)
    assert xle_wrong == "XLF"
    assert np.array_equal(
        got[rows_of(table, "XLE")], np.concatenate([[0.0], want]).astype(np.float32)
    )
    want = 0.4 * move["XLB"][0][:-1].astype(np.float32).astype(float)
    assert np.array_equal(
        got[rows_of(table, "S01")], np.concatenate([[0.0], want]).astype(np.float32)
    )


def test_zero_at_the_open_without_a_trade_and_without_a_match():
    move, seen = small()
    seen["XLE"][1, 3] = 0.0
    table = table_of(move, seen)
    found = hand_match(S00=("XLE", 0.5, "XLU"))
    got = sd.channels_of(table, found, NAMES)
    late = rows_of(table, "S00", 1)[4]
    assert got["sector"][late] == 0.0  # XLE did not trade at minute 3
    assert got["wrong"][late] == np.float32(0.5 * np.float32(move["XLU"][1, 3]))
    for kind in sd.KINDS:
        assert all(
            got[kind][rows_of(table, t, d)[0]] == 0.0 for t in NAMES for d in (0, 1)
        )
        for t in (
            "S01",
            "S02",
            "S03",
            "QQQ",
            "XLK",
        ):  # no match, or no instrument in it
            assert not got[kind][rows_of(table, t)].any()


def test_a_channel_is_clipped_as_every_input_is():
    move, seen = small()
    move["XLE"][0, 1], move["XLE"][0, 2] = 8.0, -8.0
    table = table_of(move, seen)
    got = sd.channels_of(table, hand_match(S00=("XLE", 1.5, "XLU")), NAMES)["sector"]
    assert got[rows_of(table, "S00")][2:4].tolist() == [10.0, -10.0]


def test_a_channel_reads_the_minute_before_and_nothing_later():
    rng = np.random.default_rng(7)
    move = {t: rng.normal(size=(3, 30)) for t in NAMES}
    seen = {t: (rng.random((3, 30)) > 0.2).astype(float) for t in NAMES}
    found = hand_match(
        S00=("XLE", 0.6, "XLU"), S01=("XLE", 0.3, "XLK"), S03=("XLF", 0.2, "XLE")
    )
    base = sd.channels_of(table_of(move, seen), found, NAMES)
    later = {t: v.copy() for t, v in move.items()}
    for v in later.values():
        v[:, 12:] = rng.normal(size=(3, 18))
    changed = sd.channels_of(table_of(later, seen), found, NAMES)
    minute = table_of(move, seen)["minute"]
    for kind in sd.KINDS:
        assert np.array_equal(base[kind][minute <= 12], changed[kind][minute <= 12])
        assert not np.array_equal(base[kind][minute == 13], changed[kind][minute == 13])
    before = {t: v.copy() for t, v in move.items()}
    for v in before.values():
        v[:, 11] += 1.0
    moved = sd.channels_of(table_of(before, seen), found, NAMES)["sector"]
    assert np.array_equal(base["sector"][minute != 12], moved[minute != 12])
    assert not np.array_equal(base["sector"][minute == 12], moved[minute == 12])


def test_the_build_s_two_reckonings_agree_bit_for_bit():
    rng = np.random.default_rng(8)
    move = {t: rng.normal(size=(4, 25)) * 3 for t in NAMES}
    seen = {t: (rng.random((4, 25)) > 0.3).astype(float) for t in NAMES}
    table = table_of(move, seen, present={"S02": range(2)})
    found = hand_match(
        S00=("XLE", 0.71, "XLU"),
        S01=("XLE", 0.33, "XLK"),
        S02=("XLE", 0.07, "XLP"),
        S03=("XLV", 0.52, "XLB"),
    )
    rows = np.arange(len(table["day"]))
    fast, slow = (
        sd.channels_of(table, found, NAMES),
        sd.reckoned(table, found, NAMES, rows),
    )
    for kind in sd.KINDS:
        assert fast[kind].dtype == np.float32
        assert np.array_equal(fast[kind], slow[kind])
        assert np.count_nonzero(fast[kind]) > len(rows) // 4


def test_the_checked_days_spread_over_the_year():
    move = {t: np.zeros((40, 3)) for t in ("QQQ", "S00")}
    table = table_of(move, first=100)
    picked = np.unique(table["day"][sd.checked_rows(table)])
    assert picked.tolist() == [100, 110, 120, 129, 139]
    assert len(sd.checked_rows(table)) == sd.CHECKED_DAYS * 2 * 3


# --- the inputs -----------------------------------------------------------------------------


def test_the_inputs_are_the_ridge_s_971_then_the_channel_s_window():
    rng = np.random.default_rng(9)
    move = {t: rng.normal(size=(2, 150)) for t in NAMES}
    table = table_of(move)
    table["scalars"] = rng.normal(size=table["scalars"].shape).astype(np.float32)
    sector = {
        k: rng.normal(size=len(table["day"])).astype(np.float32) for k in sd.KINDS
    }
    rows = np.array([0, 5, 119, 120, 149, 150, 151, 299])
    for kind in sd.KINDS:
        got = sd.inputs_of(table, sector, rows, kind)
        assert got.shape == (len(rows), sd.INPUTS) and sd.INPUTS == 1_091
        assert np.array_equal(got[:, :971], nd.inputs_of(table, rows))
        for n, r in enumerate(rows):
            m = int(table["minute"][r])
            for back in range(WINDOW):
                want = sector[kind][r - back] if back <= m else 0.0
                assert got[n, 971 + WINDOW - 1 - back] == want


# --- the build ------------------------------------------------------------------------------


def stored(tmp_path: Path, tables: dict) -> Path:
    """A data folder with the network's rows: tables[(source, year)], recorded as built by
    today's code."""
    data = tmp_path / "data"
    for source in nd.SOURCES:
        out = nd.folder(data, source)
        out.mkdir(parents=True)
        record = {}
        for (src, year), table in tables.items():
            if src == source:
                np.savez(out / f"{year}.npz", **table)
                record[str(year)] = {
                    "rows": len(table["day"]),
                    "code": nd.fingerprint(),
                }
        (out / "manifest.json").write_text(json.dumps(record))
    (data / "kutbit" / "net" / "tickers.json").write_text(json.dumps(NAMES))
    return data


def years_of(seed: int, years=nd.YEARS) -> dict:
    """Both sources' rows for `years`: 40 sessions of 10 minutes a year, the walks unlike the
    real prices; S03 lists only from 2019."""
    rng = np.random.default_rng(seed)
    tables = {}
    for source in nd.SOURCES:
        for year in years:
            move = series(rng, 40, 10)
            del move["QQQ"]
            present = {"S03": range(40 if year >= 2019 else 0)}
            first = date(year, 1, 2).toordinal()
            tables[(source, year)] = table_of(move, present=present, first=first)
    return tables


def test_the_build_matches_once_on_the_real_2016_2018_and_loads_in_row_order(tmp_path):
    tables = years_of(10)
    data = stored(tmp_path, tables)
    sd.build(data, log=quiet)
    found = json.loads((data / "kutbit" / "sector" / "match.json").read_text())
    real = {
        k: np.concatenate([tables[("real", y)][k] for y in sd.MATCH_YEARS])
        for k in nd.FIELDS
    }
    assert found == json.loads(json.dumps(sd.match(real, NAMES)))
    assert (
        found["instruments"]["S00"]["sector"] == "XLE"
        and "S03" not in found["instruments"]
    )
    for source in nd.SOURCES:
        got = sd.load(data, source, (2019, 2020))
        for kind in sd.KINDS:
            want = [
                sd.channels_of(tables[(source, y)], found, NAMES)[kind]
                for y in (2019, 2020)
            ]
            assert np.array_equal(got[kind], np.concatenate(want))
    walks = sd.load(data, "random", (2016,))["sector"]
    assert not np.array_equal(walks, sd.load(data, "real", (2016,))["sector"])


def test_a_year_built_already_is_not_built_again(tmp_path):
    data = stored(tmp_path, years_of(11))
    sd.build(data, log=quiet)
    said = []
    sd.build(data, log=said.append)
    assert sum("already" in line for line in said) == 2 * len(nd.YEARS)


def test_a_channel_that_differs_from_its_reckoning_stops_the_build(
    tmp_path, monkeypatch
):
    data = stored(tmp_path, years_of(12))
    real = sd.channels_of

    def planted(table, found, names):
        out = real(table, found, names)
        out["wrong"] = out["wrong"].copy()
        out["wrong"][sd.checked_rows(table)[-1]] += np.float32(1e-3)
        return out

    monkeypatch.setattr(sd, "channels_of", planted)
    with pytest.raises(ValueError, match="wrong channel differs"):
        sd.build(data, log=quiet)


def test_load_refuses_other_code_other_rows_or_another_match(tmp_path, monkeypatch):
    data = stored(tmp_path, years_of(13))
    sd.build(data, log=quiet)
    sd.load(data, "real", (2018,))
    match = data / "kutbit" / "sector" / "match.json"
    kept = match.read_text()
    found = json.loads(kept)
    found["instruments"]["S00"]["r"] += 1e-9
    match.write_text(json.dumps(found))
    with pytest.raises(ValueError, match="not built from today's"):
        sd.load(data, "real", (2018,))
    match.write_text(kept)
    record = nd.folder(data, "real") / "manifest.json"
    rows = json.loads(record.read_text())
    rows["2018"]["code"] = {"net_data.py": "another"}
    record.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="not built from today's"):
        sd.load(data, "real", (2018,))
    sd.load(data, "real", (2017,))
    monkeypatch.setattr(sd, "fingerprint", lambda: {"sector_data.py": "another"})
    with pytest.raises(ValueError, match="not built from today's"):
        sd.load(data, "random", (2017,))


def test_years_after_2021_are_never_read(tmp_path):
    data = stored(tmp_path, years_of(14))
    with pytest.raises(ValueError, match="2022-2026"):
        sd.build(data, years=(2021, 2022), log=quiet)
    sd.build(data, log=quiet)
    with pytest.raises(ValueError, match="2022-2026"):
        sd.load(data, "real", (2022,))


def test_the_log_shows_the_match_and_the_rows_without_one(tmp_path):
    data = stored(tmp_path, years_of(15))
    said = []
    sd.build(data, log=said.append)
    found = json.loads((data / "kutbit" / "sector" / "match.json").read_text())
    used = {m["sector"] for m in found["instruments"].values()}
    assert said[0].startswith("the match on 2016-2018:")
    assert "  XLE: S00 0." in said[0]
    assert all((f"  {s}: none" in said[0]) is (s not in used) for s in sd.SECTORS)
    assert said[0].endswith(f"  {len(found['instruments'])} instruments matched")
    line = next(s for s in said if s.startswith("real 2019:"))
    assert line.startswith("real 2019: 5,200 rows; 7.7% without a match (")
    assert next(s for s in said if s.startswith("real 2018:")).startswith(
        "real 2018: 4,800 rows; 0.0% without a match ("
    )


def test_a_year_made_of_other_inputs_is_built_again(tmp_path):
    data = stored(tmp_path, years_of(16))
    sd.build(data, log=quiet)
    record = sd.folder(data, "random") / "manifest.json"
    made = json.loads(record.read_text())
    made["2020"]["made"]["match"] = "another"
    record.write_text(json.dumps(made))
    said = []
    sd.build(data, log=said.append)
    again = [
        s
        for s in said
        if not s.endswith("already") and s.startswith(("real", "random"))
    ]
    assert len(again) == 1 and again[0].startswith("random 2020: ")
    sd.load(data, "random", (2020,))


def test_each_source_s_year_answers_to_its_own_rows(tmp_path):
    data = stored(tmp_path, years_of(17))
    sd.build(data, log=quiet)
    record = nd.folder(data, "random") / "manifest.json"
    rows = json.loads(record.read_text())
    rows["2019"]["code"] = {"net_data.py": "another"}
    record.write_text(json.dumps(rows))
    sd.load(data, "real", (2019,))
    with pytest.raises(ValueError, match="random \\[2019\\]: not built from today's"):
        sd.load(data, "random", (2019,))
