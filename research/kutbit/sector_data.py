"""The sector channel of KUTBIT's third variant (decision 0028).

Each instrument sees only itself and QQQ in the skill test's 971 inputs. Variant 3 adds one
channel, `sector`: the move apart from QQQ, a minute old, of the instrument's sector.

- **The match**, made on the real rows of 2016-2018 only. Every instrument but QQQ and the nine
  sector ETFs is matched with the sector ETF whose `apart` correlates most with its own, minute
  by minute, over the minutes when both traded; `r` is that correlation. An instrument with
  fewer than MIN_SESSIONS sessions there has no match. For the wrong sector, shown beside the
  gates, each matched instrument also keeps the sector ETF whose correlation with it is nearest
  zero, and each sector ETF the other sector ETF nearest zero to it.
- **The channel** at minute t: for a matched instrument, r times its sector ETF's `apart` at
  minute t - 1; for a sector ETF, the mean of r times `apart` at minute t - 1 over its matched
  instruments that traded then; zero where that minute had no trade, in a session's first
  minute, and for an instrument without a match. It is clipped as every input is. `wrong` is
  the same with each instrument's least related sector, still scaled by its own r.
- **The build** makes the match once, from the real prices, and each year's channels of
  2016-2021 from the network's stored rows (`net_data.py`), on the real prices and on the skill
  test's random walks, which keep the real match and its r. Some days of each year are reckoned
  again one row at a time and must agree bit for bit, or the build stops. Each year goes to
  DATA/kutbit/sector/<real|random>/<year>.npz, a value a row in the network's row order, with a
  manifest of its rows and of the code it was built from. 2022-2026 are not read.

A row's 1,091 inputs are the skill test's 971 and then the channel's 120-minute window
(`inputs_of`).

    uv run python research/kutbit/sector_data.py build [DATA]
"""

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import net_data
import numpy as np
import skill

from qfin_alphaguard.kutbit.features import CHANNELS, SCALARS, WINDOW

DATA = Path("data")
HERE = Path(__file__).parent
SECTORS = ("XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY")
MARKET = "QQQ"
MATCH_YEARS = skill.FIT_YEARS  # 2016-2018, the match's only years
MIN_SESSIONS = 100  # sessions in MATCH_YEARS an instrument needs for a match
MIN_MINUTES = 1_000  # minutes when both traded, for a correlation to count
KINDS = ("sector", "wrong")
CHECKED_DAYS = 5  # days of each year reckoned again one row at a time
APART, SEEN = CHANNELS.index("apart"), CHANNELS.index("seen")
SPAN = 400  # more than a session's minutes: a day's minutes are day * SPAN + minute
INPUTS = net_data.WINDOW_INPUTS + len(SCALARS) + WINDOW  # 1,091
CODE = {"sector_data.py": HERE / "sector_data.py"}


# --- the match ------------------------------------------------------------------------------


def grid_of(table: dict, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each minute's `apart` of each instrument, where it traded: the minutes' keys in order,
    each row's place among them, and [minutes, count] values, NaN where an instrument has no
    row or its minute had no trade."""
    key = table["day"].astype(np.int64) * SPAN + table["minute"]
    keys, at = np.unique(key, return_inverse=True)
    grid = np.full((len(keys), count), np.nan)
    traded = table["channels"][:, SEEN] > 0.5
    grid[at[traded], table["instrument"][traded]] = table["channels"][traded, APART]
    return keys, at, grid


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    """The correlation of two instruments' minutes when both traded, or NaN with too few."""
    both = np.isfinite(x) & np.isfinite(y)
    if both.sum() < MIN_MINUTES:
        return math.nan
    return float(np.corrcoef(x[both], y[both])[0, 1])


def nearest(values: dict, best: bool) -> str | None:
    """The key of the highest value, or of the one nearest zero; NaN counts for neither, and a
    tie keeps the earlier."""
    finite = {k: v for k, v in values.items() if math.isfinite(v)}
    if not finite:
        return None
    if best:
        return max(finite, key=finite.get)
    return min(finite, key=lambda k: abs(finite[k]))


def match(table: dict, names: list[str]) -> dict:
    """Decision 0028's match on `table`, the real rows of MATCH_YEARS: each instrument's sector,
    its r, its least related sector and every correlation; and each sector ETF's least related
    other sector ETF."""
    _, _, grid = grid_of(table, len(names))
    pairs = np.unique(table["instrument"].astype(np.int64) * 10**7 + table["day"])
    present, sessions_of = np.unique(pairs // 10**7, return_counts=True)
    col = {s: names.index(s) for s in SECTORS}
    found: dict = {"instruments": {}, "sectors": {}}
    for i, sessions in zip(present, sessions_of.tolist(), strict=True):
        name = names[i]
        if name == MARKET or name in SECTORS or sessions < MIN_SESSIONS:
            continue
        rs = {s: correlation(grid[:, i], grid[:, c]) for s, c in col.items()}
        sector = nearest(rs, best=True)
        if sector is None:
            continue
        found["instruments"][name] = {
            "sector": sector,
            "r": rs[sector],
            "wrong": nearest(rs, best=False),
            "sessions": sessions,
            "correlations": rs,
        }
    for s in SECTORS:
        rs = {
            o: correlation(grid[:, col[s]], grid[:, c])
            for o, c in col.items()
            if o != s
        }
        found["sectors"][s] = {"wrong": nearest(rs, best=False), "correlations": rs}
    return found


def members(found: dict, names: list[str]) -> dict[str, list[str]]:
    """Each sector ETF's matched instruments, in the order of their numbers."""
    group: dict[str, list[str]] = {s: [] for s in SECTORS}
    for name in sorted(found["instruments"], key=names.index):
        group[found["instruments"][name]["sector"]].append(name)
    return group


# --- the channel ----------------------------------------------------------------------------


def per_minute(
    grid: np.ndarray, found: dict, names: list[str], kind: str
) -> np.ndarray:
    """What each instrument's channel `kind` is at the minute after each minute of `grid`,
    [minutes, instruments], before the clip."""
    col = {t: i for i, t in enumerate(names)}
    out = np.zeros(grid.shape)
    for name, m in found["instruments"].items():
        partner = m["sector"] if kind == "sector" else m["wrong"]
        if name in col and partner in col:
            out[:, col[name]] = np.nan_to_num(m["r"] * grid[:, col[partner]], nan=0.0)
    group = members(found, names)
    for s in SECTORS:
        if s not in col:
            continue
        source = s if kind == "sector" else found["sectors"][s]["wrong"]
        total, count = np.zeros(len(grid)), np.zeros(len(grid))
        for name in group.get(source, []):
            scaled = found["instruments"][name]["r"] * grid[:, col[name]]
            traded = np.isfinite(scaled)
            total = total + np.where(traded, scaled, 0.0)
            count = count + traded
        out[:, col[s]] = np.where(count > 0, total / np.maximum(count, 1), 0.0)
    return out


def channels_of(table: dict, found: dict, names: list[str]) -> dict[str, np.ndarray]:
    """Each row's channels, `sector` and `wrong`, as float32: from the minute before the row's,
    in the same session."""
    keys, at, grid = grid_of(table, len(names))
    own = keys[at]
    before = np.searchsorted(keys, own - 1)
    has = np.zeros(len(own), dtype=bool)
    inside = before < len(keys)
    has[inside] = keys[before[inside]] == own[inside] - 1
    has &= table["minute"] > 0
    out = {}
    for kind in KINDS:
        values = per_minute(grid, found, names, kind)
        row = np.zeros(len(own))
        row[has] = values[before[has], table["instrument"][has]]
        out[kind] = np.clip(row, -skill.CLIP_INPUT, skill.CLIP_INPUT).astype(np.float32)
    return out


def reckoned(table: dict, found: dict, names: list[str], rows: np.ndarray) -> dict:
    """The channels of `rows` reckoned again, one row at a time, from a dictionary of each
    traded minute's `apart`: the build's check of `channels_of`."""
    day, minute = table["day"][rows], table["minute"][rows]
    instrument = table["instrument"][rows]
    same = np.isin(table["day"], np.unique(day))
    apart = {}
    for k in np.flatnonzero(same):
        if table["channels"][k, SEEN] > 0.5:
            key = (
                int(table["day"][k]),
                int(table["minute"][k]),
                int(table["instrument"][k]),
            )
            apart[key] = float(table["channels"][k, APART])
    group = members(found, names)
    col = {t: i for i, t in enumerate(names)}
    out = {kind: np.zeros(len(rows), dtype=np.float32) for kind in KINDS}
    for n in range(len(rows)):
        d, m, name = int(day[n]), int(minute[n]), names[instrument[n]]
        for kind in KINDS:
            value = 0.0
            if m > 0 and name in found["instruments"]:
                own = found["instruments"][name]
                partner = own["sector"] if kind == "sector" else own["wrong"]
                seen = apart.get((d, m - 1, col[partner]))
                if seen is not None:
                    value = own["r"] * seen
            elif m > 0 and name in SECTORS:
                source = name if kind == "sector" else found["sectors"][name]["wrong"]
                total, count = 0.0, 0
                for other in group[source]:
                    seen = apart.get((d, m - 1, col[other]))
                    if seen is not None:
                        total += found["instruments"][other]["r"] * seen
                        count += 1
                value = total / count if count else 0.0
            clipped = min(max(value, -skill.CLIP_INPUT), skill.CLIP_INPUT)
            out[kind][n] = np.float32(clipped)
    return out


def checked_rows(table: dict) -> np.ndarray:
    """The rows of CHECKED_DAYS days spread over the table's year."""
    days = np.unique(table["day"])
    picked = days[
        np.unique(np.linspace(0, len(days) - 1, CHECKED_DAYS).round().astype(int))
    ]
    return np.flatnonzero(np.isin(table["day"], picked))


# --- the inputs -----------------------------------------------------------------------------


def inputs_of(
    net: dict, sector: dict, rows: np.ndarray, kind: str = "sector"
) -> np.ndarray:
    """The 1,091 inputs of `rows`: the skill test's 971, then the channel `kind`'s window of the
    WINDOW minutes up to each row, oldest first, with zeros before its session's open."""
    window = net_data.windows_of(sector[kind][:, None], net["minute"], rows)
    return np.concatenate(
        [net_data.inputs_of(net, rows), window.reshape(len(rows), WINDOW)], axis=1
    )


# --- the build ------------------------------------------------------------------------------


def fingerprint() -> dict[str, str]:
    """The SHA-256 of the code that builds the channels."""
    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in CODE.items()
    }


def folder(data: Path, source: str) -> Path:
    return data / "kutbit" / "sector" / source


def manifest(data: Path, source: str) -> dict:
    """Each built year's rows and the code, the network's rows and the match it came from."""
    path = folder(data, source) / "manifest.json"
    return json.loads(path.read_text()) if path.exists() else {}


def made_of(data: Path, source: str, year: int, found: dict) -> dict:
    """What a year's channels are made of: this code, the network's rows of that year, and
    the match."""
    return {
        "code": fingerprint(),
        "rows": net_data.manifest(data, source).get(str(year), {}).get("code"),
        "match": hashlib.sha256(json.dumps(found, sort_keys=True).encode()).hexdigest(),
    }


def build(data: Path, years=net_data.YEARS, log=print) -> Path:
    """The match, on the real rows of MATCH_YEARS, then each year's channels of `years` for
    both sources, each year checked on some days row by row. A year already built from the
    same code, rows and match is skipped."""
    net_data.unread(years)
    names = json.loads((data / "kutbit" / "net" / "tickers.json").read_text())
    found = match(net_data.load(data, "real", MATCH_YEARS), names)
    out = data / "kutbit" / "sector"
    out.mkdir(parents=True, exist_ok=True)
    (out / "match.json").write_text(json.dumps(found, indent=1))
    log(said(found, names))
    for source in net_data.SOURCES:
        folder(data, source).mkdir(parents=True, exist_ok=True)
        record = manifest(data, source)
        for year in years:
            path = folder(data, source) / f"{year}.npz"
            made = made_of(data, source, year, found)
            if path.exists() and record.get(str(year), {}).get("made") == made:
                log(
                    f"{source} {year}: built from the same code, rows and match already"
                )
                continue
            started = time.time()
            table = net_data.load(data, source, [year])
            values = channels_of(table, found, names)
            rows = checked_rows(table)
            again = reckoned(table, found, names, rows)
            for kind in KINDS:
                if not np.array_equal(values[kind][rows], again[kind]):
                    raise ValueError(
                        f"{source} {year}: the {kind} channel differs from its reckoning "
                        "row by row"
                    )
            partial = folder(data, source) / f"{year}.part.npz"
            np.savez(partial, **values)
            partial.replace(path)
            record[str(year)] = {"rows": len(table["day"]), "made": made}
            (folder(data, source) / "manifest.json").write_text(
                json.dumps(record, indent=1)
            )
            log(
                f"{source} {year}: {len(table['day']):,} rows; "
                f"{unmatched(table, found, names):.1%} without a match "
                f"({time.time() - started:.0f} s)"
            )
    return out


def unmatched(table: dict, found: dict, names: list[str]) -> float:
    """The share of `table`'s rows whose instrument has no match and is no sector ETF."""
    has = {names.index(t) for t in (*found["instruments"], *SECTORS) if t in names}
    return float(np.mean(~np.isin(table["instrument"], sorted(has))))


def load(data: Path, source: str, years) -> dict[str, np.ndarray]:
    """The channels of `years` from `source`, oldest first, in the network's row order. Each
    year must have been built from today's code, today's rows of the network and today's
    match."""
    net_data.unread(years)
    found = json.loads((data / "kutbit" / "sector" / "match.json").read_text())
    record = manifest(data, source)
    stale = [
        y
        for y in years
        if record.get(str(y), {}).get("made") != made_of(data, source, y, found)
    ]
    if stale:
        raise ValueError(
            f"{source} {stale}: not built from today's code, rows and match; "
            "run sector_data.py build"
        )
    parts = []
    for year in years:
        with np.load(folder(data, source) / f"{year}.npz") as stored:
            parts.append({k: stored[k] for k in KINDS})
    return {k: np.concatenate([p[k] for p in parts]) for k in KINDS}


def said(found: dict, names: list[str]) -> str:
    """The match, a line a sector ETF."""
    group = members(found, names)
    lines = [f"the match on {MATCH_YEARS[0]}-{MATCH_YEARS[-1]}:"]
    for s in SECTORS:
        inside = sorted(group[s], key=lambda t: -found["instruments"][t]["r"])
        shown = ", ".join(f"{t} {found['instruments'][t]['r']:.2f}" for t in inside)
        lines.append(f"  {s}: {shown or 'none'}")
    lines.append(f"  {len(found['instruments'])} instruments matched")
    return "\n".join(lines)


def main(argv: list[str]) -> None:
    if not argv or argv[0] != "build":
        raise SystemExit("usage: sector_data.py build [DATA]")
    data = Path(argv[1]) if len(argv) > 1 else DATA
    print(f"build: {build(data)}")


if __name__ == "__main__":
    main(sys.argv[1:])
