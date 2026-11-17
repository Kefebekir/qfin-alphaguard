"""What KUTBIT's targets look like on the development years, 2016-2021 (decision 0023).

Nothing is predicted here. For each horizon, on stocks' usable full sessions drawn at random:
how far the hedged position moves, against a round trip's cost, and whether the target `z` is
in the same units at the open, at midday and near the close.

Usage: uv run python research/kutbit/target_report.py [DATA]   (DATA, the data folder, is data/)
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
from dataset import MARKET, day_of
from labels import HORIZONS, targets

from qfin_alphaguard.kutbit.features import FULL_SESSION

DATA = Path("data")
RESULTS = Path(__file__).parent / "results_target.txt"
LAST_YEAR = 2021  # the years after it are kept for the second look
SESSIONS = 3000
COST = 2.0  # basis points for a round trip, decision 0023's figure
PARTS = (("09:30-10:00", 0, 30), ("12:00-13:00", 150, 210), ("15:00-15:30", 330, 360))


def kurtosis(x: np.ndarray) -> float:
    x = x - x.mean()
    return float((x**4).mean() / (x**2).mean() ** 2)


def report(days: pl.DataFrame, arrays: dict, sessions: int = SESSIONS) -> str:
    stocks = days.filter(
        pl.col("usable")
        & ~pl.col("etf")
        & (pl.col("minutes") == FULL_SESSION)
        & (pl.col("year") <= LAST_YEAR)
    )
    sample = stocks.sample(min(sessions, stocks.height), seed=1)
    found = {h: {"apart": [], "ret": [], "z": []} for h in HORIZONS}
    for row in sample.iter_rows(named=True):
        day = day_of(days, arrays, row["ticker"], row["day"])
        market = day_of(days, arrays, MARKET, row["day"])
        for h, labels in targets(day, market).items():
            for name in found[h]:
                found[h][name].append(labels[name][:-1])
    lines = [
        "target_report.py: KUTBIT's targets on 2016-2021 (decision 0023)",
        "",
        f"{len(sample)} usable full sessions of stocks, drawn at random; every minute of each.",
        "",
        "A long position hedged with QQQ, by how long it is held. `needed` is the share of the",
        f"move's size that a forecast must capture to pay a round trip of {COST:g} bps.",
        "",
        "minutes  move alone  hedged  needed   size of z at:  "
        + "  ".join(name for name, _, _ in PARTS)
        + "   kurtosis of z",
    ]
    for h in HORIZONS:
        apart, ret, z = (np.array(found[h][name]) for name in ("apart", "ret", "z"))
        sizes = "  ".join(f"{z[:, a:b].std():>11.2f}" for _, a, b in PARTS)
        lines.append(
            f"{h:>7}  {ret.std() * 1e4:>7.1f}bps  {apart.std() * 1e4:>4.1f}bps  "
            f"{COST / (apart.std() * 1e4):>5.0%}                  {sizes}   {kurtosis(z):>13.1f}"
        )
    return "\n".join(lines) + "\n"


def main(data: Path = DATA) -> None:
    days = pl.read_parquet(data / "kutbit" / "days.parquet")
    years = range(days["year"].min(), LAST_YEAR + 1)
    arrays = {}
    for year in years:  # read once: a field of an .npz file is unpacked at every access
        with np.load(data / "kutbit" / f"{year}.npz") as stored:
            arrays[year] = {name: stored[name] for name in stored.files}
    text = report(days, arrays)
    RESULTS.write_text(text)
    print(text)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DATA)
