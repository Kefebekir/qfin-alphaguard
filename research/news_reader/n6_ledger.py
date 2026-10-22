"""N6's development variants in the ledger, and the choice among them (pre-registration N6).

At most LIMIT variants on 2018-2021, each written in LEDGER.md before it runs. A variant that
started counts, finished or not, since a run prints each year as it goes; one that did not
finish may be run again under its name and settings, as the same variant. Once the choice is
written (variants/N6_chosen.json), no variant runs. The choice: of the finished variants whose
mean a day after costs over the four years is above both their placebos', the highest.
"""

from pathlib import Path

LEDGER = Path("research/news_reader/LEDGER.md")
CHOSEN = Path("research/news_reader/variants/N6_chosen.json")
HEADING = "## N6's variants on the development years (2018-2021)"
HEADER = "| # | Name | Run (UTC) | Settings | The variant (t) | Dealt | Context alone | Gate |"
LIMIT = 10
RUNNING, UNFINISHED = "running", "did not finish"
RULE = (
    "the highest mean a day after costs over 2018-2021 of the variants above both their "
    "placebos (pre-registration N6)"
)


def settings_text(settings: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in settings.items())


def _table(ledger: str) -> tuple[list[str], int, int]:
    """The ledger's lines and the first and last-plus-one of the table's rows."""
    lines = ledger.split("\n")
    if HEADING not in lines:
        return lines, len(lines), len(lines)
    first = lines.index(HEADING)
    while not lines[first].startswith("| # |"):
        first += 1
    first += 2  # the header and its rule
    last = first
    while last < len(lines) and lines[last].startswith("|"):
        last += 1
    return lines, first, last


def rows(ledger: str) -> list[dict]:
    """The variants as the ledger lists them."""
    lines, first, last = _table(ledger)
    out = []
    for line in lines[first:last]:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        out.append(
            {
                "n": int(cells[0]),
                "name": cells[1].strip("`"),
                "settings": cells[3],
                "result": cells[4],
            }
        )
    return out


def _with_section(ledger: str) -> str:
    if HEADING in ledger.split("\n"):
        return ledger
    return (
        ledger.rstrip("\n")
        + f"\n\n{HEADING}\n\nEach written before it runs (n6_run.py); at most {LIMIT}. The "
        "mean a day after costs over the four years, beside its two placebos.\n\n"
        + HEADER
        + "\n|---|---|---|---|---|---|---|---|\n"
    )


def _row(n: int, name: str, when: str, settings: str, cells: list[str]) -> str:
    return f"| {n} | `{name}` | {when} | {settings} | " + " | ".join(cells) + " |"


def _replace(ledger: str, n: int, row: str) -> str:
    lines, first, last = _table(ledger)
    for i in range(first, last):
        if lines[i].startswith(f"| {n} |"):
            lines[i] = row
            return "\n".join(lines)
    raise ValueError(f"no row for N6's variant {n}")


def start(
    ledger: str, name: str, settings: dict, when: str, chosen: Path | None = None
) -> tuple[str, int]:
    """The ledger with the variant written as running, and its number. Refuses a variant
    after the choice, an eleventh, a name or settings already run."""
    chosen = chosen or CHOSEN
    if chosen.exists():
        raise SystemExit(f"{chosen} is written: N6's development is over")
    text = settings_text(settings)
    known = rows(ledger)
    for r in known:
        if r["name"] == name:
            if r["result"] == UNFINISHED and r["settings"] == text:
                row = _row(r["n"], name, when, text, [RUNNING, "", "", ""])
                return _replace(ledger, r["n"], row), r["n"]
            raise SystemExit(f"{name} is variant {r['n']} already: a name runs once")
        if r["settings"] == text:
            raise SystemExit(
                f"these settings are variant {r['n']}, {r['name']}, already"
            )
    if len(known) >= LIMIT:
        raise SystemExit(f"N6 has its {LIMIT} variants; the next step is the choice")
    n = len(known) + 1
    ledger = _with_section(ledger)
    lines, _, last = _table(ledger)
    lines.insert(last, _row(n, name, when, text, [RUNNING, "", "", ""]))
    return "\n".join(lines), n


def finish(ledger: str, n: int, result: dict) -> str:
    """The variant's row with its result."""

    def cell(label: str) -> str:
        v = result["all"][label]
        return f"{v['bps']:+.2f} bps (t {v['t']:+.2f})".replace("-", "−")

    return _replace(
        ledger,
        n,
        _row(
            n,
            result["name"],
            result["run_utc"],
            settings_text(result["settings"]),
            [
                cell("model"),
                cell("dealt"),
                cell("silent"),
                "passed" if result["gate"] else "not passed",
            ],
        ),
    )


def unfinished(ledger: str, n: int) -> str:
    """The variant's row marked as not finished; it still counts."""
    lines, first, last = _table(ledger)
    for i in range(first, last):
        if lines[i].startswith(f"| {n} |"):
            cells = lines[i].split("|")
            cells[5] = f" {UNFINISHED} "
            lines[i] = "|".join(cells)
            return "\n".join(lines)
    raise ValueError(f"no row for N6's variant {n}")


def choose(ledger: str, results: dict[str, dict]) -> dict:
    """The pre-registered choice among the finished variants, from their stored results."""
    finished = [r for r in rows(ledger) if r["result"] not in (RUNNING, UNFINISHED)]
    through = [(r, results[r["name"]]) for r in finished if results[r["name"]]["gate"]]
    if not through:
        raise SystemExit(
            "no variant is above both its placebos: N6 fails at development"
        )
    row, best = max(through, key=lambda pair: pair[1]["all"]["model"]["bps"])
    return {
        "variant": row["n"],
        "name": best["name"],
        "settings": best["settings"],
        "seeds": best["seeds"],
        "development": best["all"]["model"],
        "rule": RULE,
    }
