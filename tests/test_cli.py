import json
import shutil
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from qfin_alphaguard import cli
from qfin_alphaguard.cli import build_parser, main
from qfin_alphaguard.data.eodhd import MINUTE_SCHEMA
from qfin_alphaguard.plan import plan_from_json
from qfin_alphaguard.sessions import session_days, sessions

AWS = Path(__file__).resolve().parents[1] / "infra" / "aws"
TASK_DEFINITION = AWS / "task-definition.template.json"


def test_ingest_synthetic_writes_parquet(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    exit_code = main(["ingest", "--synthetic"])

    assert exit_code == 0
    assert (tmp_path / "data" / "raw" / "prices_synthetic.parquet").exists()


@pytest.mark.parametrize("argv", [[], ["--synthetic"]])
def test_a_command_is_required(argv):
    # ["--synthetic"] is the old form of `qfin ingest --synthetic`.
    with pytest.raises(SystemExit) as stopped:
        main(argv)
    assert stopped.value.code == 2  # argparse's exit code for a usage error


def test_backtest_needs_the_daily_prices_first(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["backtest"]) == 1
    assert "run `qfin ingest` first" in capsys.readouterr().err


def test_backtest_reports_the_runs_and_writes_the_same_json_every_time(
    store, tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    guard = str(Path(__file__).resolve().parents[1] / "guard.yaml")
    argv = ["backtest", "--capital", "1000", "2000", "--guard", guard]

    assert main(argv) == 0
    first = (
        tmp_path / "data/backtests/backtest_2016-01-05_2016-01-07.json"
    ).read_text()
    output = capsys.readouterr().out
    assert main([*argv, "--out", "again.json"]) == 0
    again = (tmp_path / "again.json").read_text()

    assert first == again
    report = json.loads(first)
    assert set(report["plan"]) == {"1000", "2000"}
    assert set(report["baselines"]) == {"equal weight", "SPY"}
    for line in ("Plan, 1,000 USD", "Plan, 2,000 USD", "Same stocks, equal weight"):
        assert line in output


def test_aws_task_definition_runs_a_valid_command():
    # The nightly AWS job runs this command. If the CLI changes and the
    # template does not, this test fails before the job does.
    template = json.loads(TASK_DEFINITION.read_text(encoding="utf-8"))
    command = template["containerDefinitions"][0]["command"]

    args = build_parser().parse_args(command)

    assert args.command == "ingest"
    assert args.s3_bucket


def test_aws_task_gets_the_eodhd_key_it_may_read():
    # qfin ingest needs EODHD_API_KEY; ECS reads it from SSM with the execution
    # role, which may read that one parameter and nothing else.
    container = json.loads(TASK_DEFINITION.read_text(encoding="utf-8"))
    (secret,) = container["containerDefinitions"][0]["secrets"]
    policy = json.loads(
        (AWS / "execution-role-secrets-policy.json").read_text(encoding="utf-8")
    )
    (statement,) = policy["Statement"]

    assert secret["name"] == "EODHD_API_KEY"
    assert statement["Action"] == "ssm:GetParameters"
    assert secret["valueFrom"].replace("ACCOUNT_ID", "*") == statement["Resource"]


def test_intraday_needs_the_daily_prices_first(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["intraday"]) == 1
    assert "run `qfin ingest` first" in capsys.readouterr().err


def session_closes(code, first, last, close=600.0):
    """Daily bars closing at `close` on every session from `first` to `last`."""
    days = session_days(first, last)
    n = len(days)
    return pl.DataFrame(
        {
            "date": days,
            "ticker": [code] * n,
            "close": [close] * n,
            "adjustment": [1.0] * n,
        }
    )


class FakeProvider:
    """Stands in for EodhdClient or AlpacaClient.

    It has a minute at the end of each session, at the day's close as traded
    in `daily`, except on the days in `lacks`.
    """

    def __init__(self, name, daily, lacks=()):
        self.name = name
        self.daily = daily
        self.lacks = list(lacks)
        self.asked = []

    def daily_bars(self, code, start, end):
        return self.daily.filter(
            (pl.col("ticker") == code) & pl.col("date").is_between(start, end)
        )

    def minute_codes(self, code, history):
        return (code,)

    def minute_bars(self, code, first, last):
        self.asked.append((code, first, last))
        days = self.daily_bars(code, first, last).filter(
            ~pl.col("date").is_in(self.lacks)
        )
        closes = dict(
            zip(days["date"], days["close"] / days["adjustment"], strict=True)
        )
        ends = [
            (s.close - timedelta(minutes=1), closes[s.day])
            for s in sessions(first, last)
            if s.day in closes
        ]
        n = len(ends)
        prices = [price for _, price in ends]
        return pl.DataFrame(
            {
                "start": [start for start, _ in ends],
                "ticker": [code] * n,
                "open": prices,
                "high": prices,
                "low": prices,
                "close": prices,
                "volume": [100] * n,
            },
            schema=MINUTE_SCHEMA,
        )


def frozen_at(moment):
    """The CLI's datetime, with now() fixed at `moment`."""

    class Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return moment

    return Frozen


def with_keys(monkeypatch):
    monkeypatch.setenv("EODHD_API_KEY", "eodhd-key")
    monkeypatch.setenv("ALPACA_API_KEY_ID", "key-id")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "secret")


@pytest.mark.parametrize(
    ("now", "last_day"),
    [
        # Monday 12 October's session ended at 20:00 UTC, and a request runs
        # to the end of its last UTC day. Alpaca serves SIP minutes once they
        # are 15 minutes old, so that day can be asked for from 00:15 UTC.
        (datetime(2026, 10, 13, 0, 10, tzinfo=UTC), date(2026, 10, 11)),
        (datetime(2026, 10, 13, 0, 20, tzinfo=UTC), date(2026, 10, 12)),
    ],
)
def test_etf_minutes_ask_alpaca_first_and_only_for_days_it_may_serve(
    now, last_day, tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    with_keys(monkeypatch)
    closes = session_closes("QQQ", date(2026, 1, 1), date(2026, 12, 31))
    alpaca = FakeProvider("Alpaca", closes)
    eodhd = FakeProvider("EODHD", closes)
    monkeypatch.setattr(cli, "AlpacaClient", lambda key_id, secret: alpaca)
    monkeypatch.setattr(cli, "EodhdClient", lambda key: eodhd)
    monkeypatch.setattr(cli, "datetime", frozen_at(now))

    assert main(["etf-minutes", "QQQ", "--from-year", "2026"]) == 0

    # One minute a session is not every minute, so EODHD is asked too.
    assert alpaca.asked == eodhd.asked == [("QQQ", date(2026, 1, 1), last_day)]
    stored = pl.read_parquet(tmp_path / "data/raw/intraday/QQQ/2026.parquet")
    assert stored["start"].dt.date().to_list() == session_days(
        date(2026, 1, 1), last_day
    )
    assert set(stored["source"]) == {"Alpaca"}  # the first, on every tied day
    assert "100% of them match the daily close" in capsys.readouterr().out


GUARD = Path(__file__).resolve().parents[1] / "guard.yaml"


@pytest.fixture
def synthetic_bars(tmp_path, monkeypatch, capsys):
    """Synthetic daily bars stored in a fresh working directory."""
    monkeypatch.chdir(tmp_path)
    assert main(["ingest", "--synthetic"]) == 0
    capsys.readouterr()  # only what the test's own commands print is checked
    return tmp_path


def test_intraday_takes_eodhd_first_and_fills_its_holes_from_alpaca(
    synthetic_bars, monkeypatch, capsys
):
    raw = synthetic_bars / "data" / "raw"
    shutil.copy(raw / "prices_synthetic.parquet", raw / "prices.parquet")
    daily = pl.read_parquet(raw / "prices.parquet")
    hole = date(2024, 3, 4)
    eodhd = FakeProvider("EODHD", daily, lacks=[hole])
    alpaca = FakeProvider("Alpaca", daily)
    with_keys(monkeypatch)
    monkeypatch.setattr(cli, "EodhdClient", lambda key: eodhd)
    monkeypatch.setattr(cli, "AlpacaClient", lambda key_id, secret: alpaca)
    # Ten minutes into 2025, UTC: Alpaca cannot serve 31 December yet.
    monkeypatch.setattr(
        cli, "datetime", frozen_at(datetime(2025, 1, 1, 0, 10, tzinfo=UTC))
    )

    assert main(["intraday", "--from-year", "2024", "--to-year", "2024"]) == 0

    assert {last for _, _, last in eodhd.asked + alpaca.asked} == {date(2024, 12, 30)}
    stored = pl.read_parquet(raw / "intraday" / "AAPL" / "2024.parquet")
    on_hole = pl.col("start").dt.date() == hole
    assert stored.filter(on_hole)["source"].to_list() == ["Alpaca"]
    assert set(stored.filter(~on_hole)["source"]) == {"EODHD"}
    assert stored["start"].dt.date().to_list() == session_days(
        date(2024, 1, 1), date(2024, 12, 30)
    )
    assert "days from EODHD" in capsys.readouterr().out


def run_plan(*options):
    return main(["plan", "--synthetic", "--guard", str(GUARD), *options])


def test_plan_writes_the_next_sessions_plan_from_synthetic_bars(synthetic_bars, capsys):
    assert run_plan("--capital", "30") == 0

    # The synthetic bars end on 31 December 2024; the next session is 2 January.
    written = synthetic_bars / "data" / "plans" / "daily_plan_2025-01-02.json"
    plan = plan_from_json(written.read_text(encoding="utf-8"))
    assert plan.trading_date.isoformat() == "2025-01-02"
    assert len(plan.target_weights) == 10  # 30 USD holds ten positions of 3 USD
    assert "plan for 2025-01-02" in capsys.readouterr().out


def test_plan_refuses_a_day_the_exchange_is_closed(synthetic_bars, capsys):
    assert run_plan("--date", "2024-12-25") == 1
    assert "2024-12-25 is not a trading day" in capsys.readouterr().err


def test_plan_refuses_to_plan_from_old_bars(synthetic_bars, capsys):
    # The bars end on 31 December; a plan for 6 January needs those of 3 January.
    assert run_plan("--date", "2025-01-06") == 1
    assert "no bars for 2025-01-03" in capsys.readouterr().err
    assert not (synthetic_bars / "data" / "plans").exists()


def test_plan_for_a_day_in_the_past_uses_only_the_bars_before_it(
    synthetic_bars, capsys
):
    out = synthetic_bars / "june.json"
    assert run_plan("--date", "2024-06-03", "--out", str(out)) == 0
    assert plan_from_json(out.read_text(encoding="utf-8")).trading_date.isoformat() == (
        "2024-06-03"
    )
