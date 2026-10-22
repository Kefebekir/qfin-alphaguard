import json
import shutil
import sys
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


def test_ingest_repairs_a_broken_bar_writes_and_says_so(tmp_path, monkeypatch, capsys):
    # One bar with a wrong high and low, as EODHD sent LUMN's of 12 October
    # 2026, no longer stops the nightly job (decision 0016).
    monkeypatch.chdir(tmp_path)
    load = cli.load_prices

    def with_a_broken_bar(config):
        prices = load(config)
        first = (pl.col("ticker") == prices["ticker"][0]) & (
            pl.col("date") == prices["date"][0]
        )
        return prices.with_columns(
            pl.when(first)
            .then(pl.col("low") * 0.5)
            .otherwise(pl.col("high"))
            .alias("high")
        )

    monkeypatch.setattr(cli, "load_prices", with_a_broken_bar)

    assert main(["ingest", "--synthetic"]) == 0
    assert "[warning] repaired_bars: 1 bars" in capsys.readouterr().out
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


def test_aws_tasks_get_the_keys_they_may_read_and_no_other():
    # qfin ingest needs EODHD_API_KEY, qfin live Alpaca's two keys; ECS reads them from
    # SSM with the execution role, which may read those parameters and nothing else.
    policy = json.loads(
        (AWS / "execution-role-secrets-policy.json").read_text(encoding="utf-8")
    )
    (statement,) = policy["Statement"]
    assert statement["Action"] == "ssm:GetParameters"
    read = set()
    for name, keys in (
        ("task-definition.template.json", {"EODHD_API_KEY"}),
        (
            "live-task-definition.template.json",
            {"ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY"},
        ),
    ):
        task = json.loads((AWS / name).read_text(encoding="utf-8"))
        secrets = task["containerDefinitions"][0]["secrets"]
        assert {s["name"] for s in secrets} == keys
        read |= {s["valueFrom"].replace("ACCOUNT_ID", "*") for s in secrets}
    assert read == set(statement["Resource"])


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


def test_live_takes_a_day_and_the_qqq_account_s_limits():
    args = build_parser().parse_args(["live", "--day", "2026-10-19"])
    assert (args.day, args.guard, args.root) == (
        date(2026, 10, 19),
        "guard_qqq.yaml",
        "data/live",
    )


def test_live_on_a_day_without_a_session_does_nothing(capsys, tmp_path):
    assert main(["live", "--day", "2026-10-18", "--root", str(tmp_path)]) == 0
    assert capsys.readouterr().out == "no session on 2026-10-18\n"
    assert list(tmp_path.iterdir()) == []


def test_live_on_the_server_backs_up_and_alerts():
    args = build_parser().parse_args(
        [
            "live",
            "--s3-bucket",
            "b",
            "--alert-topic",
            "arn:aws:sns:eu-north-1:1:qfin-alerts",
            "--check",
        ]
    )
    assert (args.s3_bucket, args.alert_topic, args.check) == (
        "b",
        "arn:aws:sns:eu-north-1:1:qfin-alerts",
        True,
    )
    plain = build_parser().parse_args(["live"])
    assert (plain.s3_bucket, plain.alert_topic, plain.check) == (None, None, False)


def test_the_scheduler_may_run_every_task_and_the_report_s_command_parses():
    policy = json.loads(
        (AWS / "scheduler-role-policy.json").read_text(encoding="utf-8")
    )
    (run,) = [s for s in policy["Statement"] if s["Action"] == "ecs:RunTask"]
    families = {
        json.loads(path.read_text(encoding="utf-8"))["family"]
        for path in AWS.glob("*task-definition.template.json")
    }
    assert families == {
        "qfin-alphaguard",
        "qfin-live",
        "qfin-live-report",
        "qfin-news-capture",
    }
    assert set(run["Resource"]) == {
        f"arn:aws:ecs:eu-north-1:*:task-definition/{family}:*" for family in families
    }
    task = json.loads((AWS / "live-report-task-definition.template.json").read_text())
    args = build_parser().parse_args(task["containerDefinitions"][0]["command"])
    assert args.command == "live-report" and args.s3_bucket and args.alert_topic


def test_the_live_task_s_command_is_one_this_cli_accepts():
    task = json.loads(Path("infra/aws/live-task-definition.template.json").read_text())
    (container,) = task["containerDefinitions"]
    args = build_parser().parse_args(container["command"])
    assert args.command == "live" and args.s3_bucket and args.alert_topic
    assert {s["name"] for s in container["secrets"]} == {
        "ALPACA_API_KEY_ID",
        "ALPACA_API_SECRET_KEY",
    }


def test_live_report_takes_the_server_s_bucket_and_topic():
    args = build_parser().parse_args(
        [
            "live-report",
            "--day",
            "2026-10-19",
            "--s3-bucket",
            "b",
            "--alert-topic",
            "arn",
        ]
    )
    assert (args.day, args.s3_bucket, args.alert_topic, args.root) == (
        date(2026, 10, 19),
        "b",
        "arn",
        "data/live",
    )


def test_live_report_without_a_journal_says_so(capsys, tmp_path):
    assert main(["live-report", "--day", "2026-10-18", "--root", str(tmp_path)]) == 0
    assert main(["live-report", "--day", "2026-10-19", "--root", str(tmp_path)]) == 1
    assert "no journal for 2026-10-19" in capsys.readouterr().out


def test_the_news_capture_task_s_command_parses_and_reads_alpaca_s_keys():
    task = json.loads((AWS / "news-capture-task-definition.template.json").read_text())
    (container,) = task["containerDefinitions"]
    args = build_parser().parse_args(container["command"])
    assert args.command == "news-capture" and args.s3_bucket and args.alert_topic
    assert args.root == "data/news_capture"
    assert {s["name"] for s in container["secrets"]} == {
        "ALPACA_API_KEY_ID",
        "ALPACA_API_SECRET_KEY",
    }


def test_a_news_capture_that_fails_is_told_and_a_holiday_is_not_a_failure(
    tmp_path, monkeypatch, capsys
):
    import qfin_alphaguard.data.news_capture as nc

    monkeypatch.setenv("ALPACA_API_KEY_ID", "k")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "s")
    sent = []

    class FakeSns:
        def publish(self, **message):
            sent.append(message)

    fake_boto3 = type(sys)("boto3")
    fake_boto3.client = lambda service: FakeSns()
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    root = str(tmp_path / "news")
    # Thanksgiving: no session, nothing to capture, not a failure.
    assert main(["news-capture", "--day", "2026-11-26", "--root", root]) == 0
    # A session outside 15:30-16:30 New York time: refused, and told.
    monkeypatch.setattr(
        nc,
        "capture",
        lambda *a, **k: (_ for _ in ()).throw(nc.CaptureError("too early")),
    )
    code = main(
        ["news-capture", "--day", "2026-11-27", "--root", root, "--alert-topic", "arn"]
    )
    assert code == 1
    assert sent and "too early" in sent[0]["Message"] and sent[0]["TopicArn"] == "arn"
    assert "failed: too early" in capsys.readouterr().err


def test_a_news_capture_is_kept_in_s3_and_asks_s3_whether_it_exists(
    tmp_path, monkeypatch, capsys
):
    import qfin_alphaguard.data.news_capture as nc
    from qfin_alphaguard import cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ALPACA_API_KEY_ID", "k")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "s")
    asked = []

    class FakeS3:
        def list_objects_v2(self, **query):
            asked.append(query)
            return {"KeyCount": 0}

    fake_boto3 = type(sys)("boto3")
    fake_boto3.client = lambda service: FakeS3()
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)

    def capture(day, root, get, exists_elsewhere):
        path = root / f"{day}.jsonl"
        if exists_elsewhere(path):
            raise nc.CaptureError("held in S3 already")
        root.mkdir(parents=True, exist_ok=True)
        path.write_text('{"id": 1}\n{"id": 2}\n')
        return path

    monkeypatch.setattr(nc, "capture", capture)
    uploaded = []
    monkeypatch.setattr(
        cli,
        "upload_to_s3",
        lambda path, bucket: uploaded.append(path) or f"s3://{bucket}/x",
    )
    assert main(["news-capture", "--day", "2026-11-27", "--s3-bucket", "bkt"]) == 0
    assert asked == [{"Bucket": "bkt", "Prefix": "data/news_capture/2026-11-27.jsonl"}]
    assert uploaded == [Path("data/news_capture/2026-11-27.jsonl")]
    assert "2 articles captured, s3://bkt/x" in capsys.readouterr().out
    # Once S3 holds the day, the capture is refused.
    monkeypatch.setattr(FakeS3, "list_objects_v2", lambda self, **q: {"KeyCount": 1})
    assert main(["news-capture", "--day", "2026-11-27", "--s3-bucket", "bkt"]) == 1
    assert "held in S3 already" in capsys.readouterr().err
