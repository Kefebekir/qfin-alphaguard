import json
from pathlib import Path

import pytest

from qfin_alphaguard.cli import build_parser, main
from qfin_alphaguard.plan import plan_from_json

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


@pytest.mark.parametrize("command", ["backtest"])
def test_commands_not_built_yet_fail_and_say_so(command, capsys):
    assert main([command]) == 1
    assert "not built yet" in capsys.readouterr().err


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


GUARD = Path(__file__).resolve().parents[1] / "guard.yaml"


@pytest.fixture
def synthetic_bars(tmp_path, monkeypatch, capsys):
    """Synthetic daily bars stored in a fresh working directory."""
    monkeypatch.chdir(tmp_path)
    assert main(["ingest", "--synthetic"]) == 0
    capsys.readouterr()  # only what the test's own commands print is checked
    return tmp_path


def run_plan(*options):
    return main(["plan", "--synthetic", "--guard", str(GUARD), *options])


def test_plan_writes_the_next_sessions_plan_from_synthetic_bars(synthetic_bars, capsys):
    assert run_plan("--capital", "30000") == 0

    # The synthetic bars end on 31 December 2024; the next session is 2 January.
    written = synthetic_bars / "data" / "plans" / "daily_plan_2025-01-02.json"
    plan = plan_from_json(written.read_text(encoding="utf-8"))
    assert plan.trading_date.isoformat() == "2025-01-02"
    assert len(plan.target_weights) == 10  # 30,000 USD holds ten 3,000 USD positions
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
