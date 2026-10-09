import json
from pathlib import Path

import pytest

from qfin_alphaguard.cli import build_parser, main

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


@pytest.mark.parametrize("command", ["plan", "backtest"])
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
