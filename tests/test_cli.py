import json
from pathlib import Path

import pytest

from qfin_alphaguard.cli import build_parser, main

TASK_DEFINITION = (
    Path(__file__).resolve().parents[1]
    / "infra"
    / "aws"
    / "task-definition.template.json"
)


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
