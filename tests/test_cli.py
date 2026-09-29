from qfin_alphaguard.cli import main


def test_synthetic_run_writes_parquet(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    exit_code = main(["--synthetic"])

    assert exit_code == 0
    assert (tmp_path / "data" / "raw" / "prices_synthetic.parquet").exists()
