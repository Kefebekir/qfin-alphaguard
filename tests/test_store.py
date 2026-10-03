from pathlib import Path

from qfin_alphaguard.config import Config
from qfin_alphaguard.data.store import (
    prices_path,
    read_prices,
    upload_to_s3,
    write_prices,
)
from qfin_alphaguard.data.synthetic import generate_prices


def test_roundtrip_preserves_data(tmp_path):
    df = generate_prices(Config())
    path = write_prices(df, tmp_path / "nested" / "prices.parquet")

    assert path.exists()
    assert read_prices(path).equals(df)


def test_synthetic_and_real_paths_differ():
    assert prices_path(Config(synthetic=True)) != prices_path(Config())


class FakeS3Client:
    """Stands in for boto3's S3 client and records what it was asked to do."""

    def __init__(self):
        self.uploads = []

    def upload_file(self, filename, bucket, key):
        self.uploads.append((filename, bucket, key))


def test_upload_to_s3_uses_relative_path_as_key(tmp_path, monkeypatch):
    fake = FakeS3Client()
    monkeypatch.setattr("qfin_alphaguard.data.store.boto3.client", lambda service: fake)
    monkeypatch.chdir(tmp_path)

    path = write_prices(generate_prices(Config()), Path("data/raw/prices.parquet"))
    uri = upload_to_s3(path, "test-bucket")

    assert uri == "s3://test-bucket/data/raw/prices.parquet"
    assert fake.uploads == [(str(path), "test-bucket", "data/raw/prices.parquet")]
