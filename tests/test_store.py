from qfin_alphaguard.config import Config
from qfin_alphaguard.data.store import prices_path, read_prices, write_prices
from qfin_alphaguard.data.synthetic import generate_prices


def test_roundtrip_preserves_data(tmp_path):
    df = generate_prices(Config())
    path = write_prices(df, tmp_path / "nested" / "prices.parquet")

    assert path.exists()
    assert read_prices(path).equals(df)


def test_synthetic_and_real_paths_differ():
    assert prices_path(Config(synthetic=True)) != prices_path(Config())
