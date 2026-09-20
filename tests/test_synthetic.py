from qfin_alphaguard.config import Config
from qfin_alphaguard.data.synthetic import generate_prices


def test_columns():
    df = generate_prices(Config())
    assert df.columns == ["date", "ticker", "close"]


def test_shape():
    df = generate_prices(Config())
    assert len(df) == df["date"].n_unique() * df["ticker"].n_unique()


def test_positive_prices():
    df = generate_prices(Config())
    assert (df["close"] > 0).all()
