"""Efe's news reader model (pre-registrations N1 and N2).

Principal components of the headlines' embedding, the count as it is, then a ridge
regression. Efe's two choices: how many components to keep and how hard the ridge shrinks.
"""

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make(components=16, penalty=10.0):
    """A function that builds a fresh, untrained model for each test year."""
    embedding = list(range(768))  # the first 768 columns
    count = [768]  # the last column, log(1 + count)
    return lambda: Pipeline(
        [
            (
                "features",
                ColumnTransformer(
                    [
                        (
                            "pca",
                            PCA(n_components=components, random_state=0),
                            embedding,
                        ),
                        ("count", "passthrough", count),
                    ]
                ),
            ),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=penalty)),
        ]
    )
