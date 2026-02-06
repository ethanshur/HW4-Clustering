import numpy as np
import pytest
from cluster.utils import make_clusters
from cluster.kmeans import KMeans

def test_kmeans_all():
    X, _ = make_clusters(n=300, m=4, k=3, scale=0.6, seed=0)

    km = KMeans(k=3, tol=1e-6, max_iter=100)
    km.fit(X)

    labels = km.predict(X)
    assert labels.shape == (X.shape[0],)
    assert np.issubdtype(labels.dtype, np.integer)
    assert labels.min() >= 0 and labels.max() < 3

    centroids = km.get_centroids()
    assert centroids.shape == (3, X.shape[1])

    with pytest.raises(ValueError):
        KMeans(k=0)
    with pytest.raises(ValueError):
        KMeans(k=-1)
    with pytest.raises(ValueError):
        KMeans(k=2.5)

    X_small, _ = make_clusters(n=3, m=2, k=3, seed=1)
    km_bad = KMeans(k=4)
    with pytest.raises(ValueError):
        km_bad.fit(X_small)

    X_bad = np.random.randn(10, 2)
    with pytest.raises(ValueError):
        km.predict(X_bad)
