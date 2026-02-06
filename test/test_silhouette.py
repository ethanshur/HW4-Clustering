import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, silhouette_samples
from cluster.silhouette import Silhouette


def test_silhouette_matches_sklearn():
    X, y = make_blobs(n_samples=200, centers=4, n_features=5, random_state=0)

    sil = Silhouette()
    ours = sil.score(X, y)

    assert ours.shape == (X.shape[0],)

    sk_samples = silhouette_samples(X, y, metric="euclidean")
    assert np.allclose(ours, sk_samples, atol=1e-2)

    sk_mean = silhouette_score(X, y, metric="euclidean")
    assert np.isclose(float(np.mean(ours)), float(sk_mean), atol=1e-2)
