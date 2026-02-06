import numpy as np
from scipy.spatial.distance import cdist

class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """
        self.is_fit_ = False

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations
        """
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            raise ValueError("X must be a 2d numpy array")
        if not isinstance(y, np.ndarray) or y.ndim != 1:
            raise ValueError("y must be a 1d numpy array")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same num rows")

        clusters = np.unique(y)
        n = X.shape[0]
        scores = np.zeros(n, dtype=float)

        for i in range(n):
            xi = X[i:i + 1]
            ci = y[i]

            # mean distance to other points in same cluster
            same_idx = np.where(y == ci)[0]
            if same_idx.size <= 1:
                scores[i] = 0.0
                continue

            same_idx_wo_i = same_idx[same_idx != i]
            a_i = float(np.mean(cdist(xi, X[same_idx_wo_i])))

            # minimum mean distance to points in any other cluster
            b_i = float("inf")
            for c in clusters:
                if c == ci:
                    continue
                other_idx = np.where(y == c)[0]
                if other_idx.size == 0:
                    continue
                avg_dist = float(np.mean(cdist(xi, X[other_idx])))
                if avg_dist < b_i:
                    b_i = avg_dist

            # silhouette for point i
            denom = max(a_i, b_i)
            scores[i] = 0.0 if denom == 0.0 else (b_i - a_i) / denom

        return scores
