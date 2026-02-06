import numpy as np
from scipy.spatial.distance import cdist

class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """
        # hyperparams
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        if not isinstance(tol, (int, float)) or tol <= 0:
            raise ValueError("tol must be a positive number")
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError("max_iter must be a positive integer")

        self.k = k
        self.tol = float(tol)
        self.max_iter = max_iter

        # set during fit
        self.X_ = None
        self.centroids_ = None    # shape: (k, n_features)
        self.n_features_ = None   # used to validate predict input
        self.is_fit_ = False      # for errors

    def fit(self, mat: np.ndarray):
        if not isinstance(mat, np.ndarray) or mat.ndim != 2:
            raise ValueError("mat must be a 2d numpy array.")

        # reset state for if recalled
        self.is_fit_ = False

        n_samples, n_features = mat.shape
        if n_samples < self.k:
            raise ValueError("k cannot be greater than the number of samples.")

        self.n_features_ = n_features

        init_idx = np.random.choice(n_samples, size=self.k, replace=False)
        self.centroids_ = mat[init_idx].copy()

        self.X_ = mat
        iters = 0
        prev_err = float("inf")

        while iters < self.max_iter:
            distances = cdist(mat, self.centroids_)
            labels = np.argmin(distances, axis=1)
            assigned = distances[np.arange(n_samples), labels]
            curr_err = float(np.mean(assigned ** 2))

            # update step
            centroids = np.zeros((self.k, self.n_features_))
            for j in range(self.k):
                points = mat[labels == j]
                if points.shape[0] == 0:
                    centroids[j] = mat[np.random.randint(n_samples)]
                else:
                    centroids[j] = np.mean(points, axis=0)
            self.centroids_ = centroids

            # check if delta between errors is past the tolerance threshold
            if abs(prev_err - curr_err) <= self.tol:
                break
            prev_err = curr_err

            iters += 1

        self.is_fit_ = True

    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        if not self.is_fit_ or self.centroids_ is None or self.n_features_ is None:
            raise ValueError("Model has not been fit yet")
        if not isinstance(mat, np.ndarray) or mat.ndim != 2:
            raise ValueError("mat must be a 2D numpy array")
        if mat.shape[1] != self.n_features_:
            raise ValueError()

        distances = cdist(mat, self.centroids_)
        return np.argmin(distances, axis=1)


    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """
        if not self.is_fit_ or self.centroids_ is None:
            raise ValueError("Model has not been fit yet")
        if self.X_ is None:
            raise ValueError("Training data was not stored")

        distances = cdist(self.X_, self.centroids_)
        labels = np.argmin(distances, axis=1)
        assigned = distances[np.arange(self.X_.shape[0]), labels]
        return float(np.mean(assigned ** 2))

    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        if not self.is_fit_ or self.centroids_ is None:
            raise ValueError("model has not been fit yet")
        return self.centroids_
