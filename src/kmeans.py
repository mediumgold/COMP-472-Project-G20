"""
k-Means clustering from scratch (unsupervised: the labels are never used).

Lloyd's algorithm:
  1. pick k starting centres with k-means++ (spread out: each new centre is chosen with
     probability proportional to its squared distance from the centres picked so far)
  2. assign every point to its nearest centre
  3. move each centre to the mean of its points
  4. repeat 2-3 until the assignments stop changing
The result depends on the starting centres, so we run it n_init times and keep the
run with the lowest inertia (sum of squared distances to the assigned centre).
"""
import numpy as np


def squared_distances(X, C, X_sq=None):
    if X_sq is None:
        X_sq = (X ** 2).sum(1)[:, None]
    return np.maximum(X_sq - 2 * X @ C.T + (C ** 2).sum(1)[None, :], 0.0)


class KMeans:
    def __init__(self, n_clusters=8, n_init=5, max_iter=300, tol=1e-6, random_state=None):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def kmeans_plus_plus(self, X, X_sq, rng):
        n = len(X)
        centres = [X[rng.integers(n)]]
        d2 = squared_distances(X, np.array(centres), X_sq)[:, 0]
        for _ in range(1, self.n_clusters):
            centres.append(X[rng.choice(n, p=d2 / d2.sum())])
            d2 = np.minimum(d2, squared_distances(X, centres[-1][None, :], X_sq)[:, 0])
        return np.array(centres)

    def run_once(self, X, X_sq, rng):
        C = self.kmeans_plus_plus(X, X_sq, rng)
        labels = None
        for _ in range(self.max_iter):
            D = squared_distances(X, C, X_sq)
            new_labels = D.argmin(1)
            if labels is not None and np.array_equal(new_labels, labels):
                break
            labels = new_labels
            # new centre = mean of the points assigned to it
            C_new = np.zeros_like(C)
            np.add.at(C_new, labels, X)
            counts = np.bincount(labels, minlength=self.n_clusters)
            empty = counts == 0
            C_new[~empty] /= counts[~empty, None]
            if empty.any():   # restart an empty cluster at the point farthest from its centre
                far = D[np.arange(len(X)), labels].argsort()[::-1][:empty.sum()]
                C_new[empty] = X[far]
            shift = ((C_new - C) ** 2).sum()
            C = C_new
            if shift < self.tol:
                break
        D = squared_distances(X, C, X_sq)
        labels = D.argmin(1)
        inertia = float(D[np.arange(len(X)), labels].sum())
        return C, labels, inertia

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        X_sq = (X ** 2).sum(1)[:, None]
        rng = np.random.default_rng(self.random_state)
        best = None
        for _ in range(self.n_init):
            run = self.run_once(X, X_sq, rng)
            if best is None or run[2] < best[2]:
                best = run
        self.centres, self.labels, self.inertia = best
        return self

    def predict(self, X):
        return squared_distances(np.asarray(X, dtype=np.float64), self.centres).argmin(1)
