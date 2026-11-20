"""
k-Nearest Neighbours, written from scratch with NumPy.

There is no real training: fit() just stores the training set. To classify a point
we find the k closest training points and let them vote.

Parameters
  k        number of neighbours that vote
  metric   "euclidean"     straight-line distance ||a - b||
           "cosine"        1 - cos(angle between a and b), ignores vector length
           "intersection"  1 - sum_i min(a_i, b_i)              (histograms only)
           "chi2"          0.5 * sum_i (a_i - b_i)^2 / (a_i + b_i) (histograms only)
           For the two histogram distances every row is first rescaled to sum to 1.
  weights  "uniform"   every neighbour gets 1 vote
           "distance"  closer neighbours get bigger votes: weight = 1 / distance^power
  power    how strongly closer neighbours count (only for weights="distance")
"""
import numpy as np


class KNNClassifier:
    def __init__(self, k=5, metric="euclidean", weights="uniform", power=1.0, n_classes=None):
        assert metric in ("euclidean", "cosine", "intersection", "chi2")
        self.k = k
        self.metric = metric
        self.weights = weights
        self.power = power
        self.n_classes = n_classes

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=np.float32)
        self.y_train = np.asarray(y, dtype=np.int64)
        if self.n_classes is None:
            self.n_classes = int(self.y_train.max()) + 1
        # Precompute what we can about the training rows once, instead of for every query
        if self.metric == "euclidean":
            self.train_sq_norms = (self.X_train ** 2).sum(axis=1)
        elif self.metric == "cosine":
            self.train_unit = self.X_train / (np.linalg.norm(self.X_train, axis=1, keepdims=True) + 1e-12)
        else:
            self.train_hist = to_histogram(self.X_train)
        return self

    # ---------------------------------------------------------------- distances
    def distances(self, Q):
        """Matrix of distances from every query row in Q to every training row."""
        if self.metric == "euclidean":
            # |a - b|^2 = |a|^2 - 2 a.b + |b|^2, so one matrix multiply does all pairs at once
            d2 = (Q ** 2).sum(axis=1, keepdims=True) - 2.0 * Q @ self.X_train.T + self.train_sq_norms
            return np.sqrt(np.maximum(d2, 0.0))
        if self.metric == "cosine":
            Q_unit = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
            return 1.0 - Q_unit @ self.train_unit.T
        return self.histogram_distances(Q)

    def histogram_distances(self, Q):
        """
        Intersection and chi2 for histograms whose rows sum to 1:
            intersection = 1 - sum_i min(a_i, b_i)
            chi2         = 1 - 2 * sum_i a_i b_i / (a_i + b_i)   (same as the formula above)
        A feature only contributes when it is non-zero in BOTH histograms. The image
        features are ~85% zeros, so we loop over the 500 features and only touch the
        (query, train) pairs where both are non-zero. Much faster than the direct formula.
        """
        Q = to_histogram(Q)
        B = self.train_hist
        S = np.zeros((len(Q), len(B)), dtype=np.float32)
        for i in range(B.shape[1]):
            q_rows = np.flatnonzero(Q[:, i])
            b_rows = np.flatnonzero(B[:, i])
            if len(q_rows) == 0 or len(b_rows) == 0:
                continue
            a = Q[q_rows, i][:, None]
            b = B[b_rows, i][None, :]
            if self.metric == "intersection":
                S[np.ix_(q_rows, b_rows)] += np.minimum(a, b)
            else:
                S[np.ix_(q_rows, b_rows)] += a * b / (a + b)
        D = 1.0 - S if self.metric == "intersection" else 1.0 - 2.0 * S
        return np.maximum(D, 0.0)

    # ------------------------------------------------------------- neighbours
    def kneighbors(self, X, k=None, chunk_size=1000):
        """(distances, indices) of the k nearest training rows, closest first."""
        k = k or self.k
        X = np.asarray(X, dtype=np.float32)
        all_d, all_i = [], []
        for start in range(0, len(X), chunk_size):          # chunks keep memory use small
            D = self.distances(X[start:start + chunk_size])
            idx = np.argpartition(D, k - 1, axis=1)[:, :k]   # the k smallest, not yet sorted
            d = np.take_along_axis(D, idx, axis=1)
            order = np.argsort(d, axis=1)
            all_i.append(np.take_along_axis(idx, order, axis=1))
            all_d.append(np.take_along_axis(d, order, axis=1))
        return np.vstack(all_d), np.vstack(all_i)

    def vote(self, dist, idx, k=None, weights=None, power=None):
        """
        Turn neighbour lists into class probabilities (rows sum to 1).
        Kept separate from kneighbors() so that in the notebooks we can compute the
        neighbours once and then try many values of k / weights / power cheaply.
        """
        k = k or self.k
        weights = weights or self.weights
        power = self.power if power is None else power
        dist, idx = dist[:, :k].astype(np.float64), idx[:, :k]
        labels = self.y_train[idx]
        if weights == "uniform":
            w = np.ones_like(dist)
        else:
            # (closest distance / distance)^power ranks neighbours the same way as
            # 1/distance^power, but stays between 0 and 1, so big powers don't overflow
            w = ((dist[:, :1] + 1e-8) / (dist + 1e-8)) ** power
        # tiny bonus for closer neighbours, so a tied vote goes to the nearest neighbour's class
        w = w + 1e-6 * (k - np.arange(k))[None, :]
        scores = np.zeros((len(labels), self.n_classes))
        np.add.at(scores, (np.arange(len(labels))[:, None], labels), w)
        return scores / scores.sum(axis=1, keepdims=True)

    def predict_proba(self, X):
        dist, idx = self.kneighbors(X)
        return self.vote(dist, idx)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


def to_histogram(X):
    """Rescale each (non-negative) row to sum to 1."""
    assert (X >= 0).all(), "histogram distances need non-negative features (don't standardize)"
    return X / (X.sum(axis=1, keepdims=True) + 1e-12)
