"""
Feature preprocessing.

Anything that is learned from data (means, standard deviations, PCA directions) is
computed in fit() on the TRAINING rows only, and then applied with transform() to
validation / test rows. Otherwise information from the validation set would leak in.

Modes (combine with "+", e.g. "hellinger+standardize"):
  "none"         use the features as they are
  "hellinger"    (images) divide each histogram by its total, then take the square root.
                 Stops a few very frequent visual words from dominating distances.
  "l2"           scale each row to length 1 (only the direction is kept)
  "standardize"  per feature: (x - mean) / std

pca_dims: if given, project onto the top principal components afterwards (via SVD).
"""
import numpy as np


class Preprocessor:
    def __init__(self, mode="none", pca_dims=None):
        self.steps = [] if mode == "none" else mode.split("+")
        for step in self.steps:
            assert step in ("hellinger", "l2", "standardize"), f"unknown step {step}"
        self.pca_dims = pca_dims

    def _hellinger_l2(self, X):
        X = X.astype(np.float32)
        if "hellinger" in self.steps:
            total = X.sum(axis=1, keepdims=True)
            total[total == 0] = 1.0
            X = np.sqrt(X / total)
        if "l2" in self.steps:
            X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        return X

    def fit(self, X):
        X = self._hellinger_l2(X)
        if "standardize" in self.steps:
            self.mean = X.mean(axis=0)
            self.std = X.std(axis=0) + 1e-8
            X = (X - self.mean) / self.std
        if self.pca_dims:
            self.pca_mean = X.mean(axis=0)
            _, S, Vt = np.linalg.svd(X - self.pca_mean, full_matrices=False)
            self.components = Vt[: self.pca_dims]
            self.explained_variance_ratio = (S ** 2 / (S ** 2).sum())[: self.pca_dims]
        return self

    def transform(self, X):
        X = self._hellinger_l2(X)
        if "standardize" in self.steps:
            X = (X - self.mean) / self.std
        if self.pca_dims:
            X = (X - self.pca_mean) @ self.components.T
        return X.astype(np.float32)

    def fit_transform(self, X):
        return self.fit(X).transform(X)
