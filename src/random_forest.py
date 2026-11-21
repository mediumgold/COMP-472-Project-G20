"""
Random Forest, written from scratch on top of our DecisionTreeClassifier.

A single deep tree has low bias but high variance (a slightly different training
set gives a very different tree). A forest trains many trees that are made
different on purpose and averages their predicted probabilities:
  1. bootstrap: each tree trains on n rows drawn WITH replacement (~63% unique rows)
  2. at every node only max_features random features are tried (default sqrt(d))

Out-of-bag (OOB) score: every training row is predicted only by the trees that did
NOT see it in their bootstrap sample, which gives a free estimate of accuracy.

Feature importance: the forest exposes `feature_importances`, the mean of the per-tree
Gini importance vectors (each tree's vector sums to 1, so their mean does too).

n_jobs > 1 grows trees in parallel processes. Each tree's randomness comes from its
own seed, so the forest is identical whether we use 1 core or 12.
"""
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .decision_tree import Binner, DecisionTreeClassifier

# Data shared with the worker processes (set once per process by _init_worker)
_shared = {}


def _init_worker(Xb, y, binner, tree_params, n_classes, bootstrap):
    _shared.update(Xb=Xb, y=y, binner=binner, tree_params=tree_params,
                   n_classes=n_classes, bootstrap=bootstrap)


def _grow_tree(seeds):
    """Grow one tree. seeds[0] picks the bootstrap rows, seeds[1] the random features."""
    s = _shared
    n = len(s["y"])
    rng = np.random.default_rng(int(seeds[0]))
    rows = rng.integers(0, n, size=n) if s["bootstrap"] else np.arange(n)
    tree = DecisionTreeClassifier(**s["tree_params"], random_state=int(seeds[1]))
    tree.fit_binned(s["Xb"], s["y"], s["binner"], rows, s["n_classes"])
    return tree, rows


class RandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 max_features="sqrt", criterion="gini", bootstrap=True, n_bins=32,
                 random_state=None, n_jobs=1):
        self.n_estimators = n_estimators
        self.tree_params = dict(max_depth=max_depth, min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf, criterion=criterion,
                                max_features=max_features, n_bins=n_bins)
        self.bootstrap = bootstrap
        self.n_bins = n_bins
        self.random_state = random_state
        self.n_jobs = n_jobs           # -1 = all CPU cores

    def fit(self, X, y, n_classes=None):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        n = len(y)
        self.n_classes = int(n_classes or (y.max() + 1))

        # bin the data once; all trees share it and only differ in which rows they get
        self.binner = Binner(self.n_bins).fit(X)
        Xb = self.binner.transform(X)
        seeds = np.random.default_rng(self.random_state).integers(2 ** 31, size=(self.n_estimators, 2))

        n_jobs = os.cpu_count() if self.n_jobs == -1 else self.n_jobs
        worker_args = (Xb, y, self.binner, self.tree_params, self.n_classes, self.bootstrap)
        if n_jobs == 1:
            _init_worker(*worker_args)
            results = list(map(_grow_tree, seeds))
        else:
            with ProcessPoolExecutor(n_jobs, initializer=_init_worker, initargs=worker_args) as pool:
                results = list(pool.map(_grow_tree, seeds))

        self.trees = []
        oob_sum = np.zeros((n, self.n_classes))
        oob_count = np.zeros(n)
        for tree, rows in results:
            self.trees.append(tree)
            if self.bootstrap:
                out_of_bag = np.ones(n, dtype=bool)
                out_of_bag[rows] = False
                oob_sum[out_of_bag] += tree.predict_proba(X[out_of_bag])
                oob_count[out_of_bag] += 1
        if self.bootstrap:
            seen = oob_count > 0
            self.oob_score = float(np.mean(oob_sum[seen].argmax(1) == y[seen]))

        # mean of the trees' Gini importance (same definition as the single tree)
        props = np.mean([t.feature_importances for t in self.trees], axis=0)
        self.feature_importances = props / props.sum() if props.sum() > 0 else props
        return self

    def predict_proba(self, X):
        total = np.zeros((len(X), self.n_classes))
        for tree in self.trees:
            total += tree.predict_proba(X)
        return total / len(self.trees)

    def staged_predict_proba(self, X):
        """Prediction of the forest made of the first 1, 2, ..., n_estimators trees."""
        total = np.zeros((len(X), self.n_classes))
        for t, tree in enumerate(self.trees, start=1):
            total += tree.predict_proba(X)
            yield total / t

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)
