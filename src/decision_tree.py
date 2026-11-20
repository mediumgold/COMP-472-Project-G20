"""
Decision Tree classifier (CART style), written from scratch with NumPy.

How it learns: start with all training rows in one node. Try every question
"is feature j <= t ?" and keep the one whose two children are the most pure
(lowest Gini impurity or entropy). Repeat in each child until a stopping rule
says stop. A leaf predicts the class proportions of the training rows in it.

Hyperparameters
  max_depth          max number of questions on a path (None = no limit)
  min_samples_split  a node needs at least this many rows to be split
  min_samples_leaf   each child must keep at least this many rows
  criterion          "gini" (1 - sum p^2) or "entropy" (-sum p log p)
  max_features       features tried per node: None = all, "sqrt", "log2", a fraction or a number
                     (the Random Forest uses "sqrt")
  n_bins             see below

Speed trick: instead of trying every distinct value of a feature as a threshold,
each feature gets n_bins - 1 candidate thresholds at its quantiles (31 for n_bins=32).
Every value is replaced by the number of its bin, so for one node we can count the
classes on each side of every threshold of every feature with a single np.bincount
and a cumulative sum. Notebook 03 shows 32 bins is as accurate as 64 or 128.
"""
import numpy as np


class Binner:
    """Finds the candidate thresholds of each feature and converts values to bin numbers."""

    def __init__(self, n_bins=32):
        self.n_bins = n_bins

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        quantiles = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        Q = np.quantile(X, quantiles, axis=0).astype(np.float32)   # (n_bins - 1, n_features)
        n_features = X.shape[1]
        self.edges = [np.unique(Q[:, j]) for j in range(n_features)]   # drop repeated thresholds
        # thresholds[j, b] = t such that "bin <= b" is the same question as "x <= t"
        self.thresholds = np.full((n_features, self.n_bins), np.inf, dtype=np.float32)
        n_thresholds = np.array([len(e) for e in self.edges])
        for j, e in enumerate(self.edges):
            self.thresholds[j, :len(e)] = e
        # valid[j, b] is False for the unused threshold slots of features with few distinct values
        self.valid = np.arange(self.n_bins)[None, :] < n_thresholds[:, None]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        Xb = np.empty(X.shape, dtype=np.uint8)
        for j, e in enumerate(self.edges):
            Xb[:, j] = np.searchsorted(e, X[:, j], side="left")
        return Xb


def xlogx(a):
    a = np.asarray(a, dtype=np.float64)
    out = np.zeros_like(a)
    out[a > 0] = a[a > 0] * np.log(a[a > 0])
    return out


class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 criterion="gini", max_features=None, n_bins=32, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.max_features = max_features
        self.n_bins = n_bins
        self.random_state = random_state

    # ------------------------------------------------------------------ impurity
    def impurity(self, counts):
        p = counts / counts.sum()
        if self.criterion == "gini":
            return 1.0 - np.sum(p ** 2)
        p = p[p > 0]
        return -np.sum(p * np.log(p))

    def children_impurity(self, left, right, n_left, n_right, n):
        """
        Weighted impurity (n_left * imp(left) + n_right * imp(right)) / n for every
        (feature, threshold) at once. left/right hold class counts: (features, thresholds, classes).
        """
        if self.criterion == "gini":
            # n * gini = n - sum_c count_c^2 / n
            g = (n_left - (left ** 2).sum(-1) / np.maximum(n_left, 1)) + \
                (n_right - (right ** 2).sum(-1) / np.maximum(n_right, 1))
        else:
            # n * entropy = n log n - sum_c count_c log count_c
            g = (xlogx(n_left) - xlogx(left).sum(-1)) + (xlogx(n_right) - xlogx(right).sum(-1))
        return g / n

    def n_features_to_try(self, d):
        mf = self.max_features
        if mf is None:
            return d
        if mf == "sqrt":
            return max(1, int(np.sqrt(d)))
        if mf == "log2":
            return max(1, int(np.log2(d)))
        if isinstance(mf, float):
            return max(1, int(mf * d))
        return min(d, int(mf))

    # ------------------------------------------------------------- split search
    def best_split(self, Xb, rows, y_node, parent_impurity, rng):
        """Best (feature, bin, gain) for the node holding `rows`, or None if no split helps."""
        n, d = len(rows), Xb.shape[1]
        B = self.n_bins
        # only the classes present in this node matter, so renumber them 0..C-1 (smaller histograms)
        _, y_node = np.unique(y_node, return_inverse=True)
        C = int(y_node.max()) + 1

        k = self.n_features_to_try(d)
        if k >= d:
            feature_groups = [np.arange(d)]
        else:
            # random subset of k features; if none of them can split the node, try the next k
            perm = rng.permutation(d)
            feature_groups = [perm[i:i + k] for i in range(0, d, k)]

        for feats in feature_groups:
            bins = Xb[np.ix_(rows, feats)].astype(np.int64)          # (n, m) bin numbers
            m = len(feats)
            # hist[f, b, c] = number of rows of class c with bin b for feature f
            code = ((np.arange(m) * B)[None, :] + bins) * C + y_node[:, None]
            hist = np.bincount(code.ravel(), minlength=m * B * C).reshape(m, B, C)
            left = np.cumsum(hist, axis=1).astype(np.float64)       # class counts with bin <= b
            right = left[:, -1:, :] - left
            n_left = left.sum(-1)
            n_right = n - n_left
            ok = ((n_left >= self.min_samples_leaf) & (n_right >= self.min_samples_leaf)
                  & self.valid_thresholds[feats])
            if not ok.any():
                continue
            child = self.children_impurity(left, right, n_left, n_right, n)
            child[~ok] = np.inf
            f, b = np.unravel_index(np.argmin(child), child.shape)
            gain = parent_impurity - child[f, b]
            if gain <= 1e-12:
                return None
            return int(feats[f]), int(b), float(gain)
        return None

    # ------------------------------------------------------------------ fitting
    def fit(self, X, y, n_classes=None):
        y = np.asarray(y, dtype=np.int64)
        binner = Binner(self.n_bins).fit(X)
        return self.fit_binned(binner.transform(X), y, binner, np.arange(len(y)), n_classes)

    def fit_binned(self, Xb, y, binner, rows, n_classes=None):
        """Grow the tree on `rows` of already-binned data (the Random Forest reuses this)."""
        self.n_classes = int(n_classes or (y.max() + 1))
        self.valid_thresholds = binner.valid
        rng = np.random.default_rng(self.random_state)
        max_depth = np.inf if self.max_depth is None else self.max_depth

        # The tree is stored as parallel lists, one entry per node.
        # feature = -1 means the node is a leaf.
        feature, threshold, left_child, right_child, value, depth = [], [], [], [], [], []
        importances = np.zeros(Xb.shape[1])

        def add_node(counts, d):
            feature.append(-1)
            threshold.append(np.nan)
            left_child.append(-1)
            right_child.append(-1)
            value.append(counts / counts.sum())
            depth.append(d)
            return len(feature) - 1

        root_counts = np.bincount(y[rows], minlength=self.n_classes).astype(np.float64)
        stack = [(add_node(root_counts, 0), rows, 0)]     # nodes still to be split
        n_total = len(rows)
        while stack:
            node, node_rows, d = stack.pop()
            n = len(node_rows)
            y_node = y[node_rows]
            counts = np.bincount(y_node, minlength=self.n_classes)
            if (d >= max_depth or n < self.min_samples_split
                    or n < 2 * self.min_samples_leaf or counts.max() == n):
                continue                                   # stopping rule: stays a leaf
            split = self.best_split(Xb, node_rows, y_node, self.impurity(counts), rng)
            if split is None:
                continue
            f, b, gain = split
            goes_left = Xb[node_rows, f] <= b
            rows_l, rows_r = node_rows[goes_left], node_rows[~goes_left]
            importances[f] += gain * n / n_total
            feature[node] = f
            threshold[node] = binner.thresholds[f, b]
            l = add_node(np.bincount(y[rows_l], minlength=self.n_classes).astype(np.float64), d + 1)
            r = add_node(np.bincount(y[rows_r], minlength=self.n_classes).astype(np.float64), d + 1)
            left_child[node], right_child[node] = l, r
            stack.append((r, rows_r, d + 1))
            stack.append((l, rows_l, d + 1))

        self.feature = np.array(feature, dtype=np.int64)
        self.threshold = np.array(threshold, dtype=np.float32)
        self.left = np.array(left_child, dtype=np.int64)
        self.right = np.array(right_child, dtype=np.int64)
        self.value = np.array(value, dtype=np.float32)
        self.node_depth = np.array(depth, dtype=np.int64)
        total = importances.sum()
        self.feature_importances = importances / total if total > 0 else importances
        return self

    # --------------------------------------------------------------- prediction
    @property
    def n_leaves(self):
        return int((self.feature < 0).sum())

    @property
    def depth(self):
        return int(self.node_depth.max())

    def apply(self, X, max_depth=None):
        """
        Node that each row ends up in. With max_depth we stop early, which gives exactly
        the prediction of a tree trained with that max_depth (the tree grows top-down).
        """
        X = np.asarray(X, dtype=np.float32)
        node = np.zeros(len(X), dtype=np.int64)
        active = np.arange(len(X))
        d = 0
        while active.size and (max_depth is None or d < max_depth):
            active = active[self.feature[node[active]] >= 0]       # rows already in a leaf are done
            if not active.size:
                break
            nd = node[active]
            goes_left = X[active, self.feature[nd]] <= self.threshold[nd]
            node[active] = np.where(goes_left, self.left[nd], self.right[nd])
            d += 1
        return node

    def predict_proba(self, X, max_depth=None):
        return self.value[self.apply(X, max_depth)]

    def predict(self, X, max_depth=None):
        return self.predict_proba(X, max_depth).argmax(axis=1)
