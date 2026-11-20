"""
Loading the competition data and splitting it into train / validation.

Every notebook and main.py uses get_split(), so all models are compared on the
exact same validation rows.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

SEED = 42
VAL_FRACTION = 0.2
N_CLASSES = {"image": 50, "sentiment": 3}
SENTIMENT_NAMES = ["Negative", "Neutral", "Positive"]


def read_csv(name):
    """Read one competition CSV and return (features, labels, ids). Test files have no labels."""
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - download the Kaggle CSVs into the data/ folder")
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int64) if "label" in df.columns else None
    ids = df["id"].to_numpy()
    return X, y, ids


def load_task(task):
    """
    Returns X_train, y_train, X_test, test_ids for task = "image" or "sentiment".
    Reading the big CSVs is slow, so the first call saves them to data/cache_<task>.npz.
    """
    cache = DATA_DIR / f"cache_{task}.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return d["X_train"], d["y_train"], d["X_test"], d["test_ids"]

    X_train, y_train, _ = read_csv(f"{task}_train")
    X_test, _, test_ids = read_csv(f"{task}_test")
    np.savez(cache, X_train=X_train, y_train=y_train, X_test=X_test, test_ids=test_ids)
    return X_train, y_train, X_test, test_ids


def train_val_split(y, val_fraction=VAL_FRACTION, seed=SEED):
    """
    Stratified split: every class is shuffled separately and 20% of it goes to validation,
    so both parts keep the same class balance (80 validation images per image class).
    Returns (train_idx, val_idx).
    """
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_fraction))
        val_idx.append(idx[:n_val])
        train_idx.append(idx[n_val:])
    return np.sort(np.concatenate(train_idx)), np.sort(np.concatenate(val_idx))


def get_split(task):
    """The one train/validation split used for every experiment: X_tr, y_tr, X_val, y_val."""
    X, y, _, _ = load_task(task)
    tr, va = train_val_split(y)
    return X[tr], y[tr], X[va], y[va]


def kfold_split(y, n_folds=5, seed=SEED):
    """
    Stratified k-fold: each class is shuffled and dealt out evenly over the folds.
    Returns a list of (train_idx, val_idx); every row is in exactly one validation fold.
    """
    rng = np.random.default_rng(seed)
    fold_of = np.empty(len(y), dtype=np.int64)
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        fold_of[idx] = np.arange(len(idx)) % n_folds
    return [(np.flatnonzero(fold_of != f), np.flatnonzero(fold_of == f)) for f in range(n_folds)]
