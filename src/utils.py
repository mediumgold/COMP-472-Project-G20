"""Small helpers shared by the notebooks: accuracy, timing, saving results, plot style."""
import json
import random
import time

import matplotlib
import numpy as np

from .data import ROOT, SEED

RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"
PRED_DIR = RESULTS_DIR / "predictions"     # validation predictions, read by notebooks 07 and 08
for d in (FIG_DIR, TABLE_DIR, PRED_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Same colour for the same thing in every figure
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MODEL_NAMES = {"knn": "k-NN", "dt": "Decision Tree", "rf": "Random Forest", "nn": "Neural Network"}


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


class Timer:
    """Usage:  with Timer() as t: ...   then t.seconds"""

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.seconds = time.perf_counter() - self.start


def save_table(df, name):
    df.to_csv(TABLE_DIR / f"{name}.csv", index=False)


def save_figure(fig, name):
    fig.savefig(FIG_DIR / f"{name}.png")


def save_predictions(task, model, y_pred, proba, info):
    """Save a model's validation predictions + a small json with its config and scores."""
    np.savez(PRED_DIR / f"{task}_{model}.npz", y_pred=y_pred, proba=proba)
    with open(PRED_DIR / f"{task}_{model}.json", "w") as f:
        json.dump(info, f, indent=2)


def load_predictions(task, model):
    d = np.load(PRED_DIR / f"{task}_{model}.npz")
    with open(PRED_DIR / f"{task}_{model}.json") as f:
        info = json.load(f)
    return d["y_pred"], d["proba"], info


matplotlib.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e0",
    "axes.prop_cycle": matplotlib.cycler(color=COLORS),
    "lines.linewidth": 2.0, "lines.markersize": 5,
    "font.size": 10, "axes.titlesize": 11, "legend.frameon": False,
})
