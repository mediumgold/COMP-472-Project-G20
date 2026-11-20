"""
Run all four classifiers on both datasets with one command.

Uses the same 80/20 validation split as the notebooks and each model's best
configuration (found in notebooks 02-05), and prints the validation accuracy table.

    python main.py                     # all 4 models x both tasks
    python main.py --task sentiment    # one task only
    python main.py --models knn,nn     # some models only (knn, dt, rf, nn)
    python main.py --rf-trees 100      # quicker Random Forest (400 = the reported result)
    python main.py --submission        # also retrain the final models on all data -> submission.csv

The notebooks contain the full experiments; this script only re-runs the final settings.
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch

from src.data import get_split, load_task, DATA_DIR, ROOT, N_CLASSES, SEED
from src.preprocessing import Preprocessor
from src.knn import KNNClassifier
from src.decision_tree import DecisionTreeClassifier
from src.random_forest import RandomForestClassifier
from src.neural_net import train_mlp, predict_proba
from src.utils import MODEL_NAMES, TABLE_DIR, accuracy, set_seed

torch.set_num_threads(1)   # small networks: 1 thread is fastest

# Best configuration of each model per task, chosen on the validation set
BEST = {
    "image": {
        "knn": dict(preprocessing="none", k=20, metric="intersection", weights="distance", power=16),  # nb 02
        "dt": dict(criterion="gini", max_depth=None, min_samples_leaf=50),                              # nb 03
        "rf": dict(max_features="sqrt", max_depth=None, min_samples_leaf=1),                            # nb 04
        "nn": dict(preprocessing="standardize", hidden=(1024, 512), activation="relu", dropout=0.0,
                   lr=1e-3, optimizer="adam", weight_decay=0.0, batch_size=128, epochs=60),             # nb 05
    },
    "sentiment": {
        "knn": dict(preprocessing="none", k=50, metric="cosine", weights="distance", power=2),
        "dt": dict(criterion="gini", max_depth=None, min_samples_leaf=50),
        "rf": dict(max_features="sqrt", max_depth=None, min_samples_leaf=1),
        "nn": dict(preprocessing="none", hidden=(512, 256), activation="relu", dropout=0.5,
                   lr=1e-3, optimizer="adam", weight_decay=1e-4, batch_size=128, epochs=80),
    },
}
NN_PATIENCE = 15        # early stopping on the validation set

# Final Kaggle models (chosen in notebook 08):
#   image     -> the k-NN above, with all 20,000 training images as neighbours
#   sentiment -> average of 5 networks (seeds 42-46) trained on all 12,000 tweets for 9 epochs
#                (the median best epoch of the 5 seeds in notebook 05)
FINAL_NN_SEEDS = 5
FINAL_NN_EPOCHS = 9


def run_model(model, task, Xtr, ytr, Xva, yva, rf_trees, n_jobs):
    """Train one model on the training split and return its validation predictions."""
    cfg = dict(BEST[task][model])
    prep = Preprocessor(cfg.pop("preprocessing", "none")).fit(Xtr)
    A, B = prep.transform(Xtr), prep.transform(Xva)
    n_classes = N_CLASSES[task]
    if model == "knn":
        return KNNClassifier(n_classes=n_classes, **cfg).fit(A, ytr).predict(B)
    if model == "dt":
        return DecisionTreeClassifier(**cfg).fit(Xtr, ytr, n_classes).predict(Xva)
    if model == "rf":
        rf = RandomForestClassifier(n_estimators=rf_trees, random_state=SEED, n_jobs=n_jobs, **cfg)
        return rf.fit(Xtr, ytr, n_classes).predict(Xva)
    net, _, _ = train_mlp(A, ytr, B, yva, n_classes=n_classes, seed=SEED, patience=NN_PATIENCE, **cfg)
    return predict_proba(net, B).argmax(1)


def validation_table(tasks, models, rf_trees, n_jobs):
    rows = []
    for task in tasks:
        Xtr, ytr, Xva, yva = get_split(task)
        print(f"\n=== {task}: {len(ytr)} training rows, {len(yva)} validation rows ===")
        for m in models:
            set_seed()
            start = time.perf_counter()
            acc = accuracy(yva, run_model(m, task, Xtr, ytr, Xva, yva, rf_trees, n_jobs))
            seconds = time.perf_counter() - start
            rows.append(dict(task=task, method=MODEL_NAMES[m], val_acc=acc, seconds=round(seconds, 1)))
            print(f"  {MODEL_NAMES[m]:15s} validation accuracy {acc:.4f}   ({seconds:.1f}s)")
    df = pd.DataFrame(rows)
    table = df.pivot(index="method", columns="task", values="val_acc").reindex([MODEL_NAMES[m] for m in models])
    print("\n=== Validation accuracy ===")
    print(table.map(lambda v: f"{v:.1%}").to_string())
    df.to_csv(TABLE_DIR / "main_results.csv", index=False)


def make_submission():
    """Retrain the final models on ALL labelled rows and write submission.csv."""
    print("\n=== Building submission.csv ===")
    predictions = []

    X, y, X_test, ids = load_task("image")
    cfg = dict(BEST["image"]["knn"])
    prep = Preprocessor(cfg.pop("preprocessing")).fit(X)
    knn = KNNClassifier(n_classes=50, **cfg).fit(prep.transform(X), y)
    predictions.append(pd.DataFrame({"id": ids, "label": knn.predict(prep.transform(X_test))}))
    print("  image: k-NN done")

    X, y, X_test, ids = load_task("sentiment")
    cfg = dict(BEST["sentiment"]["nn"], epochs=FINAL_NN_EPOCHS)
    prep = Preprocessor(cfg.pop("preprocessing")).fit(X)
    A, T = prep.transform(X), prep.transform(X_test)
    probs = []
    for s in range(FINAL_NN_SEEDS):
        net, _, _ = train_mlp(A, y, n_classes=3, seed=SEED + s, **cfg)
        probs.append(predict_proba(net, T))
    predictions.append(pd.DataFrame({"id": ids, "label": np.mean(probs, axis=0).argmax(1)}))
    print(f"  sentiment: {FINAL_NN_SEEDS}-network ensemble done")

    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    sub = sample[["id"]].merge(pd.concat(predictions, ignore_index=True), on="id", how="left",
                               validate="one_to_one")
    assert len(sub) == 10000 and sub["label"].notna().all()
    sub["label"] = sub["label"].astype(int)
    sub.to_csv(ROOT / "submission.csv", index=False)
    print("  saved submission.csv")


def main():
    parser = argparse.ArgumentParser(description="Run all classifiers on both COMP 472 datasets.")
    parser.add_argument("--task", choices=["image", "sentiment", "both"], default="both")
    parser.add_argument("--models", default="knn,dt,rf,nn", help="comma-separated: knn, dt, rf, nn")
    parser.add_argument("--rf-trees", type=int, default=400)
    parser.add_argument("--jobs", type=int, default=4, help="CPU processes for the Random Forest")
    parser.add_argument("--submission", action="store_true", help="also write submission.csv")
    parser.add_argument("--submission-only", action="store_true", help="only write submission.csv")
    args = parser.parse_args()

    tasks = ["image", "sentiment"] if args.task == "both" else [args.task]
    models = [m.strip() for m in args.models.split(",")]
    for m in models:
        if m not in MODEL_NAMES:
            parser.error(f"unknown model {m}; choose from knn, dt, rf, nn")

    start = time.perf_counter()
    if not args.submission_only:
        validation_table(tasks, models, args.rf_trees, args.jobs)
    if args.submission or args.submission_only:
        make_submission()
    print(f"\nTotal time: {(time.perf_counter() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
