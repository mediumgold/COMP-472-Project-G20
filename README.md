# COMP 472 Project: Visual Object Recognition & Sentiment Analysis

Group G20 · Concordia University · Fall 2026

We compare four classifiers on two datasets of pre-extracted features:

* **k-Nearest Neighbours**, **Decision Tree** and **Random Forest**, written from scratch in NumPy
* a feed-forward **Neural Network** in PyTorch

plus an unsupervised **k-Means** clustering experiment (also from scratch).

| Task | Training rows | Classes | Features |
|---|---|---|---|
| Image (object recognition) | 20,000 (400 per class) | 50 | 500-dim bag-of-visual-words histograms |
| Sentiment (TweetEval) | 12,000 (4,000 per class) | 3 (Negative / Neutral / Positive) | 768-dim BERT embeddings |

## Results (validation accuracy, best configuration of each method)

| Method | Image | Sentiment |
|---|---|---|
| k-NN | **61.5%** | 59.3% |
| Decision Tree | 11.9% | 49.0% |
| Random Forest | 36.6% | 60.3% |
| Neural Network | 38.7% | **68.1%** |

Chance level is 2% for images and 33% for sentiment. Submitted to Kaggle: k-NN for images and an
ensemble of 5 networks for sentiment. See [FINDINGS.md](FINDINGS.md) for what we learned and why.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download the five competition CSVs from Kaggle into a `data/` folder (it is not in git because
`sentiment_train.csv` is over GitHub's 100 MB limit):

```
data/image_train.csv  data/image_test.csv  data/sentiment_train.csv  data/sentiment_test.csv  data/sample_submission.csv
```

**Quick version:** `python main.py` trains every model's final configuration on both datasets and
prints the table above (about 25 minutes on a laptop, mostly the Random Forest). `python main.py --submission` also writes `submission.csv`.

**Full experiments:** the notebooks in `notebooks/` contain every sweep and plot and are saved with
their outputs, so you can read them without running anything. To re-run, open `jupyter lab` and run
them in order (01 to 08). Notebooks 04 (Random Forest) and 05 (Neural Network) take the longest.

## Project layout

```
main.py                    one command: final configuration of every model on both tasks
src/
  data.py                  loading the CSVs, the train/validation split, k-fold split
  preprocessing.py         standardize, Hellinger, L2, PCA
  knn.py                   k-NN (from scratch)
  decision_tree.py         Decision Tree (from scratch)
  random_forest.py         Random Forest (from scratch, uses our decision tree)
  neural_net.py            MLP and training loop (PyTorch)
  kmeans.py                k-Means (from scratch)
  utils.py                 accuracy, timing, saving results, plot style
notebooks/
  01_data_exploration      what the features look like
  02_knn                   distance, preprocessing, k, voting, 5-fold check
  03_decision_tree         depth, criterion, leaf / split size
  04_random_forest         number of trees, features per split, depth; vs a single tree
  05_neural_network        preprocessing, learning rate, size, activation, regularization, ensemble
  06_kmeans_clustering     do the features cluster by class?
  07_model_comparison      results table, per-class scores, shared mistakes, cost
  08_final_submission      choose the final models, retrain on all data, write submission.csv
results/figures/           all plots
results/tables/            a few summary tables (the notebooks write more, but those aren't committed)
```

## Method

* **One fixed validation split** for everything: 80% train / 20% validation, stratified (each class
  keeps its share), seed 42. Every model and hyperparameter is compared on the same rows.
* **No leakage:** anything learned from data (standardization, PCA) is fitted on the training part only.
* **Choices are made on validation only**, never on the Kaggle leaderboard. The final models are then
  retrained on all labelled rows.
* With 4,000 / 2,400 validation rows, differences under about 1 point are noise. For the image k-NN,
  where it mattered, we confirmed the result with 5-fold cross-validation.

## What we wrote vs. what comes from libraries

| Part | Ours | Library |
|---|---|---|
| k-NN, Decision Tree, Random Forest, k-Means | yes (NumPy) | |
| Preprocessing (standardize, Hellinger, PCA via SVD) | yes (NumPy) | |
| Neural network architecture and training loop | yes | PyTorch (`torch.nn`, `torch.optim`) |
| Metrics (F1, confusion matrix, NMI, ARI) | | scikit-learn `sklearn.metrics` |
| Tables and plots | | pandas, matplotlib |

No scikit-learn models are used.

## Team

* Manlu Xu: neural network, k-Means clustering
* Hz80th: decision tree and random forest
* mediumgold: k-NN, data exploration, model comparison and submission

## Acknowledgements

This code was developed with the help of an AI tool (Claude, by Anthropic), as the project rules
require us to acknowledge. Every team member should be able to explain any part of `src/` and the notebooks.
