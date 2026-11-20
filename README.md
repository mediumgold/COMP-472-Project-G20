# COMP 472 Project: Visual Object Recognition & Sentiment Analysis

Group G20 · Concordia University · Fall 2026

We compare k-NN, a Decision Tree, a Random Forest (all written from scratch in NumPy) and a
feed-forward Neural Network (PyTorch) on two datasets of pre-extracted features:
image object recognition (50 classes) and tweet sentiment (3 classes).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Put the competition CSVs from Kaggle in a `data/` folder (not in git, `sentiment_train.csv` is too big).
