# Findings

All numbers come from the notebooks (saved with their outputs). "Validation" means our fixed,
stratified 20% hold-out: 4,000 images and 2,400 tweets. Differences under about 1 point are noise.

## 1. Results

| Method | Image (chance 2%) | Sentiment (chance 33%) | Best configuration |
|---|---|---|---|
| k-NN | **61.5%** | 59.3% | Image: raw histograms, histogram intersection, k = 20, votes ~ 1/distance^16. Sentiment: raw, cosine, k = 50, votes ~ 1/distance^2 |
| Decision Tree | 11.9% | 49.0% | Gini, no depth limit, at least 50 rows per leaf |
| Random Forest | 36.6% | 60.3% | 400 trees, sqrt(d) features per split, fully grown trees |
| Neural Network | 38.7% | **68.1%** | Image: 500-1024-512-50, ReLU, Adam, standardized input. Sentiment: 768-512-256-3, dropout 0.5, weight decay 1e-4 |

The network numbers are the best run of the search; averaged over 5 seeds they are 38.6% and 68.1%.

**Submitted to Kaggle:** k-NN for images (trained on all 20,000 images) and the average of 5 networks
for sentiment (retrained on all 12,000 tweets for 9 epochs, validation 67.9%).

## 2. The two datasets are very different (notebook 01)

* **Images are bag-of-visual-words histograms**: non-negative, ~85% zeros, length exactly 1, values are
  scaled counts of 500 "visual words". A single feature is weak evidence, which is bad for trees.
* **Tweets are dense BERT embeddings**, with a few dimensions on very different scales.
* Neither shows its classes in a 2-D PCA plot. Along the Negative-Positive direction, **Neutral sits in
  the middle and overlaps both**, which already suggests Neutral will be the hardest class.

## 3. Each algorithm

### k-NN (notebook 02)
* **The distance function mattered most for images.** Cosine/Euclidean on raw histograms: 51.9%;
  after the Hellinger transform: 56.3%; with **histogram intersection** on the raw histograms: 58.9%.
  Intersection adds up min(a_i, b_i), so a very frequent visual word only counts as much as both
  images share; the damping Hellinger does by hand is built in.
* **Strong distance weighting** (votes ~ 1/distance^16) lets the closest few neighbours decide and
  raises images to **61.5%** (cosine: 58.05%). Sentiment prefers mild weighting (p = 2) and k = 50.
* **k works in opposite directions:** on images, accuracy *drops* as k grows with equal votes
  (underfitting); on sentiment, k = 1 is worst (48.7%, overfitting to single noisy tweets).
* **Why:** when an image's nearest training image is very close, 1-NN is right 93% of the time; when
  it is far, only 20%. Image classes are made of many small groups of near-relatives (probably several
  photos of the same object). Tweets have no such groups, so averaging many neighbours helps.
* **5-fold cross-validation** confirms intersection is really better: 62.8% vs 59.1% for the best
  cosine setting, better on all 5 folds.
* Tried and dropped: TF-IDF weighting of visual words (+0.8, too small to matter), PCA before k-NN
  (worse), k-NN on a trained network's hidden layer (much worse, about 41%).

### Decision Tree (notebook 03)
* **Sentiment shows textbook under/overfitting:** depth 1-3 underfits, validation peaks at depth 5
  (48.5%), deeper trees reach 100% training accuracy while validation drops to 42.5%.
* **Images overfit at every depth:** 100% training accuracy, about 11% validation. Each split asks about
  one visual word, and the tree runs out of rows long before combining enough weak evidence.
* **min_samples_leaf** is the most useful brake: sentiment goes from 42.5% (leaf 1) to 49.0% (leaf 50).
* Gini vs entropy: no real difference. Our tree uses 31 quantile thresholds per feature for speed;
  64 or 128 are not more accurate.

### Random Forest (notebook 04)
* **Much better than one tree:** +24.7 points on images, +11.3 on sentiment.
* **Number of trees:** sentiment levels off around 60% after 100-150 trees; images are still improving
  at 400 trees (each tree is a very noisy voter over 50 classes).
* **Fully grown trees are best**, unlike for a single tree: deep trees have low bias and high variance,
  and averaging removes the variance. Training accuracy is still 100%, but validation is far higher.
* max_features: log2 is too few; sqrt(d) and 10% are close (at 400 trees sqrt(d) won, 36.6% vs 35.6%).

### Neural Network (notebook 05)
* **Overfits quickly:** 100% training accuracy within ~10 epochs on images; we keep the best
  validation epoch (early stopping).
* **Images:** wider is better (no hidden layer 31.5%, 256-64 33.3%, 1024-512 38.7%); ReLU/GELU beat
  tanh/sigmoid; lr 0.01 is too big. An ensemble of 5 networks reaches 40.8%.
* **Sentiment:** everything lands between 67% and 68%, *including no hidden layer at all* (logistic
  regression, 67.5%). The features seem to hold about 68% worth of linearly usable information; extra
  layers only overfit. Dropout 0.5 doesn't raise the peak but makes training much more forgiving.

## 4. Comparing the methods

* **The ranking flips between tasks.** Images: k-NN >> NN > RF >> DT. Sentiment: NN > RF ~ k-NN >> DT.
  k-NN thrives on local structure (groups of near-relatives); the network thrives when the signal is
  spread thinly over many dimensions. One-feature-at-a-time splits suit neither, so the tree is last.
* **More complex is not automatically better.** On sentiment, logistic regression is within noise of the
  best network. On images the "simplest" method, k-NN, wins by over 20 points because it matches the
  structure of the data.
* **Ensembles** help unstable models (5 image networks: 38.6% -> 40.8%) but not stable ones (sentiment:
  68.1% -> 67.9%), and averaging k-NN with the networks on images *hurts* (53.7% vs 61.5% for k-NN alone).

## 5. Clustering (notebook 06)

* **Tweets don't cluster by sentiment:** with k = 3 every cluster has plenty of all three labels
  (purity 0.42 vs 0.34 for random clusters, NMI 0.02).
* **Image clusters follow the classes only weakly** (k = 50: purity 0.11 vs 0.04 random), and keep improving
  up to k = 200, consistent with "many small groups per class".
* The class pairs k-Means mixes up are often the same pairs the classifiers confuse (notebook 07).

## 6. Error analysis (notebook 07)

* **Neutral is the hardest sentiment class for every model** (F1 0.45-0.59). The network sends 22% of
  Neutral tweets to Negative and 22% to Positive; Negative -> Positive mistakes are rare (7%). k-NN
  almost refuses to predict Neutral (recall 35%), because neutral tweets sit between the other two.
* **Image confusions:** the models agree on which classes are hard (rank correlation 0.67-0.86;
  hardest 29, 22, 9, 27, 40). The most confused pairs (37 <-> 39, 30 <-> 31, 2 <-> 3, ...) are mostly
  the same pairs k-Means puts in the same clusters: they really are close in feature space.
* **The models make the same mistakes:** 28.4% of validation images are missed by all four models
  (13.2% expected if they were independent); 15.5% of tweets (vs 2.6%).
* **Does the network fix the others' mistakes?** On sentiment, yes: it gets right 42% of k-NN's and
  37% of the forest's mistakes, while breaking only 11-14% of their correct answers. On images, no:
  it fixes 17% of k-NN's mistakes but breaks 47% of its correct answers.
* **The network is over-confident on images** (predictions with confidence >= 0.95 are right only 61% of
  the time) but well calibrated on sentiment (96%).

## 7. How the final model was chosen (notebook 08)

1. Every choice was made on validation, never on the Kaggle leaderboard.
2. Candidates: k-NN, Random Forest, the 5-network ensemble, and averages of these. Images: k-NN alone
   won. Sentiment: the 5-network ensemble won.
3. We did not pick the single best network (68.1%): that number is the winner of the search on the same
   validation set, so it is inflated. The 5-network ensemble (67.9%) is within noise of it and of the
   5-seed mean (68.1%), and doesn't depend on one lucky seed.
4. The chosen models were retrained on all labelled rows. The networks were trained for 9 epochs (the
   median best epoch on validation), since no validation set is left for early stopping.

## 8. Caveats

* **Sentiment test predictions lean Negative** (47% of test tweets vs 37% on the balanced validation
  set). The test set probably has a different class mix (TweetEval's original test split is not
  balanced). We did not adjust anything based on the test set.
* **Image predictions are not perfectly even** across classes (62-138 per class, the test set has 100
  of each), though much more even than with cosine distance (44-175).
* The neural-network numbers can move by a few tenths of a point between computers (floating-point
  differences between platforms add up over training). Everything else is exactly reproducible.
* Best-of-search validation scores are slightly optimistic; expect the leaderboard within a couple of points.
