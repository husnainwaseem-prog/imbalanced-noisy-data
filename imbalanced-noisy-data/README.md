# Handling Imbalanced and Noisy Data

Fraud detection on a highly imbalanced dataset (0.49% positives):
compare class weights, SMOTE oversampling, undersampling and threshold
tuning on the same logistic-regression model, and show why accuracy
is misleading when fraud is rare.

## Results (test set, 20,099 rows, 99 frauds, natural prevalence)

| Approach | Thr | Accuracy | Precision | Recall | F1 | PR-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | 0.50 | 0.9989 | 0.9753 | 0.7980 | 0.8778 | 0.8985 |
| Class weights | 0.50 | 0.9746 | 0.1544 | 0.9293 | 0.2647 | 0.8689 |
| SMOTE | 0.50 | 0.9816 | 0.2022 | 0.9293 | 0.3321 | 0.8739 |
| Undersampling | 0.50 | 0.9672 | 0.1235 | 0.9293 | 0.2180 | 0.7495 |
| Threshold tuned (max F1) | 0.98 | 0.9982 | 0.7890 | 0.8687 | 0.8269 | 0.8689 |
| **Recommended: recall ≥ 0.90** | **0.59** | 0.9815 | 0.2000 | 0.9192 | 0.3285 | 0.8689 |
| Dummy (all negative) | — | 0.9951 | 0.00 | 0.00 | 0.00 | 0.0049 |

Headline: the dummy all-negative classifier scores **99.51% accuracy
and catches zero fraud**. Accuracy cannot tell a good fraud model from
a useless one here.

**Recommendation:** class-weighted logistic regression at threshold
**0.59** (recall ≥ 0.90 operating point) — catches 91 of 99 test
frauds, misses 8, at ~18 false alarms per 1,000 transactions. Full
justification in `REPORT.md` §6.

Also measured: with 5% of training labels flipped, SMOTE's F1
collapses the most (0.33 → 0.02) — it amplifies label noise — while
the baseline barely moves. See `REPORT.md` §5.

## Dataset

Kaggle credit-card fraud (284,807 transactions, 492 frauds). The
shipped working set (`data/creditcard.csv`) keeps all 492 fraud cases
+ a random 100,000 legitimate transactions (seed 42): 100,492 rows,
0.49% positives. The full 98 MB file is not included. Splits are
stratified 60/20/20 (seed 42). Resampling is applied to the training
fold only — the test set keeps its natural prevalence.

## How to run

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate          # macOS / Linux
pip install -r requirements.txt
python -m src.cli evaluate           # full comparison + plots (~20 s)
python -m src.cli compare            # print the saved table
python -m src.cli plot               # regenerate plots only
```

## Layout

```
src/
  data.py       load the working set, stratified 60/20/20 split
  models.py     baseline, class weights, SMOTE, undersampling, threshold tuning
  evaluate.py   metrics, dummy classifier, per-1000 framing, label noise
  plots.py      PR curves, threshold trade-off, confusion matrix
  cli.py        evaluate / compare / plot / run-all
tests/
  test_imbalanced.py   17 tests
data/
  creditcard.csv       working set (100,492 rows)
results/
  metrics.json / metrics.md    full results
  probas.npz                   test + validation probabilities
  plots/                       pr_curves.png, threshold_tradeoff.png,
                               confusion_recommended.png
```

## Tests

```bash
python -m pytest tests/ -q     # 17 passed
```

## Notes

- `imbalanced-learn` is used for SMOTE when installed; if it is not
  available the code falls back to a small numpy SMOTE in
  `src/models.py` automatically.
- Thresholds are tuned on the validation set, never on the test set.
