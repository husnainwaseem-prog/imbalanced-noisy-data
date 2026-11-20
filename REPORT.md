# Handling Imbalanced and Noisy Data — Report

## 1. Dataset

Real data: the Kaggle credit-card fraud set (284,807 European card
transactions, 492 frauds, 0.17% positives). The full file is ~98 MB,
so the project ships a working set instead: **all 492 fraud cases +
a random sample of 100,000 legitimate transactions (seed 42)** — 100,492
rows, 0.49% positives. Still far under 2%, and every fraud case is kept
because fraud is the scarcest resource in the data.

Features: the 28 anonymised PCA components (V1–V28) + `Amount`. `Time`
was dropped — it is just seconds since the first transaction and
carries no fraud signal. `Amount` is heavily skewed (fraud median
$9.25, mean $122.21, max $2,125.87), so it is scaled inside the model
pipeline rather than used raw.

Split (seed 42, stratified): train 60% (60,295 rows, 295 frauds),
validation 20% (20,098 rows, 98 frauds), test 20% (20,099 rows, 99
frauds). Stratification keeps the 0.49% fraud rate in every split.

**No-leakage rule, enforced in the code:** SMOTE and undersampling
are applied to the training fold only. The validation and test sets
keep their natural prevalence. Resampling the test set would
manufacture fraud cases the model is then tested on — that is leakage,
and it would make every metric meaningless.

## 2. Why accuracy is misleading here

The headline number first. A classifier that predicts "not fraud" for
every transaction scores:

| Classifier | Accuracy | Precision | Recall | F1 | PR-AUC |
| --- | --- | --- | --- | --- | --- |
| Dummy (all negative) | **0.9951** | 0.00 | 0.00 | 0.00 | 0.0049 |

99.51% accuracy — and it catches zero fraud. Accuracy is dominated by
the 99.5% of transactions that are legitimate, so it barely moves
whatever the model does: across all six rows in the results table it
sits between 0.967 and 0.999 while F1 ranges from 0.22 to 0.88. A
single accuracy number cannot tell a good fraud model from a useless
one. The metrics that matter here are precision, recall, F1 and
PR-AUC (average precision), because they are computed on the rare
class, not the easy majority.

## 3. The four approaches

Same base model throughout — logistic regression on scaled features —
so any difference comes from the imbalance handling, not the
classifier. Thresholds for the four approaches are the default 0.5;
threshold tuning is a separate approach on top of class weights.

1. **Baseline (no handling).** Plain logistic regression. The control
   row: it shows what happens when the imbalance is ignored.
2. **Class weights.** `class_weight="balanced"` — fraud examples count
   ~100x more in the loss, so the model stops ignoring them.
3. **SMOTE oversampling.** Synthetic fraud cases are interpolated
   between real fraud cases and their nearest neighbours, in the
   training fold only, until the classes are 50/50
   (imbalanced-learn; a small numpy SMOTE in `src/models.py` is the
   automatic fallback if imbalanced-learn is missing).
4. **Undersampling.** The majority class in the training fold is
   randomly cut down to the minority count (295 fraud + 295 legit =
   590 training rows). It throws away ~99% of the legitimate data.
5. **Threshold tuning.** The class-weighted model, but the decision
   threshold is tuned on the validation set (max F1 → 0.98), and a
   second operating point is tuned for recall ≥ 0.90 (→ 0.59).
   Thresholds are tuned on validation, never on test.

## 4. Results (test set, natural prevalence, 99 frauds)

| Approach | Thr | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 0.50 | 0.9989 | 0.9753 | 0.7980 | 0.8778 | 0.9827 | 0.8985 |
| Class weights | 0.50 | 0.9746 | 0.1544 | 0.9293 | 0.2647 | 0.9817 | 0.8689 |
| SMOTE | 0.50 | 0.9816 | 0.2022 | 0.9293 | 0.3321 | 0.9828 | 0.8739 |
| Undersampling | 0.50 | 0.9672 | 0.1235 | 0.9293 | 0.2180 | 0.9807 | 0.7495 |
| Threshold tuned (max F1) | 0.98 | 0.9982 | 0.7890 | 0.8687 | 0.8269 | 0.9817 | 0.8689 |
| **Recommended: recall ≥ 0.90** | **0.59** | 0.9815 | 0.2000 | 0.9192 | 0.3285 | 0.9817 | 0.8689 |
| Dummy (all negative) | — | 0.9951 | 0.00 | 0.00 | 0.00 | 0.50 | 0.0049 |

What the table says, per 1,000 transactions (≈ 5 frauds):

| Approach | Frauds caught | Frauds missed | False alarms |
| --- | --- | --- | --- |
| Baseline | 3.9 | 1.0 | 0.1 |
| Class weights | 4.6 | 0.3 | 25.1 |
| SMOTE | 4.6 | 0.3 | 18.1 |
| Undersampling | 4.6 | 0.3 | 32.5 |
| Threshold tuned (max F1) | 4.3 | 0.6 | 1.1 |
| **Recommended (recall ≥ 0.90)** | **4.5** | **0.4** | **18.1** |

Reading the results:

- **At the default 0.5 threshold, the imbalance handlers all catch
  the same frauds (recall 0.9293, 92 of 99) and differ only in how
  many false alarms they raise.** Class weights, SMOTE and
  undersampling buy recall with precision: 25, 18 and 33 false alarms
  per 1,000 transactions respectively.
- **The baseline has the best F1 (0.8778) and the best PR-AUC
  (0.8985) — but it misses 20 of 99 frauds.** F1 rewards the balance
  of precision and recall; this business does not value them equally.
- **Resampling does not improve the ranking.** ROC-AUC is ~0.98 for
  every approach and PR-AUC is highest for the baseline. The classes
  are already fairly separable, so manufacturing or deleting training
  rows adds nothing to the model's ability to rank fraud above legit.
  Undersampling is clearly the worst (PR-AUC 0.7495): training on 590
  rows throws away too much signal.
- **Threshold tuning is where the real gain is.** The same
  class-weighted model goes from F1 0.2647 at threshold 0.5 to 0.8269
  at its tuned threshold 0.98 — no retraining, just a different
  operating point. Note the threshold itself is extreme (0.98):
  class weighting inflates fraud probabilities, so the tuned
  threshold has to be high to compensate. Thresholds are not
  comparable across models; each one must be tuned on validation.

The PR curves (`results/plots/pr_curves.png`) show the same story:
baseline on top, SMOTE and class weights close behind, undersampling
clearly below, and the no-skill line flat at 0.005.

## 5. Noise robustness (5% of training labels flipped)

Real fraud labels are noisy — chargebacks arrive late, some fraud is
never reported, some legitimate transactions get flagged by mistake.
Retraining every approach with 5% of training labels flipped at
random, scored at the same 0.5 threshold:

| Approach | F1 clean → noisy | PR-AUC clean → noisy |
| --- | --- | --- |
| Baseline | 0.8778 → 0.8523 (−0.03) | 0.8985 → 0.8742 (−0.02) |
| Class weights | 0.2647 → 0.0460 (−0.22) | 0.8689 → 0.8670 (−0.00) |
| SMOTE | 0.3321 → 0.0247 (−0.31) | 0.8739 → 0.8097 (−0.06) |
| Undersampling | 0.2180 → 0.0338 (−0.18) | 0.7495 → 0.8108 (+0.06) |

Two findings:

- **SMOTE amplifies label noise.** It synthesises new points from
  minority neighbours, so a mislabelled fraud case breeds a cluster
  of fake fraud cases around it. Its F1 collapses the most (−0.31).
  On noisy real-world labels, SMOTE is the riskiest of the four.
- **Ranking survives better than the operating point.** Class
  weights' PR-AUC barely moves (−0.00) while its F1 at 0.5 collapses
  — the model still ranks fraud correctly, but the fixed 0.5
  threshold is no longer the right operating point once the labels
  are noisy. That is another argument for tuning the threshold on
  held-out data instead of trusting 0.5.
- Undersampling's PR-AUC moving up under noise is not a real effect:
  with only ~590 training rows its numbers swing widely run to run.
  Treat it as instability, not a benefit.

## 6. Recommendation

**Use the class-weighted logistic regression with the decision
threshold set to 0.59 — the operating point tuned for recall ≥ 0.90.**

On the test set it catches 91 of 99 frauds (recall 0.9192) and misses
8, at a precision of 0.20: 364 false alarms in 20,099 transactions,
about 18 per 1,000, or 2.3% of transactions sent to manual review.

Why this point, and not the max-F1 point (threshold 0.98)?

- The business cares more about missing fraud than about false
  alarms. The max-F1 point misses 13 frauds; the recommended point
  misses 8 — a ~38% cut in missed fraud.
- The cost asymmetry justifies it. A missed fraud costs the
  transaction value (mean $122 in this data) plus chargeback fees and
  the risk of repeat fraud on a compromised card. A false alarm costs
  a manual review — a few minutes of analyst time. Even at ~68 extra
  false alarms per extra fraud caught, the trade pays for itself if
  one review costs less than ~$2 of analyst time, which it does.
- The review volume is operable: 2.3% of transactions in a queue is a
  workload a fraud team can staff for; 0.5% would be nicer, but not
  at the price of 5 more missed frauds per 20,000 transactions.
- If review capacity is ever the binding constraint, fall back to the
  max-F1 threshold 0.98 (86 of 99 caught, only ~1 false alarm per
  1,000) — the same model, one number changed.

Why not SMOTE or undersampling? At the same recall they raise as many
or more false alarms, SMOTE is the most fragile under label noise,
and undersampling throws away 99% of the legitimate data for the
worst PR-AUC of the four. Class weights + threshold tuning gets the
same recall with none of those costs.

## 7. Limitations

- **Only 99 frauds in the test set.** Recall 0.92 means 91/99 — a
  handful of cases either way moves it by ~5 points. Treat the
  recall numbers as estimates with wide uncertainty, not exact.
- **Thresholds were tuned on 98 validation frauds** — same caveat.
  Re-tune the threshold on fresh data before deploying.
- **V1–V28 are anonymised PCA components**, so no domain features
  (merchant, device, geography) could be used. A production model
  would add those.
- **SMOTE points are interpolations**, not real transactions; the
  noise experiment shows they can mislead when labels are dirty.
- One dataset, one model family (logistic regression). Gradient
  boosting would likely rank slightly better, but the comparison
  between imbalance approaches — the point of this project — would
  not change.
