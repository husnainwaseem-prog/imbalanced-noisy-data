"""The four imbalance approaches (+ plain baseline), one base model.

Base model everywhere: logistic regression on scaled features
(StandardScaler fit on the training fold only, inside the pipeline).
Keeping the model fixed means any difference in the results comes from
the imbalance handling, not from the classifier.

Approaches (all resampling touches the TRAINING fold only):
  baseline      - plain logistic regression, no handling.
  class_weight  - LogisticRegression(class_weight="balanced").
  smote         - SMOTE oversampling of the training fold up to 50/50,
                  then plain logistic regression. Uses imbalanced-learn
                  when installed; otherwise the small numpy SMOTE in
                  this file (same idea: interpolate between a minority
                  point and one of its k nearest minority neighbours).
  undersample   - random undersampling of the majority class in the
                  training fold down to the minority count, then plain
                  logistic regression.
  threshold     - class-weighted model, but the decision threshold is
                  tuned on the VALIDATION set instead of using 0.5.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def make_lr(**kwargs) -> Pipeline:
    """Logistic regression on scaled features (scaler fit on train)."""
    params = {"max_iter": 1000, "solver": "lbfgs", "random_state": RANDOM_STATE}
    params.update(kwargs)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(**params)),
    ])


def _numpy_smote(X: np.ndarray, y: np.ndarray, seed: int = RANDOM_STATE,
                 k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Minimal SMOTE: synthesise minority points by interpolation.

    For each new point, pick a random minority example, pick one of its
    k nearest minority neighbours at random, and place the new point
    somewhere on the line between them. Returns a 50/50 dataset.
    """
    rng = np.random.default_rng(seed)
    X_min = X[y == 1]
    X_maj = X[y == 0]
    n_needed = len(X_maj) - len(X_min)
    if n_needed <= 0:
        return X, y
    k = min(k, len(X_min) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_min)
    _, idx = nn.kneighbors(X_min)
    synth = np.empty((n_needed, X.shape[1]))
    for i in range(n_needed):
        j = rng.integers(len(X_min))
        neighbour = idx[j, rng.integers(1, k + 1)]
        gap = rng.random()
        synth[i] = X_min[j] + gap * (X_min[neighbour] - X_min[j])
    X_new = np.vstack([X, synth])
    y_new = np.concatenate([y, np.ones(n_needed, dtype=y.dtype)])
    order = rng.permutation(len(y_new))
    return X_new[order], y_new[order]


def smote_resample(X_train, y_train, seed: int = RANDOM_STATE):
    """SMOTE the training fold to 50/50. Returns (X_res, y_res).

    Prefers imbalanced-learn's SMOTE; falls back to the numpy version
    above when imbalanced-learn is not installed. Returns a DataFrame
    with the original feature names, so the pipeline sees the same
    columns it was fitted on.
    """
    X = X_train.values if hasattr(X_train, "values") else np.asarray(X_train)
    y = y_train.values if hasattr(y_train, "values") else np.asarray(y_train)
    try:
        from imblearn.over_sampling import SMOTE
        X_res, y_res = SMOTE(random_state=seed).fit_resample(X, y)
    except ImportError:
        X_res, y_res = _numpy_smote(X, y, seed=seed)
    cols = list(X_train.columns) if hasattr(X_train, "columns") else None
    if cols is not None:
        X_res = pd.DataFrame(X_res, columns=cols)
    return X_res, y_res


def undersample(X_train, y_train, seed: int = RANDOM_STATE):
    """Random undersampling of the majority class to the minority count.

    Training fold only. Throws away most of the legitimate transactions,
    which is exactly the trade-off this approach is judged on.
    """
    rng = np.random.default_rng(seed)
    y = y_train.values if hasattr(y_train, "values") else np.asarray(y_train)
    X = X_train.values if hasattr(X_train, "values") else np.asarray(X_train)
    idx_min = np.where(y == 1)[0]
    idx_maj = np.where(y == 0)[0]
    idx_maj_down = rng.choice(idx_maj, size=len(idx_min), replace=False)
    keep = np.concatenate([idx_min, idx_maj_down])
    rng.shuffle(keep)
    X_res, y_res = X[keep], y[keep]
    if hasattr(X_train, "columns"):
        X_res = pd.DataFrame(X_res, columns=list(X_train.columns))
    return X_res, y_res


def fit_baseline(X_train, y_train) -> Pipeline:
    return make_lr().fit(X_train, y_train)


def fit_class_weight(X_train, y_train) -> Pipeline:
    return make_lr(class_weight="balanced").fit(X_train, y_train)


def fit_smote(X_train, y_train) -> Pipeline:
    X_res, y_res = smote_resample(X_train, y_train)
    return make_lr().fit(X_res, y_res)


def fit_undersample(X_train, y_train) -> Pipeline:
    X_res, y_res = undersample(X_train, y_train)
    return make_lr().fit(X_res, y_res)


def predict_proba(model, X) -> np.ndarray:
    """Probability of the positive (fraud) class."""
    return model.predict_proba(X)[:, 1]


def predict_at(proba: np.ndarray, threshold: float) -> np.ndarray:
    return (np.asarray(proba) >= threshold).astype(int)


def tune_threshold(y_true, y_proba) -> dict:
    """Best-F1 threshold, scanned on the given (validation) labels.

    Returns the threshold and the precision/recall/F1 it gives.
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    best = {"threshold": 0.5, "f1": -1.0, "precision": 0.0, "recall": 0.0}
    for t in np.arange(0.01, 1.0, 0.01):
        pred = predict_at(y_proba, float(t))
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best["f1"]:
            best = {
                "threshold": float(t),
                "f1": float(f1),
                "precision": float(precision_score(y_true, pred, zero_division=0)),
                "recall": float(recall_score(y_true, pred, zero_division=0)),
            }
    return best


def recall_threshold(y_true, y_proba, target: float = 0.90) -> dict:
    """Threshold that hits recall >= target with the best precision.

    The operating point for a business that would rather raise a false
    alarm than miss a fraud. Among thresholds reaching the recall
    target, picks the one with the highest precision (fewest false
    alarms for the same catch rate). Returns None values if the target
    cannot be reached.
    """
    from sklearn.metrics import precision_score, recall_score

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    best = None
    for t in np.arange(0.01, 1.0, 0.01):
        pred = predict_at(y_proba, float(t))
        rec = recall_score(y_true, pred, zero_division=0)
        if rec >= target:
            prec = precision_score(y_true, pred, zero_division=0)
            if best is None or prec > best["precision"]:
                best = {"threshold": float(t), "precision": float(prec),
                        "recall": float(rec)}
    if best is None:
        return {"threshold": None, "precision": 0.0, "recall": 0.0}
    return best
