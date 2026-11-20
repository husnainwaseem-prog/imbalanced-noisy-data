"""Evaluation: metrics, the misleading-accuracy demo, and framing.

Every number is computed on the untouched test set (natural prevalence).
The dummy all-negative classifier is the headline: it scores ~99.5%
accuracy while catching zero fraud, which is the whole point of the
project.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             precision_recall_curve, recall_score,
                             roc_auc_score)


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """All metrics for one approach on the test set."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def dummy_metrics(y_true) -> dict:
    """The all-negative classifier: predict 'not fraud' for everything.

    Accuracy looks great (it equals the share of legitimate
    transactions). Recall and F1 are exactly 0: it catches no fraud.
    """
    y_true = np.asarray(y_true)
    y_pred = np.zeros_like(y_true)
    y_proba = np.zeros(len(y_true), dtype=float)
    m = compute_metrics(y_true, y_pred, y_proba)
    m["roc_auc"] = 0.5  # a constant score has no ranking power
    m["pr_auc"] = float(y_true.mean())  # no better than prevalence
    return m


def pr_curve_data(y_true, y_proba):
    """Precision/recall points for a PR curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    return precision, recall


def per_1000_transactions(metrics: dict, prevalence: float) -> dict:
    """What one approach means per 1,000 transactions.

    Turns abstract rates into business numbers: out of 1,000
    transactions, ~1000*prevalence are fraud. Recall says how many of
    those are caught; precision says how many alarms are real.
    """
    frauds = 1000 * prevalence
    caught = frauds * metrics["recall"]
    missed = frauds - caught
    alarms = caught / metrics["precision"] if metrics["precision"] > 0 else 0.0
    false_alarms = alarms - caught
    return {
        "frauds_per_1000": round(float(frauds), 1),
        "caught_per_1000": round(float(caught), 1),
        "missed_per_1000": round(float(missed), 1),
        "false_alarms_per_1000": round(float(false_alarms), 1),
    }


def add_label_noise(y_train, frac: float = 0.05,
                    seed: int = 42) -> np.ndarray:
    """Flip a fraction of training labels at random (simulated noise).

    Real fraud labels are noisy: chargebacks arrive late, some fraud is
    never reported, and some legitimate transactions get flagged by
    mistake. This simulates that and tests which approach survives it.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y_train).copy()
    n_flip = int(round(frac * len(y)))
    idx = rng.choice(len(y), size=n_flip, replace=False)
    y[idx] = 1 - y[idx]
    return y
