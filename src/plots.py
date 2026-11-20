"""Plots: PR curves, threshold trade-off, confusion matrix."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from .evaluate import pr_curve_data


def plot_pr_curves(results: dict, prevalence: float, path: str) -> None:
    """One PR curve per approach, plus the no-skill prevalence line."""
    plt.figure(figsize=(8, 6))
    for name, r in results.items():
        precision, recall = pr_curve_data(r["y_true"], r["y_proba"])
        plt.plot(recall, precision, lw=2,
                 label=f"{name} (PR-AUC {r['metrics']['pr_auc']:.3f})")
    plt.axhline(prevalence, color="grey", ls="--", lw=1.5,
                label=f"No-skill (prevalence {prevalence:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curves (test set, natural prevalence)")
    plt.legend(loc="lower left", fontsize=9)
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_threshold_tradeoff(y_true, y_proba, path: str) -> None:
    """Precision / recall / F1 against the decision threshold."""
    from sklearn.metrics import f1_score, precision_score, recall_score

    thresholds = np.arange(0.01, 1.0, 0.01)
    precisions, recalls, f1s = [], [], []
    for t in thresholds:
        pred = (np.asarray(y_proba) >= t).astype(int)
        precisions.append(precision_score(y_true, pred, zero_division=0))
        recalls.append(recall_score(y_true, pred, zero_division=0))
        f1s.append(f1_score(y_true, pred, zero_division=0))
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, precisions, label="Precision", lw=2)
    plt.plot(thresholds, recalls, label="Recall", lw=2)
    plt.plot(thresholds, f1s, label="F1", lw=2)
    plt.xlabel("Decision threshold")
    plt.ylabel("Score")
    plt.title("Precision / Recall / F1 vs threshold (validation set)")
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_confusion(y_true, y_pred, path: str, title: str = "Confusion matrix") -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted legit", "Predicted fraud"])
    ax.set_yticklabels(["Actual legit", "Actual fraud"])
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
