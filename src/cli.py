"""Command-line interface.

Subcommands (all verified):
  evaluate / run-all   train every approach, tune thresholds on the
                       validation set, score on the untouched test set,
                       run the 5% label-noise check, save results and
                       print the comparison table.
  compare              print the comparison table from saved results
                       (runs evaluate first if no results exist).
  plot                 regenerate all plots from saved results.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

# Allow running as `python -m src.cli` from the project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import data as data_mod
from src import evaluate as ev
from src import models as mod
from src import plots as plt_mod

APPROACHES = ["baseline", "class_weight", "smote", "undersample"]


def _root(args) -> str:
    return args.root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _results_dir(root: str) -> str:
    return os.path.join(root, "results")


def _fit_all(split, y_train_noisy=None):
    """Train every approach. y_train_noisy optionally replaces labels."""
    ytr = split.y_train if y_train_noisy is None else y_train_noisy
    return {
        "baseline": mod.fit_baseline(split.X_train, ytr),
        "class_weight": mod.fit_class_weight(split.X_train, ytr),
        "smote": mod.fit_smote(split.X_train, ytr),
        "undersample": mod.fit_undersample(split.X_train, ytr),
    }


def _score(models, X, y_true, thresholds=None):
    """Score fitted models on a set; thresholds optionally override 0.5."""
    out = {}
    for name, model in models.items():
        proba = mod.predict_proba(model, X)
        t = 0.5 if thresholds is None else thresholds.get(name, 0.5)
        pred = mod.predict_at(proba, t)
        out[name] = {
            "metrics": ev.compute_metrics(y_true, pred, proba),
            "y_true": np.asarray(y_true),
            "y_proba": proba,
            "threshold": float(t),
        }
    return out


def run_evaluate(root: str) -> dict:
    df = data_mod.load_working_set(root)
    split = data_mod.split(df)
    prev = data_mod.prevalence(df[data_mod.TARGET])
    test_prev = data_mod.prevalence(split.y_test)

    # --- train on clean labels ---
    models = _fit_all(split)

    # --- score the four approaches at the default 0.5 threshold ---
    # Apples to apples: the only difference between these rows is the
    # imbalance handling, not the decision threshold.
    results = _score(models, split.X_test, split.y_test,
                     thresholds={name: 0.5 for name in APPROACHES})

    # --- threshold tuning as its own approach, on the class-weighted
    # model, tuned on the VALIDATION set (never the test set) ---
    val_proba_cw = mod.predict_proba(models["class_weight"], split.X_val)
    best_f1 = mod.tune_threshold(split.y_val, val_proba_cw)
    cw_proba = results["class_weight"]["y_proba"]
    pred_tuned = mod.predict_at(cw_proba, best_f1["threshold"])
    results["threshold_tuned"] = {
        "metrics": ev.compute_metrics(split.y_test, pred_tuned, cw_proba),
        "y_true": np.asarray(split.y_test),
        "y_proba": cw_proba,
        "threshold": float(best_f1["threshold"]),
    }
    # --- recall >= 0.90 operating point (same model, lower threshold) ---
    recall90 = mod.recall_threshold(split.y_val, val_proba_cw, target=0.90)
    if recall90["threshold"] is not None:
        pred90 = mod.predict_at(cw_proba, recall90["threshold"])
        results["recall90"] = {
            "metrics": ev.compute_metrics(split.y_test, pred90, cw_proba),
            "y_true": np.asarray(split.y_test),
            "y_proba": cw_proba,
            "threshold": float(recall90["threshold"]),
        }
    else:
        results["recall90"] = dict(results["threshold_tuned"])

    # --- dummy all-negative classifier (the misleading-accuracy demo) ---
    dummy = ev.dummy_metrics(split.y_test)

    # --- noise robustness: 5% flipped training labels, retrain, re-score
    # at the SAME 0.5 threshold, so the only difference is the noise. ---
    y_noisy = ev.add_label_noise(split.y_train, frac=0.05, seed=42)
    noisy_models = _fit_all(split, y_train_noisy=y_noisy)
    noisy_results = _score(noisy_models, split.X_test, split.y_test,
                           thresholds={name: 0.5 for name in APPROACHES})
    noise = {}
    for name in APPROACHES:
        noise[name] = {
            "f1_clean": results[name]["metrics"]["f1"],
            "f1_noisy": noisy_results[name]["metrics"]["f1"],
            "pr_auc_clean": results[name]["metrics"]["pr_auc"],
            "pr_auc_noisy": noisy_results[name]["metrics"]["pr_auc"],
        }

    # --- per-1000-transactions framing for the recommendation ---
    per1000 = {name: ev.per_1000_transactions(r["metrics"], test_prev)
               for name, r in results.items()}

    summary = {
        "dataset": {
            "rows": int(len(df)),
            "frauds": int(df[data_mod.TARGET].sum()),
            "prevalence": prev,
            "train_rows": int(len(split.X_train)),
            "val_rows": int(len(split.X_val)),
            "test_rows": int(len(split.X_test)),
            "test_prevalence": test_prev,
            "test_frauds": int(split.y_test.sum()),
        },
        "thresholds_on_validation": {
            "best_f1": best_f1,
            "recall90": recall90,
        },
        "recall90_threshold": recall90,
        "dummy_all_negative": dummy,
        "approaches": {
            name: {"metrics": r["metrics"], "threshold": r["threshold"],
                   "per_1000": per1000[name]}
            for name, r in results.items()
        },
        "noise_robustness_5pct_label_flip": noise,
    }

    # --- save ---
    os.makedirs(_results_dir(root), exist_ok=True)
    os.makedirs(os.path.join(_results_dir(root), "plots"), exist_ok=True)
    with open(os.path.join(_results_dir(root), "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(_results_dir(root), "metrics.md"), "w") as f:
        f.write(_markdown_table(summary))
    np.savez_compressed(
        os.path.join(_results_dir(root), "probas.npz"),
        y_true=np.asarray(split.y_test),
        y_val=np.asarray(split.y_val),
        val_proba_cw=val_proba_cw,
        **{f"proba_{k}": v["y_proba"] for k, v in results.items()},
        **{f"pred_{k}": mod.predict_at(v["y_proba"], v["threshold"])
           for k, v in results.items()},
    )
    make_plots(root, results, test_prev)
    return summary


def make_plots(root: str, results=None, test_prev=None) -> None:
    plots_dir = os.path.join(_results_dir(root), "plots")
    os.makedirs(plots_dir, exist_ok=True)
    if results is None:
        data = np.load(os.path.join(_results_dir(root), "probas.npz"))
        y_true = data["y_true"]
        with open(os.path.join(_results_dir(root), "metrics.json")) as f:
            summary = json.load(f)
        test_prev = summary["dataset"]["test_prevalence"]
        results = {}
        for name in APPROACHES + ["threshold_tuned", "recall90"]:
            if f"proba_{name}" not in data:
                continue
            proba = data[f"proba_{name}"]
            pred = data[f"pred_{name}"]
            results[name] = {
                "y_true": y_true, "y_proba": proba,
                "metrics": ev.compute_metrics(y_true, pred, proba),
                "threshold": summary["approaches"][name]["threshold"],
            }
    plot_results = {k: v for k, v in results.items() if k in APPROACHES}
    plt_mod.plot_pr_curves(plot_results, test_prev,
                           os.path.join(plots_dir, "pr_curves.png"))
    # Threshold trade-off uses the VALIDATION probas of the
    # class-weighted model — the same numbers the thresholds were tuned
    # on, so the plot shows the tuning decision, not test-set peeking.
    data = np.load(os.path.join(_results_dir(root), "probas.npz"))
    plt_mod.plot_threshold_tradeoff(
        data["y_val"], data["val_proba_cw"],
        os.path.join(plots_dir, "threshold_tradeoff.png"))
    rec = results.get("recall90", results["class_weight"])
    plt_mod.plot_confusion(
        rec["y_true"],
        mod.predict_at(rec["y_proba"], rec["threshold"]),
        os.path.join(plots_dir, "confusion_recommended.png"),
        title="Recommended operating point")


def _markdown_table(summary: dict) -> str:
    lines = ["# Results (test set, natural prevalence)\n",
             "| Approach | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, a in summary["approaches"].items():
        m = a["metrics"]
        lines.append(
            f"| {name} | {a['threshold']:.2f} | {m['accuracy']:.4f} | "
            f"{m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | "
            f"{m['roc_auc']:.4f} | {m['pr_auc']:.4f} |")
    d = summary["dummy_all_negative"]
    lines.append(
        f"| dummy (all negative) | — | {d['accuracy']:.4f} | {d['precision']:.4f} "
        f"| {d['recall']:.4f} | {d['f1']:.4f} | {d['roc_auc']:.4f} | {d['pr_auc']:.4f} |")
    return "\n".join(lines) + "\n"


def print_table(summary: dict) -> None:
    d = summary["dataset"]
    print(f"\nDataset: {d['rows']:,} rows, {d['frauds']:,} frauds "
          f"({d['prevalence']:.2%} positives); "
          f"test: {d['test_rows']:,} rows, {d['test_frauds']} frauds "
          f"({d['test_prevalence']:.2%})")
    print(f"Dummy all-negative classifier: accuracy "
          f"{summary['dummy_all_negative']['accuracy']:.4f}, recall "
          f"{summary['dummy_all_negative']['recall']:.2f}, "
          f"F1 {summary['dummy_all_negative']['f1']:.2f}")
    print("\nApproach            Thr   Acc    Prec   Rec    F1     ROC-AUC  PR-AUC")
    for name, a in summary["approaches"].items():
        m = a["metrics"]
        print(f"{name:<18} {a['threshold']:.2f}  {m['accuracy']:.4f} "
              f"{m['precision']:.4f} {m['recall']:.4f} {m['f1']:.4f} "
              f"{m['roc_auc']:.4f}   {m['pr_auc']:.4f}")
    r90 = summary["recall90_threshold"]
    print(f"\nRecall>=0.90 operating point (class-weighted model): "
          f"threshold {r90['threshold']}, precision {r90['precision']:.4f}, "
          f"recall {r90['recall']:.4f}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="imbalanced")
    parser.add_argument("--root", default=None,
                        help="project root (default: detected)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("evaluate", help="run the full comparison")
    sub.add_parser("run-all", help="evaluate + plots (same as evaluate)")
    sub.add_parser("compare", help="print the saved comparison table")
    sub.add_parser("plot", help="regenerate plots from saved results")
    args = parser.parse_args(argv)
    root = _root(args)

    if args.cmd in ("evaluate", "run-all"):
        summary = run_evaluate(root)
        print_table(summary)
        print(f"\nSaved results/metrics.json, results/metrics.md, "
              f"results/probas.npz and plots in results/plots/.")
    elif args.cmd == "compare":
        path = os.path.join(_results_dir(root), "metrics.json")
        if not os.path.exists(path):
            summary = run_evaluate(root)
        else:
            with open(path) as f:
                summary = json.load(f)
        print_table(summary)
    elif args.cmd == "plot":
        path = os.path.join(_results_dir(root), "metrics.json")
        if not os.path.exists(path):
            run_evaluate(root)
        else:
            make_plots(root)
        print("Plots regenerated in results/plots/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
