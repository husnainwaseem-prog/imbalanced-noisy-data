"""Tests for the imbalanced-fraud study.

Covers: stratified splits, resampling touching train only, metric
functions on known inputs, threshold tuning on toy cases, PR-AUC
beating the dummy baseline, label-noise behaviour, and the CLI.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src import data as data_mod
from src import evaluate as ev
from src import models as mod
from src.cli import main as cli_main

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(os.path.join(ROOT, "data", "creditcard.csv"))


@pytest.fixture(scope="module")
def split(df):
    return data_mod.split(df)


# --- splits -----------------------------------------------------------------

def test_split_stratified_prevalence(df, split):
    overall = data_mod.prevalence(df["Class"])
    for y in (split.y_train, split.y_val, split.y_test):
        assert abs(data_mod.prevalence(y) - overall) < 0.001


def test_split_sizes(df, split):
    n = len(df)
    assert abs(len(split.y_train) / n - 0.60) < 0.01
    assert abs(len(split.y_val) / n - 0.20) < 0.01
    assert abs(len(split.y_test) / n - 0.20) < 0.01


def test_split_disjoint(df, split):
    tr = set(split.X_train.index)
    va = set(split.X_val.index)
    te = set(split.X_test.index)
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert len(tr | va | te) == len(df)


def test_working_set_prevalence_under_2pct(df):
    assert data_mod.prevalence(df["Class"]) < 0.02
    assert data_mod.prevalence(df["Class"]) > 0.0


def test_build_working_set_keeps_all_frauds():
    rng = np.random.default_rng(0)
    fake = pd.DataFrame({
        "V1": rng.normal(size=1010), "Amount": rng.uniform(1, 100, 1010),
        "Class": [1] * 10 + [0] * 1000,
    })
    working = data_mod.build_working_set(fake, n_legit=1000)
    assert int(working["Class"].sum()) == 10  # every fraud kept
    assert len(working) == 1010
    assert data_mod.prevalence(working["Class"]) < 0.02


# --- resampling touches train only ------------------------------------------

def test_smote_balances_train_only(split):
    X_res, y_res = mod.smote_resample(split.X_train, split.y_train)
    assert int((y_res == 1).sum()) == int((y_res == 0).sum())
    assert len(y_res) > len(split.y_train)
    # the test set is never passed to the resampler: its prevalence is
    # still the natural one
    assert abs(data_mod.prevalence(split.y_test) - 0.0049) < 0.001


def test_undersample_balances_train_only(split):
    X_res, y_res = mod.undersample(split.X_train, split.y_train)
    assert int((y_res == 1).sum()) == int((y_res == 0).sum())
    assert len(y_res) == 2 * int(split.y_train.sum())


# --- metrics on known inputs -------------------------------------------------

def test_metrics_perfect_predictions():
    y = np.array([0, 0, 1, 1])
    m = ev.compute_metrics(y, y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["accuracy"] == 1.0 and m["precision"] == 1.0
    assert m["recall"] == 1.0 and m["f1"] == 1.0
    assert m["roc_auc"] == 1.0 and m["pr_auc"] == 1.0
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (2, 0, 2, 0)


def test_metrics_all_negative_dummy():
    y = np.array([0] * 99 + [1])
    m = ev.dummy_metrics(y)
    assert m["accuracy"] == pytest.approx(0.99)
    assert m["recall"] == 0.0 and m["f1"] == 0.0 and m["precision"] == 0.0
    assert m["pr_auc"] == pytest.approx(0.01)


def test_pr_auc_model_beats_dummy():
    X, y = make_classification(n_samples=2000, n_features=10,
                               n_informative=5, weights=[0.98, 0.02],
                               random_state=42)
    model = mod.fit_baseline(pd.DataFrame(X), pd.Series(y))
    proba = mod.predict_proba(model, pd.DataFrame(X))
    pred = mod.predict_at(proba, 0.5)
    m = ev.compute_metrics(y, pred, proba)
    assert m["pr_auc"] > data_mod.prevalence(y)  # better than no-skill
    assert m["pr_auc"] > 0.1


# --- threshold tuning on toy cases -------------------------------------------

def test_tune_threshold_picks_best_f1():
    y = np.array([0, 0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    best = mod.tune_threshold(y, proba)
    assert best["f1"] == pytest.approx(1.0)
    assert 0.3 < best["threshold"] <= 0.8


def test_recall_threshold_achieves_target():
    y = np.array([0, 0, 1, 1, 1])
    proba = np.array([0.1, 0.2, 0.5, 0.7, 0.9])
    out = mod.recall_threshold(y, proba, target=0.90)
    assert out["recall"] >= 0.90
    assert out["precision"] == pytest.approx(1.0)
    assert abs(out["threshold"] - 0.21) < 0.011


def test_recall_threshold_unreachable_returns_none():
    # the only positive scores 0.0, so no threshold reaches recall 0.9
    y = np.array([0, 1])
    proba = np.array([0.5, 0.0])
    out = mod.recall_threshold(y, proba, target=0.90)
    assert out["threshold"] is None


# --- label noise --------------------------------------------------------------

def test_add_label_noise_flips_fraction():
    y = np.zeros(1000, dtype=int)
    noisy = ev.add_label_noise(y, frac=0.05, seed=42)
    assert int((noisy != y).sum()) == 50


def test_per_1000_framing():
    m = {"recall": 0.9, "precision": 0.2}
    out = ev.per_1000_transactions(m, prevalence=0.005)
    assert out["frauds_per_1000"] == pytest.approx(5.0)
    assert out["caught_per_1000"] == pytest.approx(4.5)
    assert out["missed_per_1000"] == pytest.approx(0.5)
    assert out["false_alarms_per_1000"] == pytest.approx(18.0)


# --- CLI ----------------------------------------------------------------------

def _synthetic_root(tmp_path):
    """A small fraud-like CSV with the real column names."""
    rng = np.random.default_rng(7)
    n, p = 5000, 25
    X, y = make_classification(n_samples=n, n_features=28, n_informative=8,
                               weights=[0.995, 0.005], random_state=7)
    cols = {f"V{i+1}": X[:, i] for i in range(28)}
    cols["Time"] = rng.uniform(0, 100000, n)
    cols["Amount"] = rng.uniform(1, 200, n)
    cols["Class"] = y
    root = tmp_path / "proj"
    (root / "data").mkdir(parents=True)
    pd.DataFrame(cols).to_csv(root / "data" / "creditcard.csv", index=False)
    return str(root)


def test_cli_evaluate_runs(tmp_path):
    root = _synthetic_root(tmp_path)
    assert cli_main(["--root", root, "evaluate"]) == 0
    summary = json.load(open(os.path.join(root, "results", "metrics.json")))
    assert set(summary["approaches"]) == {
        "baseline", "class_weight", "smote", "undersample",
        "threshold_tuned", "recall90"}
    assert summary["dummy_all_negative"]["recall"] == 0.0
    assert summary["dataset"]["prevalence"] < 0.02


def test_cli_compare_and_plot_run(tmp_path):
    root = _synthetic_root(tmp_path)
    assert cli_main(["--root", root, "evaluate"]) == 0
    assert cli_main(["--root", root, "compare"]) == 0
    assert cli_main(["--root", root, "plot"]) == 0
    for name in ("pr_curves.png", "threshold_tradeoff.png",
                 "confusion_recommended.png"):
        path = os.path.join(root, "results", "plots", name)
        assert os.path.exists(path) and os.path.getsize(path) > 1000
