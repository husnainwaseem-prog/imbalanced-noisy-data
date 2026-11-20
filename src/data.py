"""Data loading and splitting for the fraud-imbalance study.

Dataset: Kaggle credit-card fraud (284,807 European card transactions,
492 frauds = 0.17% positives). The full file is ~98 MB, so the project
ships a working set instead: ALL 492 fraud cases + a random sample of
100,000 legitimate transactions (seed 42) -> 100,492 rows, 0.49%
positives. Still far under 2%, and small enough to run anywhere.

Golden rule, enforced here: resampling NEVER touches the validation or
test sets. The test set keeps its natural prevalence, so every metric
reported is what the business would actually see.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET = "Class"
RANDOM_STATE = 42
N_LEGIT = 100_000


@dataclass
class Split:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def load_full(path: str) -> pd.DataFrame:
    """Load the full Kaggle credit-card fraud CSV."""
    return pd.read_csv(path)


def build_working_set(df: pd.DataFrame, n_legit: int = N_LEGIT,
                      seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Keep every fraud case + a random sample of legitimate rows.

    Returns a shuffled working set. Fraud is rare, so throwing away
    fraud cases would waste the scarcest resource we have.
    """
    fraud = df[df[TARGET] == 1]
    legit = df[df[TARGET] == 0]
    legit_sample = legit.sample(n=min(n_legit, len(legit)),
                                random_state=seed)
    working = pd.concat([fraud, legit_sample], ignore_index=True)
    return working.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def prevalence(y) -> float:
    """Fraction of positive (fraud) labels."""
    y = np.asarray(y)
    return float(y.mean())


def split(df: pd.DataFrame, seed: int = RANDOM_STATE) -> Split:
    """Stratified 60/20/20 train/validation/test split.

    Stratified, so each split keeps the same fraud rate as the full
    working set. Splitting happens BEFORE any resampling: resampling
    is applied to the training fold only, downstream.
    """
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.40, stratify=y, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.50, stratify=y_rest,
        random_state=seed)
    return Split(X_train, X_val, X_test, y_train, y_val, y_test)


def load_working_set(root: str) -> pd.DataFrame:
    """Load the shipped working set (data/creditcard.csv).

    If it is missing but the full download is present, build it.
    """
    path = os.path.join(root, "data", "creditcard.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    full_path = os.path.join(root, "data", "creditcard_full.csv")
    if not os.path.exists(full_path):
        raise FileNotFoundError(
            f"No data file found. Expected {path} (or {full_path} to "
            "build it from).")
    working = build_working_set(load_full(full_path))
    working.to_csv(path, index=False)
    return working
