"""Leakage-safe comparison of Feature Sets 0 / A / B (Member 2, M2-08).

A plain logistic-regression baseline is used ONLY to compare feature sets; final
model selection and tuning belong to the modelling stage. Everything that learns
from data (PCA, K-Means, imputer, scaler, feature selection, classifier) is
refitted inside each training fold. The held-out test participants are never
passed to this module.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.data_io import assert_no_target_leakage
from src.features.select_features import LeakageSafeSelector


def stratified_split(y: pd.Series, test_size: float = 0.2, seed: int = 42):
  """Reproducible stratified train/test split of participant ids (SEQN)."""
  train_ids, test_ids = train_test_split(
      y.index, test_size=test_size, stratify=y, random_state=seed)
  assert set(train_ids).isdisjoint(test_ids)
  return train_ids, test_ids


def make_pipeline(selector_kwargs: dict | None = None, seed: int = 42) -> Pipeline:
  return Pipeline([
      ("select", LeakageSafeSelector(random_state=seed, **(selector_kwargs or {}))),
      ("impute", SimpleImputer(strategy="median")),
      ("scale", StandardScaler()),
      ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
  ])


def cross_validate_set(
    build_X: Callable[[pd.Index], pd.DataFrame],
    y: pd.Series,
    train_ids: pd.Index,
    n_splits: int = 5,
    seed: int = 42,
    selector_kwargs: dict | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
  """Stratified k-fold CV on TRAINING participants.

  ``build_X(fit_ids)`` returns the feature matrix for all participants, with any
  learned component (PCA, K-Means) fitted on ``fit_ids`` only. For fixed
  feature sets it simply ignores ``fit_ids``.

  Returns (per-fold metrics, how often each feature was selected across folds).
  """
  y_tr = y.loc[train_ids]
  skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
  rows, picked = [], []
  for fold, (tr, va) in enumerate(skf.split(np.zeros(len(y_tr)), y_tr), start=1):
    tr_ids, va_ids = y_tr.index[tr], y_tr.index[va]
    X = build_X(tr_ids)
    assert_no_target_leakage(X.columns)
    pipe = make_pipeline(selector_kwargs, seed).fit(X.loc[tr_ids], y_tr.loc[tr_ids])
    proba = pipe.predict_proba(X.loc[va_ids])[:, 1]
    rows.append({
        "fold": fold,
        "roc_auc": roc_auc_score(y_tr.loc[va_ids], proba),
        "pr_auc": average_precision_score(y_tr.loc[va_ids], proba),
        "n_selected": len(pipe.named_steps["select"].selected_),
        "n_candidates": X.shape[1],
    })
    picked += pipe.named_steps["select"].selected_
  return pd.DataFrame(rows), pd.Series(picked).value_counts().rename("folds_selected")


def compare_feature_sets(
    builders: dict[str, Callable[[pd.Index], pd.DataFrame]],
    y: pd.Series,
    train_ids: pd.Index,
    **kwargs,
) -> tuple[pd.DataFrame, dict]:
  """Run cross_validate_set for every set with identical folds and participants."""
  summary, folds, freq = [], {}, {}
  for name, build_X in builders.items():
    f, p = cross_validate_set(build_X, y, train_ids, **kwargs)
    folds[name], freq[name] = f, p
    summary.append({
        "feature_set": name,
        "n_participants": len(train_ids),
        "roc_auc_mean": f["roc_auc"].mean(), "roc_auc_std": f["roc_auc"].std(),
        "pr_auc_mean": f["pr_auc"].mean(),
        "avg_n_selected": f["n_selected"].mean(),
        "n_candidates": int(f["n_candidates"].iloc[0]),
    })
  return pd.DataFrame(summary).set_index("feature_set"), {"folds": folds, "freq": freq}
