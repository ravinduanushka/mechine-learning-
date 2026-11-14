"""Feature selection pipeline for the cardiometabolic risk model (Member 2, M2-07).

Stages (all fitted on TRAINING data only):
  1. correlation filter  - drop one of each pair with |r| > 0.9
  2. mutual information  - rank, keep the top-k informative features
  3. RFE                 - recursive elimination with logistic regression
  4. L1 logistic         - keep features with non-zero L1 weight
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import RFE, SelectFromModel, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def correlation_filter(x: pd.DataFrame, threshold: float = 0.9) -> list[str]:
  """Return columns kept after dropping one of each pair with |r| > threshold."""
  corr = x.corr().abs()
  upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
  to_drop = [c for c in upper.columns if (upper[c] > threshold).any()]
  return [c for c in x.columns if c not in to_drop]


def apply_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    corr_threshold: float = 0.9,
    mi_top_k: int = 30,
    rfe_n_features: int = 15,
    l1_C: float = 0.1,
    random_state: int = 42,
    return_report: bool = False,
    verbose: bool = True,
):
  """Correlation filter -> Mutual Information -> RFE -> L1 selection.

  Returns the list of selected feature names (or ``(names, report)`` when
  ``return_report`` is True). ``report`` holds the MI table and the features
  surviving each stage. If L1 would remove everything, the RFE set is returned
  and ``report["l1_fallback"]`` is True.
  """
  say = print if verbose else (lambda *a, **k: None)

  say(f"1. Correlation filter (|r| > {corr_threshold})...")
  kept = correlation_filter(X_train, threshold=corr_threshold)
  say(f"   kept {len(kept)} of {X_train.shape[1]} features")

  imputer = SimpleImputer(strategy="median")
  X_imp = imputer.fit_transform(X_train[kept])
  X_std = StandardScaler().fit_transform(X_imp)  # RFE / L1 are scale-sensitive

  say("2. Mutual Information ranking...")
  mi = mutual_info_classif(X_imp, y_train, random_state=random_state)
  mi_df = pd.DataFrame({"Feature": kept, "MI_Score": mi}).sort_values(
      "MI_Score", ascending=False, ignore_index=True)
  mi_keep = mi_df.loc[mi_df["MI_Score"] > 0, "Feature"].head(mi_top_k).tolist()
  if not mi_keep:  # all MI == 0: nothing informative, keep ranking order anyway
    mi_keep = mi_df["Feature"].head(mi_top_k).tolist()
  say(mi_df.head(10).to_string(index=False))
  say(f"   kept {len(mi_keep)} features with MI > 0 (top {mi_top_k})")

  say("3. Recursive Feature Elimination...")
  pos = [kept.index(c) for c in mi_keep]
  n_rfe = max(1, min(rfe_n_features, len(mi_keep)))
  rfe = RFE(LogisticRegression(max_iter=1000, random_state=random_state),
            n_features_to_select=n_rfe).fit(X_std[:, pos], y_train)
  rfe_keep = [c for c, s in zip(mi_keep, rfe.support_) if s]
  say(f"   kept {len(rfe_keep)} features")

  say("4. L1 regularised logistic selection...")
  pos = [kept.index(c) for c in rfe_keep]
  l1 = SelectFromModel(
      LogisticRegression(l1_ratio=1.0, solver="liblinear", C=l1_C,
                         random_state=random_state)
  ).fit(X_std[:, pos], y_train)
  selected = [c for c, s in zip(rfe_keep, l1.get_support()) if s]
  fallback = not selected
  if fallback:
    warnings.warn("L1 removed every feature; returning the RFE set instead.")
    selected = rfe_keep
  say(f"Selected {len(selected)} features.")

  if return_report:
    report = {"after_correlation": kept, "mi_table": mi_df, "after_mi": mi_keep,
              "after_rfe": rfe_keep, "selected": selected, "l1_fallback": fallback}
    return selected, report
  return selected


class LeakageSafeSelector(BaseEstimator, TransformerMixin):
  """sklearn transformer that runs apply_feature_selection inside ``fit``.

  Placed in a Pipeline, it is re-fitted on each CV training fold only, so the
  validation fold never influences which features are chosen.
  """

  def __init__(self, corr_threshold=0.9, mi_top_k=30, rfe_n_features=15,
               l1_C=0.1, random_state=42):
    self.corr_threshold = corr_threshold
    self.mi_top_k = mi_top_k
    self.rfe_n_features = rfe_n_features
    self.l1_C = l1_C
    self.random_state = random_state

  def fit(self, X, y):
    X = pd.DataFrame(X)
    self.selected_ = apply_feature_selection(
        X, pd.Series(np.asarray(y)), self.corr_threshold, self.mi_top_k,
        self.rfe_n_features, self.l1_C, self.random_state, verbose=False)
    self.columns_ = list(X.columns)
    return self

  def transform(self, X):
    return pd.DataFrame(X)[self.selected_]
