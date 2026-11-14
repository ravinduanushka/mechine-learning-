"""Leakage-safe feature selection for the cardiometabolic risk model (Member 2).

Pipeline, fitted on TRAINING data only:
  1. correlation filter  - drop one feature of every pair with |Pearson r| > 0.9
  2. mutual information  - rank with ``mutual_info_classif`` and keep the top-k
  3. L1 logistic         - keep features with a non-zero L1-penalised coefficient

Call ``apply_feature_selection`` on the training split only, then apply the
returned column list to the validation / test data. Inside cross-validation,
call it on each training fold.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

__all__ = [
    "correlation_filter",
    "mutual_information_ranking",
    "l1_selection",
    "apply_feature_selection",
]


def _validate_inputs(X: pd.DataFrame, y: pd.Series) -> pd.Series:
  """Check types, alignment and content; return y aligned to X's index."""
  if not isinstance(X, pd.DataFrame):
    raise TypeError(f"X_train must be a pandas DataFrame, got {type(X).__name__}")
  if X.empty or X.shape[1] == 0:
    raise ValueError("X_train is empty")
  if not isinstance(y, (pd.Series, np.ndarray, list)):
    raise TypeError(f"y_train must be a Series, array or list, got {type(y).__name__}")
  if not isinstance(y, pd.Series):
    y = pd.Series(np.asarray(y), index=X.index)
  if len(y) != len(X):
    raise ValueError(f"X_train has {len(X)} rows but y_train has {len(y)}")
  if not y.index.equals(X.index):
    if set(y.index) != set(X.index):
      raise ValueError("X_train and y_train indices do not match")
    y = y.loc[X.index]
  if y.isna().any():
    raise ValueError("y_train contains missing values")
  if y.nunique() < 2:
    raise ValueError("y_train must contain at least two classes")
  non_numeric = X.columns[~X.dtypes.map(pd.api.types.is_numeric_dtype)].tolist()
  if non_numeric:
    raise ValueError(f"X_train has non-numeric columns (encode them first): {non_numeric}")
  if X.columns.duplicated().any():
    raise ValueError("X_train has duplicate column names")
  if np.isinf(X.to_numpy(dtype=float)).any():
    raise ValueError("X_train contains infinite values")
  return y


def correlation_filter(X: pd.DataFrame, threshold: float = 0.9) -> list[str]:
  """Columns kept after dropping one of each pair with |Pearson r| > threshold.

  The first column of a correlated pair (in column order) is kept. Constant
  columns have undefined correlation and are dropped with a warning.
  """
  if not 0 < threshold <= 1:
    raise ValueError(f"threshold must be in (0, 1], got {threshold}")
  constant = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
  if constant:
    warnings.warn(f"Dropping constant columns: {constant}")
  work = X.drop(columns=constant)
  corr = work.corr(method="pearson").abs()
  upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
  to_drop = {col for col in upper.columns if (upper[col] > threshold).any()}
  return [c for c in work.columns if c not in to_drop]


def mutual_information_ranking(
    X: pd.DataFrame, y: pd.Series, random_state: int = 42
) -> pd.DataFrame:
  """Mutual information of every column with the class label, best first.

  Missing values are median-imputed (statistics from ``X`` only).
  """
  X_imp = SimpleImputer(strategy="median").fit_transform(X)
  scores = mutual_info_classif(X_imp, y, random_state=random_state)
  return (pd.DataFrame({"feature": X.columns, "mi_score": scores})
          .sort_values("mi_score", ascending=False, ignore_index=True))


def l1_selection(
    X: pd.DataFrame, y: pd.Series, C: float = 0.1, random_state: int = 42
) -> list[str]:
  """Features with a non-zero coefficient in an L1-penalised logistic regression.

  Features are median-imputed and standardised first (L1 is scale-sensitive).
  Smaller ``C`` means stronger regularisation and fewer selected features.
  """
  if C <= 0:
    raise ValueError(f"C must be positive, got {C}")
  X_std = StandardScaler().fit_transform(
      SimpleImputer(strategy="median").fit_transform(X))
  model = LogisticRegression(l1_ratio=1.0, solver="liblinear", C=C,
                             random_state=random_state)
  mask = SelectFromModel(model).fit(X_std, y).get_support()
  return [c for c, keep in zip(X.columns, mask) if keep]


def apply_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    corr_threshold: float = 0.9,
    mi_top_k: int | None = 30,
    l1_C: float = 0.1,
    random_state: int = 42,
    return_report: bool = False,
    verbose: bool = True,
) -> list[str] | tuple[list[str], dict]:
  """Correlation filter -> mutual information -> L1 selection (training data only).

  Args:
    X_train: numeric training features (NaNs allowed; imputed internally).
    y_train: binary training labels aligned with ``X_train``.
    corr_threshold: drop one of each pair with |r| above this value.
    mi_top_k: keep at most this many features with mutual information > 0
      (None keeps all with MI > 0).
    l1_C: inverse L1 regularisation strength.
    random_state: seed for mutual information and the logistic regression.
    return_report: also return per-stage results.
    verbose: print a short progress summary.

  Returns:
    The selected feature names, or ``(names, report)`` when ``return_report``.
    If the L1 step removes every feature, the mutual-information set is returned
    instead (with a warning, and ``report["l1_fallback"] = True``).

  Raises:
    TypeError / ValueError: for invalid input types, shapes, labels or parameters.
  """
  if mi_top_k is not None and mi_top_k < 1:
    raise ValueError(f"mi_top_k must be >= 1 or None, got {mi_top_k}")
  y_train = _validate_inputs(X_train, y_train)
  say = print if verbose else (lambda *a, **k: None)

  kept = correlation_filter(X_train, corr_threshold)
  if not kept:
    raise ValueError("Correlation filter removed every feature")
  say(f"1. correlation filter (|r| > {corr_threshold}): {len(kept)}/{X_train.shape[1]} kept")

  mi_table = mutual_information_ranking(X_train[kept], y_train, random_state)
  informative = mi_table.loc[mi_table["mi_score"] > 0, "feature"]
  mi_kept = informative.head(mi_top_k).tolist() if len(informative) else []
  if not mi_kept:  # no feature carries information: keep the ranking's top rows
    warnings.warn("All mutual-information scores are zero; keeping top-ranked features.")
    mi_kept = mi_table["feature"].head(mi_top_k or len(mi_table)).tolist()
  say(f"2. mutual information: {len(mi_kept)} kept")

  selected = l1_selection(X_train[mi_kept], y_train, l1_C, random_state)
  fallback = not selected
  if fallback:
    warnings.warn("L1 removed every feature; returning the mutual-information set.")
    selected = mi_kept
  say(f"3. L1 logistic (C={l1_C}): {len(selected)} selected")

  if return_report:
    return selected, {
        "after_correlation": kept,
        "mi_table": mi_table,
        "after_mi": mi_kept,
        "selected": selected,
        "l1_fallback": fallback,
    }
  return selected
