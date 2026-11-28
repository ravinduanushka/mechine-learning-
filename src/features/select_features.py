"""Feature selection for the cardiometabolic risk model (Member 2).

Two layers live here:

* ``apply_feature_selection`` / ``LeakageSafeSelector`` - the original compact
  pipeline (correlation -> MI -> RFE -> L1), still used by ``evaluate_sets.py``.
* The multi-stage, strategy-comparing pipeline (``run_selection``,
  ``compare_strategies_cv``, ``save_selection`` / ``load_selected_features``)
  used by ``notebooks/04_feature_selection.ipynb``.

Multi-stage pipeline (every step is fitted on TRAINING rows only):
  0. drop non-feature columns (IDs, survey weights, split flag, target)
  1. drop constant / near-constant columns
  2. collinearity: greedy |r| filter that keeps the feature most related to y
  3. candidate subsets from three families of methods
       filter   : mutual information + ANOVA F (Benjamini-Hochberg) + |corr|
       embedded : RF + GB importance, L1 logistic regression (C by inner CV)
       wrapper  : RFECV (logistic regression, ROC-AUC, one-standard-error rule)
     plus a vote-based ``consensus`` subset and an ``all_clean`` baseline.
  4. strategies are compared by repeated stratified CV in which the WHOLE
     selection is re-run inside every training fold, so scores are not
     inflated by selecting on the rows they are measured on.
  5. the winning strategy is refitted on the full training split and saved to
     JSON for ``05_model_training.ipynb``. The test split is never touched.
"""

from __future__ import annotations

import hashlib
import json
import re
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import (
    RFE, RFECV, SelectFromModel, f_classif, mutual_info_classif)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.data_io import COHORT_NOT_FEATURES, TARGET

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SELECTION_PATH = PROJECT_ROOT / "data" / "processed" / "selected_features.json"
SCHEMA_VERSION = 1


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


# =============================================================================
# Multi-stage, strategy-comparing selection (notebook 04)
# =============================================================================
NON_FEATURE_COLUMNS = frozenset(
    {"SEQN", "split", "survey_cycle", "SDMVPSU", "SDMVSTRA", TARGET, *COHORT_NOT_FEATURES})
_SURVEY_WEIGHT = re.compile(r"^WT[A-Z0-9]*$", re.IGNORECASE)   # WTSAF2YR, WTMEC2YR, ...

STRATEGIES = ("all_clean", "filter", "tree", "l1", "rfecv", "consensus")


@dataclass
class SelectionConfig:
  """Hyper-parameters of the multi-stage pipeline (all fitted on training data)."""
  near_constant_freq: float = 0.99     # drop if one value covers >= this share
  corr_threshold: float = 0.90         # |Pearson r| above which a pair is redundant
  filter_alpha: float = 0.05           # BH-adjusted ANOVA p-value cut-off
  filter_top_k: int | None = None      # optional cap on the filter subset
  tree_cum_importance: float = 0.90    # keep features covering this importance share
  l1_Cs: tuple = (0.01, 0.03, 0.1, 0.3, 1.0)
  rfecv_min_features: int = 3
  rfecv_cv: int = 5
  one_se_rule: bool = True             # RFECV: smallest set within 1 SE of the best
  consensus_min_votes: int = 2         # of the 4 methods filter/tree/l1/rfecv
  n_estimators_rf: int = 300
  n_estimators_gb: int = 150
  random_state: int = 42
  n_jobs: int = -1


# ----------------------------------------------------------------- cleaning ----
def drop_non_features(X: pd.DataFrame, extra=()) -> tuple[pd.DataFrame, list[str]]:
  """Remove IDs, survey weights, split flag, target and non-numeric columns."""
  blocked = set(NON_FEATURE_COLUMNS) | set(extra)
  drop = [c for c in X.columns
          if c in blocked or _SURVEY_WEIGHT.match(str(c))
          or not pd.api.types.is_numeric_dtype(X[c])]
  return X.drop(columns=drop), drop


def near_constant_filter(X: pd.DataFrame, max_dominant_freq: float = 0.99) -> list[str]:
  """Columns to DROP: constant, or one value covers >= ``max_dominant_freq`` of rows."""
  drop = []
  for c in X.columns:
    vc = X[c].value_counts(normalize=True, dropna=True)
    if len(vc) <= 1 or vc.iloc[0] >= max_dominant_freq:
      drop.append(c)
  return drop


def _impute(X: pd.DataFrame) -> pd.DataFrame:
  """Median imputation with statistics taken from ``X`` itself (the training rows)."""
  return pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X),
                      index=X.index, columns=X.columns)


def _abs_corr_with_y(Xi: pd.DataFrame, y: pd.Series) -> pd.Series:
  return Xi.apply(lambda c: abs(np.corrcoef(c, y)[0, 1])).fillna(0.0)


def ranked_correlation_filter(
    X: pd.DataFrame, y: pd.Series, threshold: float = 0.9
) -> tuple[list[str], dict[str, str]]:
  """Greedy redundancy filter that keeps the member of a pair most related to ``y``.

  Features are visited from most to least |corr with y|; a feature is kept
  unless it correlates above ``threshold`` with one already kept.
  Returns ``(kept, {dropped: kept_feature_it_duplicates})``.
  """
  if not 0 < threshold <= 1:
    raise ValueError(f"threshold must be in (0, 1], got {threshold}")
  Xi = _impute(X)
  order = _abs_corr_with_y(Xi, y).sort_values(ascending=False, kind="stable").index
  corr = Xi.corr().abs()
  kept: list[str] = []
  dropped: dict[str, str] = {}
  for c in order:
    twin = next((k for k in kept if corr.loc[c, k] > threshold), None)
    if twin is None:
      kept.append(c)
    else:
      dropped[c] = twin
  keep_set = set(kept)
  return [c for c in X.columns if c in keep_set], dropped   # original column order


# ------------------------------------------------------------------- filters ----
def _benjamini_hochberg(p: np.ndarray) -> np.ndarray:
  p = np.asarray(p, dtype=float)
  n = len(p)
  order = np.argsort(p)
  adj = p[order] * n / np.arange(1, n + 1)
  adj = np.minimum.accumulate(adj[::-1])[::-1]
  out = np.empty(n)
  out[order] = np.clip(adj, 0, 1)
  return out


def filter_scores(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> pd.DataFrame:
  """Univariate evidence per feature: MI, ANOVA F (BH-adjusted), |corr with y|."""
  Xi = _impute(X)
  discrete = (Xi.nunique() <= 2).to_numpy()
  mi = mutual_info_classif(Xi, y, discrete_features=discrete, random_state=random_state)
  f_stat, p_val = f_classif(Xi, y)
  p_val = np.nan_to_num(p_val, nan=1.0)
  out = pd.DataFrame({
      "feature": X.columns, "mi": mi, "f_stat": np.nan_to_num(f_stat),
      "f_pvalue": p_val, "f_pvalue_bh": _benjamini_hochberg(p_val),
      "abs_corr_y": _abs_corr_with_y(Xi, y).to_numpy(),
  })
  out["mean_rank"] = out[["mi", "f_stat", "abs_corr_y"]].rank(ascending=False).mean(axis=1)
  return out.sort_values("mean_rank", ignore_index=True)


def select_filter(scores: pd.DataFrame, alpha: float = 0.05, top_k: int | None = None) -> list[str]:
  """Significant (BH) AND informative (MI > 0) features, best mean rank first."""
  keep = scores[(scores["f_pvalue_bh"] < alpha) & (scores["mi"] > 0)]
  if keep.empty:                      # nothing passes: fall back to the single best
    keep = scores.head(1)
  if top_k:
    keep = keep.head(top_k)
  return keep["feature"].tolist()


# ------------------------------------------------------------------ embedded ----
def tree_importances(X: pd.DataFrame, y: pd.Series, cfg: SelectionConfig) -> pd.DataFrame:
  """Impurity importances from a Random Forest and a Gradient Boosting model."""
  Xi = _impute(X)
  rf = RandomForestClassifier(
      n_estimators=cfg.n_estimators_rf, min_samples_leaf=5, class_weight="balanced_subsample",
      n_jobs=cfg.n_jobs, random_state=cfg.random_state).fit(Xi, y)
  gb = GradientBoostingClassifier(
      n_estimators=cfg.n_estimators_gb, max_depth=3, subsample=0.8,
      random_state=cfg.random_state).fit(Xi, y)
  out = pd.DataFrame({"feature": X.columns, "rf_importance": rf.feature_importances_,
                      "gb_importance": gb.feature_importances_})
  out["tree_importance"] = out[["rf_importance", "gb_importance"]].apply(
      lambda c: c / c.sum() if c.sum() > 0 else c).mean(axis=1)
  return out.sort_values("tree_importance", ascending=False, ignore_index=True)


def select_tree(imp: pd.DataFrame, cum: float = 0.90) -> list[str]:
  """Smallest top set whose combined importance reaches ``cum`` of the total."""
  ordered = imp.sort_values("tree_importance", ascending=False)
  share = ordered["tree_importance"].cumsum() / ordered["tree_importance"].sum()
  n = int(np.searchsorted(share.to_numpy(), cum) + 1)
  return ordered["feature"].head(max(1, n)).tolist()


def l1_scores(X: pd.DataFrame, y: pd.Series, cfg: SelectionConfig) -> tuple[pd.DataFrame, float]:
  """L1 logistic coefficients with C picked by inner CV (ROC-AUC) on training rows."""
  pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                   ("clf", LogisticRegression(l1_ratio=1.0, solver="liblinear",
                                              class_weight="balanced",
                                              random_state=cfg.random_state))])
  search = GridSearchCV(
      pipe, {"clf__C": list(cfg.l1_Cs)}, scoring="roc_auc", n_jobs=cfg.n_jobs,
      cv=StratifiedKFold(5, shuffle=True, random_state=cfg.random_state)).fit(X, y)
  coef = search.best_estimator_.named_steps["clf"].coef_.ravel()
  out = pd.DataFrame({"feature": X.columns, "l1_coef": coef, "l1_abs_coef": np.abs(coef)})
  return (out.sort_values("l1_abs_coef", ascending=False, ignore_index=True),
          float(search.best_params_["clf__C"]))


def select_l1(coefs: pd.DataFrame) -> list[str]:
  keep = coefs.loc[coefs["l1_abs_coef"] > 0, "feature"].tolist()
  return keep or coefs["feature"].head(1).tolist()


# ------------------------------------------------------------------- wrapper ----
def rfecv_select(X: pd.DataFrame, y: pd.Series, cfg: SelectionConfig
                 ) -> tuple[list[str], pd.DataFrame]:
  """RFECV with a logistic regression, scored by ROC-AUC.

  Imputer and scaler sit inside the estimator, so they are refitted on the
  training part of every internal fold. With ``one_se_rule`` the smallest
  subset within one standard error of the best CV score is returned.
  Returns ``(features, curve)`` where ``curve`` is n_features vs CV score.
  """
  est = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                  ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                             random_state=cfg.random_state))])
  min_feat = max(1, min(cfg.rfecv_min_features, X.shape[1]))
  rfecv = RFECV(est, step=1, min_features_to_select=min_feat, scoring="roc_auc",
                cv=StratifiedKFold(cfg.rfecv_cv, shuffle=True, random_state=cfg.random_state),
                importance_getter="named_steps.clf.coef_", n_jobs=cfg.n_jobs).fit(X.to_numpy(), y)
  res = rfecv.cv_results_
  n_points = len(res["mean_test_score"])
  # candidate subset sizes run from the full set down to min_feat, in steps of 1
  n_feats = np.asarray(res["n_features"]) if "n_features" in res else \
      np.arange(min_feat, min_feat + n_points)
  curve = pd.DataFrame({"n_features": n_feats, "mean_auc": res["mean_test_score"],
                        "std_auc": res["std_test_score"]})
  n_best = int(rfecv.n_features_)
  if cfg.one_se_rule:
    best = curve.loc[curve["mean_auc"].idxmax()]
    se = best["std_auc"] / np.sqrt(cfg.rfecv_cv)
    n_best = int(curve.loc[curve["mean_auc"] >= best["mean_auc"] - se, "n_features"].min())
  if n_best == rfecv.n_features_:
    mask = rfecv.support_
  else:
    mask = RFE(est, n_features_to_select=n_best, step=1,
               importance_getter="named_steps.clf.coef_").fit(X.to_numpy(), y).support_
  return [c for c, m in zip(X.columns, mask) if m], curve


# --------------------------------------------------------------- orchestration ----
@dataclass
class SelectionResult:
  """Everything learned from one training split."""
  subsets: dict[str, list[str]]                 # strategy -> selected features
  dropped_non_feature: list[str] = field(default_factory=list)
  dropped_near_constant: list[str] = field(default_factory=list)
  dropped_collinear: dict[str, str] = field(default_factory=dict)
  scores: pd.DataFrame | None = None            # one row per surviving candidate
  rfecv_curve: pd.DataFrame | None = None
  l1_C: float | None = None
  candidates: list[str] = field(default_factory=list)   # after stages 0-2


def run_selection(X_train: pd.DataFrame, y_train, cfg: SelectionConfig | None = None,
                  extra_non_features=()) -> SelectionResult:
  """Run every stage on TRAINING data and return all candidate subsets."""
  cfg = cfg or SelectionConfig()
  if not isinstance(X_train, pd.DataFrame):
    raise TypeError("X_train must be a pandas DataFrame")
  y = pd.Series(np.asarray(y_train), index=X_train.index)
  if y.isna().any() or y.nunique() < 2:
    raise ValueError("y_train must be non-missing and contain both classes")

  X, non_feat = drop_non_features(X_train, extra_non_features)
  if X.shape[1] == 0:
    raise ValueError("No numeric feature columns in X_train")
  if np.isinf(X.to_numpy(dtype=float)).any():
    raise ValueError("X_train contains infinite values")
  const = near_constant_filter(X, cfg.near_constant_freq)
  X = X.drop(columns=const)
  if X.shape[1] == 0:
    raise ValueError("Every feature is constant / near-constant in the training data")
  kept, collinear = ranked_correlation_filter(X, y, cfg.corr_threshold)
  X = X[kept]

  fs = filter_scores(X, y, cfg.random_state)
  ti = tree_importances(X, y, cfg)
  l1, l1_C = l1_scores(X, y, cfg)
  rf_feats, curve = rfecv_select(X, y, cfg)

  methods = {
      "filter": select_filter(fs, cfg.filter_alpha, cfg.filter_top_k),
      "tree": select_tree(ti, cfg.tree_cum_importance),
      "l1": select_l1(l1),
      "rfecv": rf_feats,
  }
  votes = pd.Series(0, index=X.columns)
  for feats in methods.values():
    votes[feats] += 1
  consensus = votes[votes >= min(cfg.consensus_min_votes, len(methods))].index.tolist()
  if not consensus:
    consensus = votes.sort_values(ascending=False, kind="stable").index[:1].tolist()

  order = {c: i for i, c in enumerate(X.columns)}
  subsets = {"all_clean": list(X.columns), **methods, "consensus": consensus}
  subsets = {k: sorted(v, key=order.get) for k, v in subsets.items()}

  scores = (fs.merge(ti, on="feature").merge(l1, on="feature")
              .assign(votes=lambda d: d["feature"].map(votes)))
  for name, feats in methods.items():
    scores[f"sel_{name}"] = scores["feature"].isin(feats)
  scores["sel_consensus"] = scores["feature"].isin(consensus)
  scores = scores.sort_values(["votes", "tree_importance"], ascending=False, ignore_index=True)
  return SelectionResult(subsets, non_feat, const, collinear, scores, curve, l1_C, list(X.columns))


class StagedFeatureSelector(BaseEstimator, TransformerMixin):
  """sklearn transformer: re-runs ``run_selection`` inside ``fit`` (fold-safe)."""

  def __init__(self, strategy: str = "consensus", config: SelectionConfig | None = None):
    self.strategy = strategy
    self.config = config

  def fit(self, X, y):
    if self.strategy not in STRATEGIES:
      raise ValueError(f"strategy must be one of {STRATEGIES}, got {self.strategy!r}")
    self.result_ = run_selection(pd.DataFrame(X), y, self.config)
    self.selected_ = self.result_.subsets[self.strategy]
    return self

  def transform(self, X):
    return pd.DataFrame(X)[self.selected_]


# ------------------------------------------------------------- CV comparison ----
def default_eval_models(seed: int = 42) -> dict:
  """Two contrasting learners used only to score candidate subsets."""
  return {
      "LogReg": Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                          ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                                     random_state=seed))]),
      "RandForest": Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("clf", RandomForestClassifier(
                                  n_estimators=200, min_samples_leaf=5, n_jobs=-1,
                                  class_weight="balanced_subsample", random_state=seed))]),
  }


def _metrics(y_true, proba, threshold: float = 0.5) -> dict:
  pred = (proba >= threshold).astype(int)
  return {"roc_auc": roc_auc_score(y_true, proba),
          "pr_auc": average_precision_score(y_true, proba),
          "f1": f1_score(y_true, pred, zero_division=0),
          "precision": precision_score(y_true, pred, zero_division=0),
          "recall": recall_score(y_true, pred, zero_division=0)}


def compare_strategies_cv(X_train: pd.DataFrame, y_train, cfg: SelectionConfig | None = None,
                          n_splits: int = 5, n_repeats: int = 2, models: dict | None = None,
                          strategies=STRATEGIES, verbose: bool = True) -> dict:
  """Score every strategy with the whole selection re-run inside each training fold.

  Returns a dict with
    folds     long table (strategy, model, split, metrics)
    summary   per-strategy means over models and splits (+ ROC-AUC std/SE, Jaccard)
    by_model  per strategy x model means
    stability feature x strategy selection frequency across splits
  Metrics use a 0.5 threshold (both learners are class-balanced); ROC-AUC and
  PR-AUC are threshold-free.
  """
  cfg = cfg or SelectionConfig()
  models = models or default_eval_models(cfg.random_state)
  y = pd.Series(np.asarray(y_train), index=X_train.index)
  rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                 random_state=cfg.random_state)
  total = n_splits * n_repeats
  rows, fold_subsets = [], {s: [] for s in strategies}
  for i, (tr, va) in enumerate(rskf.split(X_train, y)):
    X_tr, y_tr, X_va, y_va = X_train.iloc[tr], y.iloc[tr], X_train.iloc[va], y.iloc[va]
    res = run_selection(X_tr, y_tr, cfg)          # fitted on the training fold only
    for strat in strategies:
      feats = res.subsets[strat]
      fold_subsets[strat].append(feats)
      for mname, model in models.items():
        fitted = clone(model).fit(X_tr[feats], y_tr)
        rows.append({"strategy": strat, "model": mname, "split": i, "n_features": len(feats),
                     **_metrics(y_va, fitted.predict_proba(X_va[feats])[:, 1])})
    if verbose:
      print(f"  split {i + 1}/{total} done", end="\r")
  folds = pd.DataFrame(rows)

  metric_cols = ["roc_auc", "pr_auc", "f1", "precision", "recall"]
  by_split = folds.groupby(["strategy", "split"]).agg(
      n_features=("n_features", "first"), **{m: (m, "mean") for m in metric_cols}).reset_index()
  g = by_split.groupby("strategy")
  summary = g[metric_cols].mean()
  summary["roc_auc_std"] = g["roc_auc"].std()
  summary["roc_auc_se"] = summary["roc_auc_std"] / np.sqrt(total)
  summary["n_features_mean"] = g["n_features"].mean()
  summary["n_features_std"] = g["n_features"].std()

  def _jaccard(subs):
    sets = [set(s) for s in subs]
    pairs = [len(a & b) / len(a | b) for k, a in enumerate(sets) for b in sets[k + 1:]]
    return float(np.mean(pairs)) if pairs else 1.0
  summary["jaccard_stability"] = pd.Series({s: _jaccard(v) for s, v in fold_subsets.items()})
  summary = summary.loc[list(strategies)].sort_values("roc_auc", ascending=False)
  by_model = folds.groupby(["strategy", "model"])[metric_cols + ["n_features"]].mean()

  stability = pd.DataFrame({
      s: pd.Series([c for sub in subs for c in sub]).value_counts() / len(subs)
      for s, subs in fold_subsets.items()}).fillna(0.0)
  stability = stability.loc[stability.max(axis=1).sort_values(ascending=False).index]
  return {"folds": folds, "summary": summary, "by_model": by_model, "stability": stability}


def choose_strategy(summary: pd.DataFrame, one_se_rule: bool = True) -> str:
  """Best mean ROC-AUC; with the 1-SE rule, the fewest features within one SE of it."""
  best = summary["roc_auc"].idxmax()
  if not one_se_rule:
    return best
  floor = summary.loc[best, "roc_auc"] - summary.loc[best, "roc_auc_se"]
  ok = summary[summary["roc_auc"] >= floor]
  return ok.sort_values(["n_features_mean", "roc_auc"], ascending=[True, False]).index[0]


# ------------------------------------------------------------- persistence ----
def columns_fingerprint(columns) -> str:
  """Short hash of a column list, to detect a rebuilt / changed feature set."""
  return hashlib.sha256("|".join(map(str, columns)).encode()).hexdigest()[:16]


def build_selection_record(strategy: str, result: SelectionResult, cfg: SelectionConfig,
                           input_columns, n_train: int,
                           cv_summary_row: pd.Series | None = None) -> dict:
  """JSON-ready record for one feature set."""
  input_columns = list(input_columns)
  return {
      "strategy": strategy,
      "selected_features": result.subsets[strategy],
      "n_selected": len(result.subsets[strategy]),
      "n_input_columns": len(input_columns),
      "input_columns_fingerprint": columns_fingerprint(input_columns),
      "n_train": int(n_train),
      "candidates_after_cleaning": result.candidates,
      "dropped": {"non_feature": result.dropped_non_feature,
                  "near_constant": result.dropped_near_constant,
                  "collinear": result.dropped_collinear},
      "all_strategy_subsets": result.subsets,
      "l1_C": result.l1_C,
      "cv_summary": (None if cv_summary_row is None
                     else {k: float(v) for k, v in cv_summary_row.items()}),
      "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()},
  }


def save_selection(records: dict[str, dict], path: Path | str = DEFAULT_SELECTION_PATH,
                   meta: dict | None = None) -> Path:
  """Write {feature_set_name: record} to JSON, merging into an existing file."""
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  old = json.loads(path.read_text(encoding="utf8")) if path.exists() else {}
  doc = {"schema_version": SCHEMA_VERSION, "target": TARGET,
         "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
         "meta": {**old.get("meta", {}), **(meta or {})},
         "feature_sets": {**old.get("feature_sets", {}), **records}}
  path.write_text(json.dumps(doc, indent=2), encoding="utf8")
  return path


def load_selected_features(feature_set: str, path: Path | str = DEFAULT_SELECTION_PATH,
                           available_columns=None) -> list[str]:
  """Selected feature names for one feature set (e.g. ``"Set B"``).

  If ``available_columns`` is given, raises when a selected feature is absent
  and warns when the raw column list differs from the one used for selection.
  """
  path = Path(path)
  if not path.exists():
    raise FileNotFoundError(f"{path} not found - run notebooks/04_feature_selection.ipynb first")
  doc = json.loads(path.read_text(encoding="utf8"))
  if doc.get("schema_version") != SCHEMA_VERSION:
    raise ValueError(f"Unsupported schema_version {doc.get('schema_version')} in {path}")
  if feature_set not in doc["feature_sets"]:
    raise KeyError(f"{feature_set!r} not in {path}; available: {list(doc['feature_sets'])}")
  rec = doc["feature_sets"][feature_set]
  feats = rec["selected_features"]
  if available_columns is not None:
    cols = list(available_columns)
    missing = [f for f in feats if f not in cols]
    if missing:
      raise ValueError(f"Selected features missing from the data: {missing}")
    if columns_fingerprint(cols) != rec["input_columns_fingerprint"]:
      warnings.warn(f"{feature_set}: columns differ from those used at selection time; "
                    "re-run notebook 04 if the feature set was rebuilt.")
  return feats
