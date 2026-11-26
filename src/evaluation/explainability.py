"""Loading the tuned pipelines, feature importance and feature-selection stability."""

from __future__ import annotations

import ast

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold

from src.features.selectors import correlation_filter, l1_selection, mutual_information_ranking
from src.models.classifiers import RANDOM_STATE, get_models
from src.models.pipeline import PROCESSED_DIR, build_pipeline

CV_SUMMARY_FILE = PROCESSED_DIR / "cv_results_summary.parquet"


# --- Tuned pipelines ---------------------------------------------------------
def load_cv_summary() -> pd.DataFrame:
    """CV results table with best_params converted from text back to dicts."""
    summary = pd.read_parquet(CV_SUMMARY_FILE)
    summary["best_params"] = summary["best_params"].map(ast.literal_eval)
    return summary


def make_tuned_pipeline(model_name: str, best_params: dict):
    """Unfitted pipeline (impute -> scale -> classifier) with the tuned hyperparameters."""
    pipe = build_pipeline(get_models()[model_name])
    return pipe.set_params(**{f"clf__{k}": v for k, v in best_params.items()})


def refit_tuned_pipeline(model_name: str, best_params: dict, X_train, y_train,
                         sample_weight=None):
    """Fit the tuned pipeline on training data, optionally with sample weights."""
    pipe = make_tuned_pipeline(model_name, best_params)
    fit_kwargs = {} if sample_weight is None else {"clf__sample_weight": np.asarray(sample_weight)}
    return pipe.fit(X_train, y_train, **fit_kwargs)


def load_or_rebuild_pipelines(data: dict, joblib_path) -> tuple[dict, str]:
    """Return {(feature_set, model): fitted pipeline} and a note on where they came from.

    Loads models/tuned_pipelines.joblib if it exists and is readable. Otherwise
    refits every pipeline from cv_results_summary and saves the joblib file.
    """
    import joblib

    if joblib_path.exists():
        try:
            stored = joblib.load(joblib_path)
            pipes = {(s, m): p for s, models in stored.items() for m, p in models.items()}
            return pipes, f"loaded {joblib_path.name}"
        except Exception as err:   # e.g. saved with another scikit-learn version
            print(f"Could not load {joblib_path.name} ({err!r}); rebuilding.")

    pipes = {}
    for _, r in load_cv_summary().iterrows():
        X_train, _, y_train, _ = data[r.feature_set]
        pipes[(r.feature_set, r.model)] = refit_tuned_pipeline(r.model, r.best_params, X_train, y_train)

    nested = {}
    for (s, m), p in pipes.items():
        nested.setdefault(s, {})[m] = p
    joblib_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(nested, joblib_path)
    return pipes, f"rebuilt from cv_results_summary and saved to {joblib_path.name}"


# --- Feature importance ------------------------------------------------------
def lr_odds_ratios(pipe, feature_names) -> pd.DataFrame:
    """Logistic Regression odds ratio per +1 SD of each feature (OR > 1 raises risk)."""
    coef = pipe.named_steps["clf"].coef_.ravel()
    return (pd.DataFrame({"feature": list(feature_names), "coefficient": coef,
                          "odds_ratio": np.exp(coef)})
            .assign(abs_coef=lambda d: d.coefficient.abs())
            .sort_values("abs_coef", ascending=False, ignore_index=True)
            .drop(columns="abs_coef"))


def tree_importance(pipe, feature_names) -> pd.DataFrame:
    """Gini (impurity) importance of a fitted tree ensemble."""
    imp = pipe.named_steps["clf"].feature_importances_
    return (pd.DataFrame({"feature": list(feature_names), "importance": imp})
            .sort_values("importance", ascending=False, ignore_index=True))


def permutation_importance_auc(pipe, X_test, y_test, n_repeats: int = 20,
                               seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Drop in test ROC-AUC when each column is randomly shuffled."""
    result = permutation_importance(pipe, X_test, y_test, scoring="roc_auc",
                                    n_repeats=n_repeats, random_state=seed, n_jobs=-1)
    return (pd.DataFrame({"feature": X_test.columns,
                          "importance_mean": result.importances_mean,
                          "importance_std": result.importances_std})
            .sort_values("importance_mean", ascending=False, ignore_index=True))


def rank_table(**importances: pd.Series) -> pd.DataFrame:
    """Rank features under each importance method (1 = top) and add the mean rank."""
    ranks = pd.DataFrame({name: s.abs().rank(ascending=False) for name, s in importances.items()})
    return ranks.assign(mean_rank=ranks.mean(axis=1)).sort_values("mean_rank")


# --- Selection stability -----------------------------------------------------
def selection_stability(X_train: pd.DataFrame, y_train: pd.Series, n_splits: int = 5,
                        mi_top_k: int = 15, l1_C: float = 0.05) -> pd.DataFrame:
    """Share of training folds in which each feature is selected.

    filter_mi:   correlation filter, then top-k mutual information
    embedded_l1: non-zero coefficient in L1 logistic regression
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    counts = {"filter_mi": pd.Series(0.0, index=X_train.columns),
              "embedded_l1": pd.Series(0.0, index=X_train.columns)}

    for fit_idx, _ in cv.split(X_train, y_train):
        X_fold, y_fold = X_train.iloc[fit_idx], y_train.iloc[fit_idx]

        kept = correlation_filter(X_fold, 0.9)
        mi = mutual_information_ranking(X_fold[kept], y_fold, RANDOM_STATE)
        counts["filter_mi"][mi["feature"].head(mi_top_k)] += 1

        counts["embedded_l1"][l1_selection(X_fold, y_fold, C=l1_C)] += 1

    freq = pd.DataFrame(counts) / n_splits
    freq["both_methods"] = freq.min(axis=1)
    return freq.sort_values(["both_methods", "embedded_l1"], ascending=False)
