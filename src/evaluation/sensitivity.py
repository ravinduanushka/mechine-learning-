"""Sensitivity analyses: survey weights (E4), cross-cycle transfer (E5), feature-group ablation (E6)."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.evaluation.explainability import make_tuned_pipeline, refit_tuned_pipeline
from src.models.classifiers import RANDOM_STATE
from src.models.pipeline import PROCESSED_DIR, TARGET, SPLIT_COL, tune_model

CYCLE_COL = "cycle_2013_2014"   # 0 = 2011-2012, 1 = 2013-2014

# Set B activity columns grouped by what they measure.
FEATURE_GROUPS = {
    "Basic activity": ["mean_daily_mims", "wake_wear_min", "sleep_wear_min"],
    "Intensity distribution": ["intensity_gradient", "mims_q1", "mims_q2", "mims_q3", "mims_q4",
                               "mims_median", "mims_p90", "min_high_intensity",
                               "longest_active_bout"],
    "Sedentary behaviour": ["sedentary_fraction", "sedentary_bouts_30min",
                            "sedentary_breaks_per_hour", "longest_sedentary_bout"],
    "Rhythm & variability": ["relative_amplitude", "day_to_day_cv_mims", "activity_entropy"],
    "Unsupervised (PCA + k-means)": ["pca_score_1", "pca_score_2", "pca_score_3",
                                     "phenotype_1", "phenotype_2", "phenotype_3"],
}


def load_survey_weights(index) -> pd.Series:
    """Fasting survey weights WTSAF2YR / 2, scaled so positive weights average 1.

    Weight 0 (outside the NHANES fasting subsample) is kept, which leaves those
    participants out of weighted fits and scores.
    """
    base = pd.read_parquet(PROCESSED_DIR / "clean_cohort_base.parquet")
    w = base.assign(SEQN=base.SEQN.astype("int64")).set_index("SEQN")["WTSAF2YR"] / 2
    w = w.reindex(index)
    if w.isna().any() or (w < 0).any():
        raise ValueError("missing or negative survey weights")
    return w / w[w > 0].mean()


def _scores(y, p, weight=None) -> dict:
    return {"roc_auc": roc_auc_score(y, p, sample_weight=weight),
            "pr_auc": average_precision_score(y, p, sample_weight=weight),
            "brier": brier_score_loss(y, p, sample_weight=weight)}


def survey_weighted_refit(model_name, best_params, X_train, X_test, y_train, y_test) -> pd.DataFrame:
    """E4: train unweighted vs survey-weighted; score each on test unweighted and weighted."""
    w_train = load_survey_weights(X_train.index)
    w_test = load_survey_weights(X_test.index)
    rows = []
    for training, weight in [("unweighted", None), ("survey-weighted", w_train)]:
        pipe = refit_tuned_pipeline(model_name, best_params, X_train, y_train, weight)
        p = pipe.predict_proba(X_test)[:, 1]
        for evaluation, w_eval in [("unweighted", None), ("survey-weighted", w_test)]:
            n_used = len(y_test) if w_eval is None else int((w_eval > 0).sum())
            rows.append({"model": model_name, "training": training, "test_evaluation": evaluation,
                         "n_train": len(y_train) if weight is None else int((weight > 0).sum()),
                         "n_test": n_used, **_scores(y_test, p, w_eval)})
    return pd.DataFrame(rows)


def cross_cycle_transfer(feature_file: str, model_name: str) -> pd.DataFrame:
    """E5: tune and train on one survey cycle, test on the other (both directions)."""
    df = pd.read_parquet(PROCESSED_DIR / feature_file)
    X = df.drop(columns=[TARGET, SPLIT_COL, CYCLE_COL])   # cycle is constant within a cycle
    y = df[TARGET].astype(int)
    cycle = df[CYCLE_COL]

    rows = []
    for train_c, test_c, label in [(0, 1, "2011-12 -> 2013-14"), (1, 0, "2013-14 -> 2011-12")]:
        tr, te = cycle == train_c, cycle == test_c
        search = tune_model(model_name, X[tr], y[tr])
        p = search.predict_proba(X[te])[:, 1]
        rows.append({"model": model_name, "direction": label,
                     "n_train": int(tr.sum()), "n_test": int(te.sum()),
                     "cv_auc_train_cycle": search.best_score_, **_scores(y[te], p),
                     "y_proba": p, "y_true": y[te].to_numpy()})
    return pd.DataFrame(rows)


def feature_group_ablation(model_name, best_params, X_train, y_train,
                           groups: dict = FEATURE_GROUPS, n_splits: int = 5) -> pd.DataFrame:
    """E6: 5-fold CV ROC-AUC on training data with each feature group removed.

    delta_auc < 0 means the removed group was useful.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    def cv_auc(cols):
        scores = cross_val_score(make_tuned_pipeline(model_name, best_params),
                                 X_train[cols], y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        return scores.mean(), scores.std()

    all_cols = list(X_train.columns)
    full_mean, full_std = cv_auc(all_cols)
    rows = [{"model": model_name, "removed_group": "(none: full Set B)", "n_features": len(all_cols),
             "cv_roc_auc": full_mean, "cv_std": full_std, "delta_auc": 0.0}]
    for group, cols in groups.items():
        keep = [c for c in all_cols if c not in cols]
        mean, std = cv_auc(keep)
        rows.append({"model": model_name, "removed_group": group, "n_features": len(keep),
                     "cv_roc_auc": mean, "cv_std": std, "delta_auc": mean - full_mean})
    return pd.DataFrame(rows)
