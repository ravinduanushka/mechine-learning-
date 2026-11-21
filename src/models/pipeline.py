"""Leakage-safe training pipeline for Member 3 (M3-01, M3-06).

Three building blocks:

  1. load_feature_set()  -> reads a feature set and splits it using the
                            fixed 80/20 stratified `split` column (M3-01)
  2. build_pipeline()    -> Impute -> Scale -> Classifier
  3. tune_model()        -> GridSearchCV inside stratified 5-fold CV (M3-06)
  4. oof_probabilities() / best_f1_threshold()
                         -> decision threshold chosen on training data only

Leakage rules followed here:
  - The test rows are separated FIRST and are never passed to tune_model().
  - The imputer and scaler live inside the Pipeline, so in every CV fold they
    are fitted on the 4 training folds only, never on the validation fold.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.classifiers import RANDOM_STATE, get_models, get_param_grids

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TARGET = "metabolic_syndrome_target"
SPLIT_COL = "split"

FEATURE_SET_FILES = {
    "Set 0": "feature_set_0.parquet",   # demographics only
    "Set A": "feature_set_a.parquet",   # + basic activity summaries
    "Set B": "feature_set_b.parquet",   # + engineered, PCA and K-means features
}


# ---------------------------------------------------------------------------
# 1. Data loading and train/test split (M3-01)
# ---------------------------------------------------------------------------
def load_feature_set(name: str, common_ids: bool = True):
    """Return X_train, X_test, y_train, y_test for one feature set.

    The stratified 80/20 split (seed 42) was created once by Member 2 and saved
    in the `split` column, so every set uses the SAME train and test people.

    Set 0 has 4,603 people but Sets A/B have 4,575 (28 have no minute data).
    With ``common_ids=True`` Set 0 is reduced to the same 4,575 people, so the
    comparison Set 0 vs A vs B is fair.
    """
    df = pd.read_parquet(PROCESSED_DIR / FEATURE_SET_FILES[name])

    if common_ids:
        ref_index = pd.read_parquet(PROCESSED_DIR / FEATURE_SET_FILES["Set B"],
                                    columns=[SPLIT_COL]).index
        df = df.loc[df.index.intersection(ref_index)]

    train = df[df[SPLIT_COL] == "train"]
    test = df[df[SPLIT_COL] == "test"]
    assert set(train.index).isdisjoint(test.index), "train and test overlap!"

    features = [c for c in df.columns if c not in (TARGET, SPLIT_COL)]
    return (train[features], test[features],
            train[TARGET].astype(int), test[TARGET].astype(int))


# ---------------------------------------------------------------------------
# 2. Leakage-safe pipeline
# ---------------------------------------------------------------------------
def build_pipeline(classifier) -> Pipeline:
    """Impute missing values -> standardise -> classifier."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),  # Set B has a few NaNs
        ("scale", StandardScaler()),                    # mean 0, std 1 (needed by LR)
        ("clf", clone(classifier)),
    ])


# ---------------------------------------------------------------------------
# 3. Hyperparameter tuning with stratified 5-fold CV (M3-06)
# ---------------------------------------------------------------------------
def tune_model(model_name: str, X_train, y_train, n_splits: int = 5,
               scoring: str = "roc_auc", n_jobs: int = -1) -> GridSearchCV:
    """Grid-search one model on the TRAINING data only.

    Every hyperparameter combination is scored by 5-fold stratified CV; the
    best one (highest mean ROC-AUC) is then refitted on the whole training set.
    """
    pipe = build_pipeline(get_models()[model_name])
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        estimator=pipe,
        param_grid=get_param_grids()[model_name],
        scoring=scoring,
        cv=cv,
        refit=True,      # refit best pipeline on all training data
        n_jobs=n_jobs,
    )
    return search.fit(X_train, y_train)


# ---------------------------------------------------------------------------
# 4. Out-of-fold probabilities and decision threshold (proposal section 4.8)
# ---------------------------------------------------------------------------
def oof_probabilities(search: GridSearchCV, X_train, y_train,
                      n_splits: int = 5) -> np.ndarray:
    """Out-of-fold P(risk = 1) for every TRAINING participant.

    The tuned pipeline is refitted in 5 folds; each person gets a probability
    from a model that never saw them. Used only to choose the threshold, so
    the test set stays untouched.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_predict(search.best_estimator_, X_train, y_train, cv=cv,
                             method="predict_proba", n_jobs=-1)[:, 1]


def best_f1_threshold(y_true, proba) -> float:
    """Threshold tau that maximises F1 on the given (out-of-fold) probabilities."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.argmax(f1[:-1])])  # last point has no threshold
