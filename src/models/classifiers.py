"""Classifiers and hyperparameter grids for Member 3 (Model Development).

Progress-evaluation scope: 3 models from 3 different families.

  - Logistic Regression  -> linear baseline            (M3-02)
  - Random Forest        -> tree ensemble, bagging      (M3-04)
  - Gradient Boosting    -> tree ensemble, boosting     (M3-05)

Each model is returned UNFITTED. Fitting happens later inside the leakage-safe
pipeline and cross-validation (see ``src/models/pipeline.py``).

Grid keys start with ``clf__`` because the classifier will be the step named
"clf" inside the scikit-learn Pipeline.
"""

from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42


def get_models(random_state: int = RANDOM_STATE) -> dict:
    """Return {model name: unfitted classifier}."""
    return {
        "Logistic Regression": LogisticRegression(
            solver="saga",             # supports both L1 and L2 penalties
            max_iter=5000,
            class_weight="balanced",   # gives the smaller class (risk = 1) more weight
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        ),
        # GradientBoostingClassifier has no class_weight option; the classes are
        # only mildly imbalanced (57% / 43%), so it is trained unweighted.
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=random_state,
        ),
    }


def get_param_grids() -> dict:
    """Return {model name: hyperparameter grid for GridSearchCV}."""
    return {
        "Logistic Regression": {
            "clf__C": [0.001, 0.01, 0.1, 1, 10, 100],   # small C = stronger regularisation
            "clf__l1_ratio": [0.0, 1.0],                 # 0 = L2 (ridge), 1 = L1 (lasso)
        },
        "Random Forest": {
            "clf__n_estimators": [300],
            "clf__max_depth": [None, 8, 16],
            "clf__min_samples_leaf": [1, 5, 10],
            "clf__max_features": ["sqrt", 0.3],
        },
        "Gradient Boosting": {
            "clf__learning_rate": [0.05, 0.1],
            "clf__n_estimators": [100, 300],
            "clf__max_depth": [2, 3],
            "clf__subsample": [0.8, 1.0],
        },
    }
