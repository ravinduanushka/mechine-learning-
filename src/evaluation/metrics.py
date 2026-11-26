"""Classification metrics and the DeLong test for comparing two ROC-AUCs."""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Metric order used in every table and figure.
METRIC_NAMES = [
    "accuracy", "precision", "recall", "f1", "balanced_accuracy",
    "roc_auc", "pr_auc", "brier",
]


def classification_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    """All metrics for one model. A probability >= threshold is predicted as 1."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "brier": brier_score_loss(y_true, y_proba),   # lower is better
    }


def _structural_components(pos: np.ndarray, neg: np.ndarray):
    """DeLong placement values for positives (V10) and negatives (V01)."""
    diff = pos[:, None] - neg[None, :]
    psi = (diff > 0) + 0.5 * (diff == 0)
    return psi.mean(axis=1), psi.mean(axis=0)


def delong_test(y_true, proba_a, proba_b) -> dict:
    """Two-sided test that two models scored on the same people have equal ROC-AUC."""
    y_true = np.asarray(y_true).astype(bool)
    a = np.asarray(proba_a, dtype=float)
    b = np.asarray(proba_b, dtype=float)

    v10_a, v01_a = _structural_components(a[y_true], a[~y_true])
    v10_b, v01_b = _structural_components(b[y_true], b[~y_true])
    auc_a, auc_b = v10_a.mean(), v10_b.mean()

    # Covariance of the two AUCs (they share the same participants)
    s10 = np.cov(np.vstack([v10_a, v10_b]))
    s01 = np.cov(np.vstack([v01_a, v01_b]))
    cov = s10 / len(v10_a) + s01 / len(v01_a)

    var_diff = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var_diff <= 0:   # identical predictions
        return {"auc_a": auc_a, "auc_b": auc_b, "auc_diff": 0.0, "z": 0.0, "p_value": 1.0}
    z = (auc_a - auc_b) / np.sqrt(var_diff)
    return {
        "auc_a": auc_a,
        "auc_b": auc_b,
        "auc_diff": auc_a - auc_b,
        "z": z,
        "p_value": 2 * stats.norm.sf(abs(z)),
    }
