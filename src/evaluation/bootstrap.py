"""Bootstrap 95% confidence intervals for test-set metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.metrics import METRIC_NAMES, classification_metrics

N_BOOT = 1000
SEED = 42


def bootstrap_indices(y_true, n_boot: int = N_BOOT, seed: int = SEED) -> list[np.ndarray]:
    """Row indices for n_boot resamples (with replacement); skips one-class resamples."""
    y_true = np.asarray(y_true)
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n_boot:
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].min() != y_true[idx].max():
            out.append(idx)
    return out


def bootstrap_metrics(y_true, y_proba, threshold: float = 0.5,
                      n_boot: int = N_BOOT, seed: int = SEED) -> pd.DataFrame:
    """Estimate and 95% percentile CI (ci_low, ci_high) for every metric."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)

    point = classification_metrics(y_true, y_proba, threshold)
    boots = pd.DataFrame([
        classification_metrics(y_true[idx], y_proba[idx], threshold)
        for idx in bootstrap_indices(y_true, n_boot, seed)
    ])
    return pd.DataFrame({
        "estimate": pd.Series(point),
        "ci_low": boots.quantile(0.025),
        "ci_high": boots.quantile(0.975),
    }).loc[METRIC_NAMES]


def paired_bootstrap_difference(y_true, proba_a, proba_b, metric: str = "roc_auc",
                                n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """metric(b) - metric(a) with a 95% CI; both models use the same resamples."""
    y_true = np.asarray(y_true)
    a = np.asarray(proba_a, dtype=float)
    b = np.asarray(proba_b, dtype=float)

    def score(idx, p):
        return classification_metrics(y_true[idx], p[idx])[metric]

    full = np.arange(len(y_true))
    diffs = np.array([score(idx, b) - score(idx, a)
                      for idx in bootstrap_indices(y_true, n_boot, seed)])
    return {
        "difference": score(full, b) - score(full, a),
        "ci_low": float(np.quantile(diffs, 0.025)),
        "ci_high": float(np.quantile(diffs, 0.975)),
        "p_improved": float((diffs > 0).mean()),   # share of resamples where b beats a
    }
