#Unit tests for evaluation package

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.evaluation import bootstrap, metrics
from src.evaluation.explainability import rank_table


def _toy(n=400, seed=0, signal=1.5):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    p_good = 1 / (1 + np.exp(-(signal * (2 * y - 1) + rng.normal(0, 1, n))))
    p_noise = rng.uniform(0, 1, n)
    return y, p_good, p_noise


def test_metrics_perfect_and_bounds():
    y = np.array([0, 0, 1, 1])
    m = metrics.classification_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["roc_auc"] == m["pr_auc"] == m["f1"] == m["accuracy"] == 1.0
    assert m["brier"] == pytest.approx(np.mean([0.01, 0.04, 0.04, 0.01]))
    assert list(m) == metrics.METRIC_NAMES


def test_threshold_changes_only_threshold_metrics():
    y, p, _ = _toy()
    lo = metrics.classification_metrics(y, p, 0.2)
    hi = metrics.classification_metrics(y, p, 0.8)
    assert lo["recall"] > hi["recall"]
    for k in ("roc_auc", "pr_auc", "brier"):
        assert lo[k] == hi[k]


def test_delong_matches_sklearn_auc_and_detects_difference():
    y, good, noise = _toy()
    res = metrics.delong_test(y, good, noise)
    assert res["auc_a"] == pytest.approx(roc_auc_score(y, good))
    assert res["auc_b"] == pytest.approx(roc_auc_score(y, noise))
    assert res["p_value"] < 0.001


def test_delong_identical_models_p_is_one():
    y, good, _ = _toy()
    assert metrics.delong_test(y, good, good)["p_value"] == 1.0


def test_bootstrap_ci_contains_estimate_and_is_reproducible():
    y, p, _ = _toy()
    a = bootstrap.bootstrap_metrics(y, p, n_boot=200)
    b = bootstrap.bootstrap_metrics(y, p, n_boot=200)
    pd.testing.assert_frame_equal(a, b)
    assert ((a.ci_low <= a.estimate) & (a.estimate <= a.ci_high)).all()


def test_bootstrap_resamples_always_have_both_classes():
    y = np.array([0] * 48 + [1] * 2)
    for idx in bootstrap.bootstrap_indices(y, n_boot=300, seed=1):
        assert set(y[idx]) == {0, 1}


def test_paired_difference_sign():
    y, good, noise = _toy()
    d = bootstrap.paired_bootstrap_difference(y, noise, good, n_boot=200)
    assert d["difference"] > 0 and d["ci_low"] > 0 and d["p_improved"] == 1.0


def test_rank_table_uses_absolute_values():
    r = rank_table(a=pd.Series({"x": -3.0, "y": 1.0}), b=pd.Series({"x": 0.9, "y": 0.1}))
    assert r.index[0] == "x" and r.loc["x", "mean_rank"] == 1.0
