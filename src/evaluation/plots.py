"""Evaluation figures (ROC/PR, confusion, calibration, CIs, importance, sensitivity)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (auc, average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_curve)

# Models get a fixed colour each; feature sets go light -> dark blue.
MODEL_COLORS = ["#257de0", "#f47230", "#00b894", "#f2a600",
                "#ee7fac", "#008a17", "#6a4ce2", "#ec4647"]
SET_COLORS = {"Set 0": "#9cc3f0", "Set A": "#4f94e6", "Set B": "#0f4c99"}
GREY = "#8c8c8c"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
    "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 2,
})


def model_colors(models) -> dict:
    """Fixed colour for each model, in the order given."""
    return {m: MODEL_COLORS[i] for i, m in enumerate(models)}


def save(fig, path: Path):
    """Save a figure (300 dpi, white background), creating the folder if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    return path


def plot_roc_pr_by_set(preds: pd.DataFrame, models, sets):
    """ROC (top) and PR (bottom) curves per model, one line per feature set."""
    fig, axes = plt.subplots(2, len(models), figsize=(4 * len(models), 7.5), squeeze=False)
    prevalence = preds.groupby(["feature_set", "model"]).y_true.mean().iloc[0]

    for j, m in enumerate(models):
        ax_roc, ax_pr = axes[0, j], axes[1, j]
        for s in sets:
            g = preds[(preds.model == m) & (preds.feature_set == s)]
            fpr, tpr, _ = roc_curve(g.y_true, g.y_proba)
            prec, rec, _ = precision_recall_curve(g.y_true, g.y_proba)
            ap = average_precision_score(g.y_true, g.y_proba)
            ax_roc.plot(fpr, tpr, color=SET_COLORS[s], label=f"{s}  AUC {auc(fpr, tpr):.3f}")
            ax_pr.plot(rec, prec, color=SET_COLORS[s], label=f"{s}  AP {ap:.3f}")
        ax_roc.plot([0, 1], [0, 1], ls="--", lw=1, color=GREY, label="No skill")
        ax_pr.axhline(prevalence, ls="--", lw=1, color=GREY, label=f"No skill ({prevalence:.2f})")
        ax_roc.set(title=m, xlabel="False positive rate", ylabel="True positive rate" if j == 0 else "")
        ax_pr.set(xlabel="Recall", ylabel="Precision" if j == 0 else "", ylim=(0.3, 1.0))
        ax_roc.legend(loc="lower right"); ax_pr.legend(loc="upper right")
    fig.suptitle("Test-set ROC (top) and precision-recall (bottom) curves by feature set", y=1.0)
    fig.tight_layout()
    return fig


def plot_roc_pr_by_model(preds: pd.DataFrame, models, feature_set: str):
    """ROC and PR curves of every model on one feature set."""
    colors = model_colors(models)
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(10, 4.3))
    sub = preds[preds.feature_set == feature_set]
    for m in models:
        g = sub[sub.model == m]
        fpr, tpr, _ = roc_curve(g.y_true, g.y_proba)
        prec, rec, _ = precision_recall_curve(g.y_true, g.y_proba)
        ax_roc.plot(fpr, tpr, color=colors[m], label=f"{m}  AUC {auc(fpr, tpr):.3f}")
        ax_pr.plot(rec, prec, color=colors[m], label=m)
    ax_roc.plot([0, 1], [0, 1], ls="--", lw=1, color=GREY, label="No skill")
    ax_pr.axhline(sub.y_true.mean(), ls="--", lw=1, color=GREY, label="No skill")
    ax_roc.set(title=f"ROC curve - {feature_set}", xlabel="False positive rate", ylabel="True positive rate")
    ax_pr.set(title=f"Precision-recall curve - {feature_set}", xlabel="Recall", ylabel="Precision", ylim=(0.3, 1.0))
    ax_roc.legend(loc="lower right"); ax_pr.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_confusion_matrices(preds: pd.DataFrame, models, feature_set: str, thresholds: dict):
    """Confusion matrices at tau = 0.5 (top) and each model's F1-optimal tau (bottom)."""
    fig, axes = plt.subplots(2, len(models), figsize=(3.6 * len(models), 6.6), squeeze=False)
    sub = preds[preds.feature_set == feature_set]
    for j, m in enumerate(models):
        g = sub[sub.model == m]
        for i, (name, tau) in enumerate([("default", 0.5), ("F1-optimal", thresholds[m])]):
            cm = confusion_matrix(g.y_true, (g.y_proba >= tau).astype(int))
            pct = cm / cm.sum(axis=1, keepdims=True)
            ax = axes[i, j]
            ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
            for (r, c), v in np.ndenumerate(cm):
                ax.text(c, r, f"{v}\n({pct[r, c]:.0%})", ha="center", va="center",
                        color="white" if pct[r, c] > 0.55 else "#191919", fontsize=10)
            ax.set_xticks([0, 1], ["Low risk", "High risk"])
            ax.set_yticks([0, 1], ["Low risk", "High risk"])
            ax.grid(False)
            ax.set(xlabel="Predicted", ylabel="True" if j == 0 else "",
                   title=f"{m}\n{name} τ = {tau:.2f}")
    fig.suptitle(f"Confusion matrices on the test set - {feature_set}", y=1.0)
    fig.tight_layout()
    return fig


def plot_calibration(preds: pd.DataFrame, models, feature_set: str, brier: dict):
    """Reliability curves: predicted probability vs observed rate of high risk."""
    colors = model_colors(models)
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    sub = preds[preds.feature_set == feature_set]
    for m in models:
        g = sub[sub.model == m]
        frac_pos, mean_pred = calibration_curve(g.y_true, g.y_proba, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", ms=5, color=colors[m],
                label=f"{m}  (Brier {brier[m]:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=GREY, label="Perfect calibration")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability",
           ylabel="Observed fraction high risk", title=f"Calibration - {feature_set} (10 quantile bins)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def plot_metric_forest(table: pd.DataFrame, metric: str, models, sets, baseline: float | None = None):
    """Estimate and 95% CI of one metric for every model and feature set."""
    colors = model_colors(models)
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(models) * len(sets) + 1.2))
    y, labels = 0, []
    for s in sets:
        for m in models:
            r = table[(table.model == m) & (table.feature_set == s) & (table.metric == metric)].iloc[0]
            ax.errorbar(r.estimate, y, xerr=[[r.estimate - r.ci_low], [r.ci_high - r.estimate]],
                        fmt="o", ms=7, color=colors[m], capsize=3, lw=1.5)
            ax.text(r.ci_high + 0.004, y, f"{r.estimate:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]",
                    va="center", fontsize=8, color="#444444")
            labels.append(f"{s} | {m}")
            y += 1
        y += 0.5
    ticks = [i + 0.5 * (i // len(models)) for i in range(len(labels))]
    ax.set_yticks(ticks, labels)
    ax.invert_yaxis()
    name = {"roc_auc": "ROC-AUC", "pr_auc": "PR-AUC"}.get(metric, metric)
    # No-skill value goes in the label; a line that far left would squash the CIs.
    note = "" if baseline is None else f"   (no-skill = {baseline:.2f})"
    ax.set(xlabel=f"Test {name} with 95% bootstrap CI (B = 1000){note}",
           title=f"Test-set {name} by model and feature set")
    ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1] + 0.03)
    fig.tight_layout()
    return fig


def plot_importance_panels(odds: pd.DataFrame, gini: pd.DataFrame, perm: pd.DataFrame,
                           top: int = 15, titles=("", "", "")):
    """Top features by LR odds ratio, RF Gini importance and permutation importance."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 0.32 * top + 1.5))

    o = odds.head(top).iloc[::-1]
    axes[0].barh(o.feature, o.odds_ratio - 1, left=1,
                 color=np.where(o.odds_ratio >= 1, "#f47230", "#257de0"), height=0.7)
    axes[0].axvline(1, color=GREY, lw=1)
    axes[0].set(xlabel="Odds ratio per +1 SD  (orange raises risk, blue lowers)", title=titles[0])

    g = gini.head(top).iloc[::-1]
    axes[1].barh(g.feature, g.importance, color="#4f94e6", height=0.7)
    axes[1].set(xlabel="Mean decrease in Gini impurity", title=titles[1])

    p = perm.head(top).iloc[::-1]
    axes[2].barh(p.feature, p.importance_mean, xerr=p.importance_std, color="#0f4c99",
                 height=0.7, error_kw={"lw": 1, "ecolor": GREY})
    axes[2].axvline(0, color=GREY, lw=1)
    axes[2].set(xlabel="Drop in test ROC-AUC when shuffled (R = 20)", title=titles[2])

    for ax in axes:
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def plot_selection_stability(freq: pd.DataFrame, top: int = 20):
    """How often each feature is selected by the filter and L1 methods."""
    order = freq[["filter_mi", "embedded_l1"]].mean(axis=1).sort_values(ascending=False, kind="stable")
    f = freq.loc[order.index].head(top).iloc[::-1]
    y = np.arange(len(f))
    fig, ax = plt.subplots(figsize=(7, 0.3 * len(f) + 1.6))
    ax.barh(y + 0.2, f.filter_mi, height=0.38, color="#257de0", label="Filter (corr + mutual information)")
    ax.barh(y - 0.2, f.embedded_l1, height=0.38, color="#f47230", label="Embedded (L1 logistic)")
    ax.set_yticks(y, f.index)
    ax.set(xlim=(0, 1.05), xlabel="Selection frequency across 5 training folds")
    ax.set_title("Feature-selection stability (Set B)", pad=28)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8)
    fig.tight_layout()
    return fig


def plot_sensitivity(e4: pd.DataFrame, e5: pd.DataFrame, e6: pd.DataFrame, models, main_auc: dict):
    """Three panels: E4 survey weights, E5 cross-cycle transfer, E6 feature-group ablation."""
    colors = model_colors(models)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), gridspec_kw={"width_ratios": [1, 1, 1.25]})
    x = np.arange(len(models))
    short = [m.replace(" ", "\n", 1) for m in models]      # two-line tick labels

    # E4
    ax = axes[0]
    sub = e4[e4.test_evaluation == "unweighted"].set_index(["model", "training"]).roc_auc
    for k, (training, shade) in enumerate([("unweighted", "#9cc3f0"), ("survey-weighted", "#0f4c99")]):
        vals = [sub[(m, training)] for m in models]
        bars = ax.bar(x + (k - 0.5) * 0.38, vals, 0.36, color=shade, label=f"{training} training")
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    ax.set_xticks(x, short)
    ax.set(ylim=(0.6, 0.85), ylabel="Test ROC-AUC", title="E4  Survey-weighted training (Set B)")
    ax.legend(loc="upper left")

    # E5 (grey dashes = main random-split result)
    ax = axes[1]
    directions = list(e5.direction.unique())
    shades = ["#4f94e6", "#0f4c99"]
    for k, d in enumerate(directions):
        vals = [e5[(e5.model == m) & (e5.direction == d)].roc_auc.iloc[0] for m in models]
        bars = ax.bar(x + (k - 0.5) * 0.38, vals, 0.36, color=shades[k], label=d)
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    for i, m in enumerate(models):
        ax.plot([i - 0.4, i + 0.4], [main_auc[m]] * 2, color=GREY, ls="--", lw=1.2,
                label="random 80/20 split" if i == 0 else None)
    ax.set_xticks(x, short)
    ax.set(ylim=(0.6, 0.85), ylabel="Test ROC-AUC", title="E5  Cross-cycle transfer (Set B)")
    ax.legend(loc="upper left")

    # E6
    ax = axes[2]
    groups = [g for g in e6.removed_group.unique() if not g.startswith("(none")]
    y = np.arange(len(groups))
    h = 0.8 / len(models)
    for k, m in enumerate(models):
        d = e6[e6.model == m].set_index("removed_group").loc[groups, "delta_auc"]
        ax.barh(y + (k - (len(models) - 1) / 2) * h, d.values, h * 0.9, color=colors[m], label=m)
    ax.axvline(0, color=GREY, lw=1)
    ax.set_yticks(y, [f"without\n{g}" for g in groups])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="Δ CV ROC-AUC vs full Set B  (< 0: group was useful)")
    ax.set_title("E6  Feature-group ablation (training CV)", pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(models), fontsize=8)

    fig.tight_layout()
    return fig
