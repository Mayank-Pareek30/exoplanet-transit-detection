"""
statistics.py
-------------
Statistical machinery for comparing models honestly.

A single-run difference of 0.001 in F1 tells you nothing: run-to-run variance
in this study is around +/- 0.005 AUC. This module provides the tools needed to
say whether an observed difference is real:

  - bootstrap confidence intervals on any metric
  - McNemar's test for paired classifier comparison on the same test set
  - DeLong-style comparison of ROC curves via bootstrap
  - aggregation of multi-seed runs into mean +/- std
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, f1_score


# ---------------------------------------------------------------------------
# Multi-seed aggregation
# ---------------------------------------------------------------------------
def aggregate_seeds(runs: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Combine per-seed results into mean and standard deviation.

    `runs` is a list (one entry per seed) of {model_name: {metric: value}}.
    Returns {model_name: {metric: (mean, std)}}.
    """
    if not runs:
        return {}
    models = runs[0].keys()
    out = {}
    for m in models:
        metrics = runs[0][m].keys()
        out[m] = {}
        for metric in metrics:
            vals = [r[m][metric] for r in runs if m in r and metric in r[m]]
            vals = [v for v in vals if isinstance(v, (int, float))]
            if vals:
                out[m][metric] = (float(np.mean(vals)), float(np.std(vals)))
    return out


def format_mean_std(agg: Dict[str, Dict[str, Tuple[float, float]]],
                    metrics: Sequence[str] = ("accuracy", "precision", "recall",
                                              "f1", "roc_auc")) -> "pd.DataFrame":
    """Render aggregated results as a table of 'mean ± std' strings."""
    import pandas as pd
    rows = []
    for model, mm in agg.items():
        row = {"Model": model}
        for metric in metrics:
            if metric in mm:
                mean, std = mm[metric]
                row[metric.replace("_", "-").upper()] = f"{mean:.3f} ± {std:.3f}"
        rows.append(row)
    df = pd.DataFrame(rows)
    # sort by the mean F1, descending
    if "F1" in df.columns:
        df["_sort"] = df["F1"].str.split(" ").str[0].astype(float)
        df = df.sort_values("_sort", ascending=False).drop(columns="_sort")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------
def bootstrap_metric(y_true: np.ndarray,
                     y_score: np.ndarray,
                     metric: str = "roc_auc",
                     n_boot: int = 2000,
                     alpha: float = 0.05,
                     seed: int = 42) -> Tuple[float, float, float]:
    """Bootstrap a metric and return (point_estimate, lower, upper).

    Resamples the test set with replacement. This gives a confidence interval
    that reflects uncertainty from the finite test set, which a single number
    hides.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)

    def compute(yt, ys):
        if len(np.unique(yt)) < 2:
            return np.nan
        if metric == "roc_auc":
            return roc_auc_score(yt, ys)
        return f1_score(yt, (ys >= 0.5).astype(int), zero_division=0)

    point = compute(y_true, y_score)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = compute(y_true[idx], y_score[idx])
        if not np.isnan(v):
            boots.append(v)
    if not boots:
        return point, np.nan, np.nan
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return float(point), lo, hi


# ---------------------------------------------------------------------------
# Paired significance tests
# ---------------------------------------------------------------------------
def mcnemar_test(y_true: np.ndarray,
                 y_pred_a: np.ndarray,
                 y_pred_b: np.ndarray) -> Dict[str, float]:
    """McNemar's test for two classifiers evaluated on the same test set.

    Builds the 2x2 contingency of disagreements:
        b = A correct, B wrong
        c = A wrong,   B correct
    Under the null hypothesis (equal error rates) b and c are exchangeable.
    Uses the exact binomial test, which is appropriate when b+c is small.

    Returns the counts, the statistic and the p-value.
    """
    y_true = np.asarray(y_true)
    a_ok = (np.asarray(y_pred_a) == y_true)
    b_ok = (np.asarray(y_pred_b) == y_true)

    b = int(np.sum(a_ok & ~b_ok))     # only A correct
    c = int(np.sum(~a_ok & b_ok))     # only B correct

    if b + c == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0}

    # exact binomial (two-sided)
    p = float(stats.binomtest(b, b + c, 0.5).pvalue)
    statistic = (abs(b - c) - 1) ** 2 / (b + c)   # continuity-corrected chi2
    return {"b": b, "c": c, "statistic": float(statistic), "p_value": p}


def bootstrap_auc_difference(y_true: np.ndarray,
                             score_a: np.ndarray,
                             score_b: np.ndarray,
                             n_boot: int = 2000,
                             seed: int = 42) -> Dict[str, float]:
    """Bootstrap the difference in ROC-AUC between two models.

    Resamples the test set and recomputes AUC_a - AUC_b each time. If the
    resulting confidence interval excludes zero, the difference is significant
    at that level. This is the paired analogue of DeLong's test and avoids the
    normality assumption.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    score_a, score_b = np.asarray(score_a), np.asarray(score_b)
    n = len(y_true)

    obs = roc_auc_score(y_true, score_a) - roc_auc_score(y_true, score_b)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        diffs.append(roc_auc_score(yt, score_a[idx]) - roc_auc_score(yt, score_b[idx]))

    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # two-sided p: proportion of bootstrap diffs on the other side of zero
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"difference": float(obs), "ci_low": float(lo), "ci_high": float(hi),
            "p_value": float(min(p, 1.0)),
            "significant": bool(lo > 0 or hi < 0)}


def pairwise_comparison_table(y_true: np.ndarray,
                              predictions: Dict[str, np.ndarray],
                              probabilities: Dict[str, np.ndarray]) -> "pd.DataFrame":
    """Run McNemar plus bootstrap AUC comparison for every pair of models."""
    import pandas as pd
    names = list(predictions.keys())
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            mc = mcnemar_test(y_true, predictions[a], predictions[b])
            bd = bootstrap_auc_difference(y_true, probabilities[a],
                                          probabilities[b], n_boot=1000)
            rows.append({
                "Model A": a,
                "Model B": b,
                "McNemar p": round(mc["p_value"], 4),
                "AUC diff (A-B)": round(bd["difference"], 4),
                "95% CI": f"[{bd['ci_low']:.3f}, {bd['ci_high']:.3f}]",
                "Significant": "yes" if bd["significant"] else "no",
            })
    return pd.DataFrame(rows)
