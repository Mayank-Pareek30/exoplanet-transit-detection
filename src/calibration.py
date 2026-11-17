"""
calibration.py
--------------
Calibration and operating-point analysis.

A classifier used for candidate screening does not simply output a label: its
score is read as a confidence, and astronomers triage by that confidence. A
model can rank well (high ROC-AUC) yet be badly calibrated, so that "0.9" does
not mean "90% of these are planets". This module measures that gap and helps
choose a defensible decision threshold rather than accepting 0.5 by default.
"""

from typing import Dict, Tuple

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (brier_score_loss, precision_recall_curve,
                             f1_score, precision_score, recall_score)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def expected_calibration_error(y_true: np.ndarray,
                               y_prob: np.ndarray,
                               n_bins: int = 10) -> float:
    """Expected Calibration Error: mean |confidence - accuracy| across bins.

    0 is perfect. Values above ~0.05 indicate the scores should not be read as
    probabilities without recalibration.
    """
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(y_true)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_prob > lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def calibration_report(y_true: np.ndarray,
                       probabilities: Dict[str, np.ndarray],
                       n_bins: int = 10) -> "pd.DataFrame":
    """Brier score and ECE for every model."""
    import pandas as pd
    rows = []
    for name, p in probabilities.items():
        rows.append({
            "Model": name,
            "Brier score": round(float(brier_score_loss(y_true, p)), 4),
            "ECE": round(expected_calibration_error(y_true, p, n_bins), 4),
        })
    return pd.DataFrame(rows).sort_values("Brier score").reset_index(drop=True)


def plot_reliability(y_true: np.ndarray,
                     probabilities: Dict[str, np.ndarray],
                     path: str = "figures/calibration.png",
                     n_bins: int = 10):
    """Reliability diagram: predicted confidence against observed frequency."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    for name, p in probabilities.items():
        frac_pos, mean_pred = calibration_curve(y_true, p, n_bins=n_bins,
                                                strategy="uniform")
        plt.plot(mean_pred, frac_pos, "o-", label=name, alpha=0.85)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed fraction of planets")
    plt.title("Reliability Diagram")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


# ---------------------------------------------------------------------------
# Operating point selection
# ---------------------------------------------------------------------------
def optimal_threshold(y_true: np.ndarray,
                      y_prob: np.ndarray,
                      objective: str = "f1") -> Dict[str, float]:
    """Find the decision threshold that maximises the chosen objective.

    The default 0.5 threshold is arbitrary. For a screening instrument the
    sensible operating point often favours recall, because a missed planet is
    lost entirely while a false positive only costs follow-up time.

    objective:
      'f1'          - balanced
      'recall_at_precision' - highest recall subject to precision >= 0.9
    """
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns one more prec/rec than thresholds
    prec, rec = prec[:-1], rec[:-1]

    if objective == "recall_at_precision":
        ok = prec >= 0.9
        if ok.any():
            i = int(np.argmax(np.where(ok, rec, -1)))
        else:
            i = int(np.argmax(prec))
    else:
        f1s = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
        i = int(np.argmax(f1s))

    t = float(thr[i])
    y_pred = (y_prob >= t).astype(int)
    return {
        "threshold": t,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def threshold_comparison(y_true: np.ndarray,
                         probabilities: Dict[str, np.ndarray]) -> "pd.DataFrame":
    """Compare default 0.5 against the F1-optimal threshold for each model."""
    import pandas as pd
    rows = []
    for name, p in probabilities.items():
        default_pred = (p >= 0.5).astype(int)
        best = optimal_threshold(y_true, p, "f1")
        rows.append({
            "Model": name,
            "F1 @ 0.5": round(float(f1_score(y_true, default_pred, zero_division=0)), 3),
            "Best threshold": round(best["threshold"], 3),
            "F1 @ best": round(best["f1"], 3),
            "Gain": round(best["f1"] - f1_score(y_true, default_pred, zero_division=0), 3),
        })
    return pd.DataFrame(rows)
