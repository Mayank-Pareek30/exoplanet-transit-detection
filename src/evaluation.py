"""
evaluation.py
-------------
Shared evaluation utilities so classical and deep models are compared on the
same footing: precision, recall, F1, ROC-AUC, confusion matrix.

Includes plotting helpers and a results-table writer.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve,
)


def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    """Return the core metrics as a dict. Accuracy is intentionally omitted
    as the headline because of the 1:99 imbalance."""
    out = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        out["roc_auc"] = roc_auc_score(y_true, y_prob)
    out["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    return out


def results_table(results: dict) -> pd.DataFrame:
    """Turn a {model_name: metrics_dict} mapping into a tidy DataFrame."""
    rows = []
    for name, m in results.items():
        # metrics may be (mean, std) tuples from k-fold, or scalars
        def val(x):
            return x[0] if isinstance(x, (tuple, list)) else x
        rows.append({
            "Model": name,
            "Precision": round(val(m.get("precision", 0)), 3),
            "Recall": round(val(m.get("recall", 0)), 3),
            "F1": round(val(m.get("f1", 0)), 3),
            "ROC-AUC": round(val(m.get("roc_auc", 0)), 3),
        })
    return pd.DataFrame(rows).sort_values("F1", ascending=False)


def save_results(results: dict, path: str = "results/comparison.csv"):
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df = results_table(results)
    df.to_csv(path, index=False)
    print(f"Saved results table to {path}")
    return df


def plot_roc(models_probs: dict, y_true, path: str = "results/roc.png"):
    """models_probs: {name: y_prob}. Saves an overlaid ROC plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 6))
    for name, y_prob in models_probs.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Model Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"Saved ROC plot to {path}")
