"""
error_analysis.py
-----------------
Analysis of where the models fail.

Reporting aggregate metrics answers "how well"; error analysis answers "how it
fails", which is what distinguishes a critical evaluation from a results dump.
Two questions are addressed here:

  1. Are the errors systematic? Do false negatives share a property (shallower
     transits, noisier curves) that explains them?
  2. Do the models fail on the same objects, or on different ones? Substantial
     disagreement implies an ensemble would help, and tells us the models have
     learned genuinely different decision rules.
"""

from typing import Dict, List

import numpy as np


# ---------------------------------------------------------------------------
# Characterising errors
# ---------------------------------------------------------------------------
def curve_properties(X: np.ndarray) -> Dict[str, np.ndarray]:
    """Simple physical descriptors used to characterise each light curve."""
    x = X[..., 0] if X.ndim == 3 else X
    centre = x.shape[1] // 2
    half = max(1, int(x.shape[1] * 0.025))
    return {
        "transit_depth": -np.min(x, axis=1),                     # positive = deeper
        "noise_level": np.std(x, axis=1),
        "central_depth": -np.min(x[:, centre - half: centre + half], axis=1),
        "snr_proxy": (-np.min(x, axis=1)) / np.clip(np.std(x, axis=1), 1e-9, None),
    }


def error_profile(X: np.ndarray,
                  y_true: np.ndarray,
                  y_pred: np.ndarray) -> "pd.DataFrame":
    """Compare curve properties across the four outcome groups.

    If false negatives have systematically lower SNR than true positives, the
    model is failing on genuinely harder objects rather than behaving randomly,
    which is a much more defensible story.
    """
    import pandas as pd
    props = curve_properties(X)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)

    groups = {
        "True positive": (y_true == 1) & (y_pred == 1),
        "False negative": (y_true == 1) & (y_pred == 0),
        "True negative": (y_true == 0) & (y_pred == 0),
        "False positive": (y_true == 0) & (y_pred == 1),
    }

    rows = []
    for gname, mask in groups.items():
        if mask.sum() == 0:
            continue
        row = {"Group": gname, "N": int(mask.sum())}
        for pname, vals in props.items():
            row[pname.replace("_", " ")] = round(float(np.mean(vals[mask])), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def hardest_cases(X: np.ndarray,
                  y_true: np.ndarray,
                  y_prob: np.ndarray,
                  n: int = 10) -> Dict[str, np.ndarray]:
    """Indices of the most confidently wrong predictions.

    These are the objects worth inspecting by eye and discussing in the paper:
    confident mistakes reveal the model's blind spots far better than borderline
    cases do.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    conf_wrong_pos = np.where(y_true == 1)[0]          # planets scored low
    conf_wrong_neg = np.where(y_true == 0)[0]          # non-planets scored high
    fn = conf_wrong_pos[np.argsort(y_prob[conf_wrong_pos])][:n]
    fp = conf_wrong_neg[np.argsort(-y_prob[conf_wrong_neg])][:n]
    return {"worst_false_negatives": fn, "worst_false_positives": fp}


# ---------------------------------------------------------------------------
# Model agreement
# ---------------------------------------------------------------------------
def agreement_matrix(predictions: Dict[str, np.ndarray]) -> "pd.DataFrame":
    """Pairwise fraction of test objects on which two models agree."""
    import pandas as pd
    names = list(predictions)
    M = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            M[i, j] = float(np.mean(predictions[a] == predictions[b]))
    return pd.DataFrame(M.round(3), index=names, columns=names)


def complementarity(y_true: np.ndarray,
                    predictions: Dict[str, np.ndarray]) -> "pd.DataFrame":
    """How often each pair of models errs on different objects.

    High complementarity means an ensemble has room to improve on either model
    alone; near-zero means the models share the same blind spots and combining
    them would add little.
    """
    import pandas as pd
    names = list(predictions)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            wrong_a = predictions[a] != y_true
            wrong_b = predictions[b] != y_true
            both = float(np.mean(wrong_a & wrong_b))
            either = float(np.mean(wrong_a | wrong_b))
            rows.append({
                "Model A": a, "Model B": b,
                "Both wrong": round(both, 3),
                "Either wrong": round(either, 3),
                "Complementarity": round(either - both, 3),
            })
    return pd.DataFrame(rows).sort_values("Complementarity", ascending=False)


def oracle_ensemble_ceiling(y_true: np.ndarray,
                            predictions: Dict[str, np.ndarray]) -> float:
    """Accuracy achievable if we always picked the model that happens to be right.

    Not attainable in practice, but it bounds what any ensemble of these models
    could reach and so tells you whether pursuing an ensemble is worthwhile.
    """
    y_true = np.asarray(y_true)
    correct = np.zeros(len(y_true), dtype=bool)
    for p in predictions.values():
        correct |= (np.asarray(p) == y_true)
    return float(correct.mean())
