"""
ablation.py
-----------
Ablation study for the preprocessing pipeline.

During development two preprocessing steps turned out to be decisive: without
phase alignment and baseline centring the deep models sat at chance. An ablation
turns that anecdote into evidence by removing one component at a time and
measuring the cost, which is what lets the paper claim these steps are necessary
rather than merely conventional.

Conditions tested:
  full            - centred, aligned curves (the pipeline as used)
  no_centring     - baseline left at ~1.0 instead of 0
  no_alignment    - each curve circularly shifted by a random phase
  shuffled_input  - flux values permuted within each curve (destroys shape,
                    preserves the value distribution) - a sanity control
"""

from typing import Callable, Dict, List

import numpy as np


# ---------------------------------------------------------------------------
# Perturbations
# ---------------------------------------------------------------------------
def remove_centring(X: np.ndarray) -> np.ndarray:
    """Undo baseline centring: shift the curves back to a ~1.0 baseline."""
    return X + 1.0


def remove_alignment(X: np.ndarray, seed: int = 42) -> np.ndarray:
    """Destroy phase alignment by rolling each curve by a random offset.

    The transit still exists and the values are untouched; only its position
    becomes unpredictable. This isolates the value of alignment from the value
    of the signal itself.
    """
    rng = np.random.default_rng(seed)
    out = np.empty_like(X)
    for i in range(len(X)):
        shift = int(rng.integers(0, X.shape[1]))
        out[i] = np.roll(X[i], shift, axis=0)
    return out


def shuffle_within_curve(X: np.ndarray, seed: int = 42) -> np.ndarray:
    """Permute flux values inside each curve.

    Every summary statistic of the distribution (mean, std, min, percentiles) is
    preserved while all temporal structure is destroyed. If a model still
    performs well here it is relying on distributional cues rather than transit
    shape - an important control for interpreting the classical baselines.
    """
    rng = np.random.default_rng(seed)
    out = np.empty_like(X)
    for i in range(len(X)):
        idx = rng.permutation(X.shape[1])
        out[i] = X[i][idx]
    return out


ABLATIONS: Dict[str, Callable] = {
    "full": lambda X, seed=42: X,
    "no_centring": lambda X, seed=42: remove_centring(X),
    "no_alignment": lambda X, seed=42: remove_alignment(X, seed),
    "shuffled_input": lambda X, seed=42: shuffle_within_curve(X, seed),
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_ablation(train_fn: Callable,
                 X_train: np.ndarray, y_train: np.ndarray,
                 X_test: np.ndarray, y_test: np.ndarray,
                 conditions: List[str] = None,
                 seed: int = 42) -> "pd.DataFrame":
    """Train and evaluate under each ablation condition.

    `train_fn(X_tr, y_tr, X_te, y_te) -> dict of metrics` keeps this agnostic to
    which model is being tested, so the same study can be run for the CNN, the
    hybrid, or a classical model.

    The same perturbation is applied to train and test: the question is whether
    the pipeline step matters, not whether the model tolerates a shifted test
    distribution.
    """
    import pandas as pd
    conditions = conditions or list(ABLATIONS)
    rows = []

    for name in conditions:
        fn = ABLATIONS[name]
        Xtr, Xte = fn(X_train, seed), fn(X_test, seed)
        metrics = train_fn(Xtr, y_train, Xte, y_test)
        rows.append({"Condition": name, **{k: round(float(v), 4)
                                           for k, v in metrics.items()
                                           if isinstance(v, (int, float))}})

    df = pd.DataFrame(rows)

    # express each condition as the change relative to the full pipeline
    if "full" in df["Condition"].values and "roc_auc" in df.columns:
        base = float(df.loc[df["Condition"] == "full", "roc_auc"].iloc[0])
        df["Δ ROC-AUC"] = (df["roc_auc"] - base).round(4)
    return df


def interpret_ablation(df: "pd.DataFrame") -> List[str]:
    """Turn the ablation table into plain statements for the write-up."""
    notes = []
    if "Δ ROC-AUC" not in df.columns:
        return notes
    for _, r in df.iterrows():
        cond, delta = r["Condition"], r["Δ ROC-AUC"]
        if cond == "full":
            continue
        if delta < -0.05:
            notes.append(f"Removing '{cond.replace('_', ' ')}' costs "
                         f"{abs(delta):.3f} ROC-AUC: this step is necessary.")
        elif delta < -0.01:
            notes.append(f"Removing '{cond.replace('_', ' ')}' costs "
                         f"{abs(delta):.3f} ROC-AUC: a modest but real effect.")
        else:
            notes.append(f"Removing '{cond.replace('_', ' ')}' changes ROC-AUC "
                         f"by {delta:+.3f}: little to no effect.")
    return notes
