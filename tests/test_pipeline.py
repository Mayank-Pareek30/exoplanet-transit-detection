"""
test_pipeline.py
----------------
Unit tests for the core pipeline.

These guard the parts of the pipeline where a silent error would corrupt every
downstream result: label mapping, curve centring, the statistical tests, and the
attribution focus metric. Run with:

    python -m pytest tests/ -v
    python tests/test_pipeline.py        # also works without pytest
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.statistics import (mcnemar_test, bootstrap_metric,
                            bootstrap_auc_difference, aggregate_seeds)
from src.explainability import transit_focus_score, focus_ratio
from src.calibration import expected_calibration_error, optimal_threshold
from src.error_analysis import (agreement_matrix, oracle_ensemble_ceiling,
                                curve_properties)
from src.ablation import remove_centring, remove_alignment, shuffle_within_curve


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def test_mcnemar_identical_models_not_significant():
    y = np.array([0, 1] * 50)
    pred = y.copy()
    r = mcnemar_test(y, pred, pred)
    assert r["b"] == 0 and r["c"] == 0
    assert r["p_value"] == 1.0


def test_mcnemar_detects_clear_difference():
    y = np.array([1] * 100)
    good = np.array([1] * 100)          # perfect
    bad = np.array([0] * 100)           # always wrong
    r = mcnemar_test(y, good, bad)
    assert r["p_value"] < 0.01


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    score = np.clip(y * 0.7 + rng.normal(0, 0.2, 300), 0, 1)
    point, lo, hi = bootstrap_metric(y, score, n_boot=200)
    assert lo <= point <= hi
    assert 0.0 <= lo and hi <= 1.0


def test_auc_difference_zero_for_same_scores():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    s = rng.random(200)
    r = bootstrap_auc_difference(y, s, s, n_boot=100)
    assert abs(r["difference"]) < 1e-9
    assert not r["significant"]


def test_aggregate_seeds_computes_mean_and_std():
    runs = [{"M": {"f1": 0.80}}, {"M": {"f1": 0.90}}]
    agg = aggregate_seeds(runs)
    mean, std = agg["M"]["f1"]
    assert abs(mean - 0.85) < 1e-9
    assert std > 0


# ---------------------------------------------------------------------------
# Explainability metrics
# ---------------------------------------------------------------------------
def test_uniform_attribution_gives_focus_ratio_near_one():
    a = np.ones(1000)
    assert abs(focus_ratio(a, 0.05) - 1.0) < 0.15


def test_centred_attribution_gives_high_focus_ratio():
    a = np.zeros(1000)
    a[490:510] = 1.0                     # all mass in the centre
    assert focus_ratio(a, 0.05) > 5.0


def test_focus_score_handles_all_zero_attribution():
    assert transit_focus_score(np.zeros(100)) == 0.0


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def test_perfect_calibration_has_low_ece():
    # scores equal to true probabilities
    rng = np.random.default_rng(2)
    p = rng.random(4000)
    y = (rng.random(4000) < p).astype(int)
    assert expected_calibration_error(y, p, n_bins=10) < 0.05


def test_optimal_threshold_beats_or_matches_default():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.5 + rng.normal(0.25, 0.25, 500), 0, 1)
    best = optimal_threshold(y, p, "f1")
    from sklearn.metrics import f1_score
    default = f1_score(y, (p >= 0.5).astype(int), zero_division=0)
    assert best["f1"] >= default - 1e-9


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------
def test_agreement_matrix_diagonal_is_one():
    preds = {"A": np.array([0, 1, 1, 0]), "B": np.array([0, 0, 1, 0])}
    M = agreement_matrix(preds)
    assert M.loc["A", "A"] == 1.0 and M.loc["B", "B"] == 1.0


def test_oracle_ceiling_at_least_best_single_model():
    y = np.array([1, 1, 0, 0])
    preds = {"A": np.array([1, 1, 1, 1]), "B": np.array([0, 0, 0, 0])}
    # A right on 2, B right on 2, oracle right on all 4
    assert oracle_ensemble_ceiling(y, preds) == 1.0


def test_curve_properties_detects_deeper_transit():
    shallow = np.zeros((1, 200, 1), dtype=np.float32)
    shallow[0, 95:105, 0] = -0.1
    deep = np.zeros((1, 200, 1), dtype=np.float32)
    deep[0, 95:105, 0] = -0.5
    assert (curve_properties(deep)["transit_depth"][0]
            > curve_properties(shallow)["transit_depth"][0])


# ---------------------------------------------------------------------------
# Ablation perturbations
# ---------------------------------------------------------------------------
def test_remove_centring_shifts_baseline():
    X = np.zeros((5, 100, 1), dtype=np.float32)
    assert np.allclose(remove_centring(X), 1.0)


def test_remove_alignment_preserves_values():
    rng = np.random.default_rng(4)
    X = rng.normal(0, 1, (10, 50, 1)).astype(np.float32)
    out = remove_alignment(X, seed=0)
    for i in range(len(X)):
        assert np.allclose(np.sort(X[i].ravel()), np.sort(out[i].ravel()))


def test_shuffle_preserves_distribution_destroys_order():
    rng = np.random.default_rng(5)
    X = rng.normal(0, 1, (4, 200, 1)).astype(np.float32)
    out = shuffle_within_curve(X, seed=0)
    for i in range(len(X)):
        assert np.allclose(np.sort(X[i].ravel()), np.sort(out[i].ravel()))
    assert not np.allclose(X, out)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Dual-view alignment
# ---------------------------------------------------------------------------
def test_dual_view_alignment_rejects_mismatch():
    """load_aligned_views must refuse to pair files whose labels disagree."""
    import tempfile, os
    import pandas as pd
    from src.dual_view import load_aligned_views

    n = 20
    g = pd.DataFrame(np.zeros((n + 2, 5)), columns=[str(i) for i in range(5)])
    g["label"] = ["CONFIRMED"] * (n + 2)
    # deliberately conflicting labels in the local file
    l = pd.DataFrame(np.zeros((n, 3)), columns=[str(i) for i in range(3)])
    l["label"] = ["FALSE POSITIVE"] * n

    d = tempfile.mkdtemp()
    gp, lp = os.path.join(d, "g.csv"), os.path.join(d, "l.csv")
    g.to_csv(gp, index=False); l.to_csv(lp, index=False)

    try:
        load_aligned_views(gp, lp, verbose=False)
    except AssertionError:
        return          # expected
    raise AssertionError("alignment check failed to reject mismatched files")



if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {len(tests)} total")
    sys.exit(1 if failed else 0)

