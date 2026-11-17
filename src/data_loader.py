"""
data_loader.py
--------------
Loads the Macedo & Zalewski (2024) Mendeley dataset files and reports the
real class balance, so you only apply imbalance handling if it is actually
needed.

Expected files (download from Mendeley 10.17632/wctcv34962.3):
    all_global.csv            -> global view: full phase-folded curve per star
    all_local.csv             -> local view: zoomed window around the transit
    lighkurve_KOI_dataset.csv  -> KOI catalogue: labels + tabular features

IMPORTANT: these light curves are ALREADY phase-folded, binned, normalised,
interpolated, and outlier-cleaned by the dataset authors. You do NOT need to
re-run phase_fold() on them — that step is done. Credit the authors (ref [5]).
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------
def map_disposition(series: pd.Series) -> pd.Series:
    """CONFIRMED -> 1, everything else (CANDIDATE / FALSE POSITIVE) -> 0."""
    return (series.astype(str).str.strip().str.upper() == "CONFIRMED").astype(int)


def report_balance(y: np.ndarray) -> dict:
    """Print and return the class balance, with a recommendation."""
    n = len(y)
    n_pos = int(y.sum())
    n_neg = n - n_pos
    pos_pct = 100 * n_pos / n if n else 0
    ratio = (n_neg / n_pos) if n_pos else float("inf")

    print(f"  Total samples : {n:,}")
    print(f"  Positives (1) : {n_pos:,}  ({pos_pct:.1f}%)")
    print(f"  Negatives (0) : {n_neg:,}  ({100-pos_pct:.1f}%)")
    print(f"  Ratio (neg:pos): {ratio:.1f} : 1")

    needs_handling = ratio >= 1.5 or ratio <= 0.67
    if needs_handling:
        print("  --> Imbalanced. Apply SMOTE + class weighting.")
    else:
        print("  --> Roughly balanced. Imbalance handling optional.")

    return {"n": n, "n_pos": n_pos, "n_neg": n_neg,
            "pos_pct": pos_pct, "ratio": ratio,
            "needs_handling": needs_handling}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_light_curves(global_path: str,
                      local_path: str | None = None,
                      koi_path: str | None = None,
                      label_col: str = "label",
                      id_col: str = "kepid"):
    """Load the global (and optional local) phase-folded light curves.

    The Macedo & Zalewski files carry the label in a column named 'label'
    (values: CONFIRMED / CANDIDATE / FALSE POSITIVE), with the flux in
    numeric columns '0'..'N'. Each row is one phase-folded light curve.

    Returns
    -------
    X_global : (n_samples, global_len, 1) float32
    X_local  : (n_samples, local_len, 1) float32 or None
    y        : (n_samples,) int64
    info     : dict from report_balance()
    """
    print(f"Loading global view from {global_path} ...")
    g = pd.read_csv(global_path)

    if label_col not in g.columns:
        raise ValueError(
            f"Expected a '{label_col}' column in {global_path}. "
            f"Found columns ending in: {list(g.columns)[-4:]}")

    y = map_disposition(g[label_col]).values
    # flux columns = everything except the label (they are named 0,1,2,...)
    flux_cols = [c for c in g.columns if c != label_col]
    X_global = g[flux_cols].values.astype(np.float32)
    X_global = _center_rows(X_global)              # baseline ~1.0 -> 0
    X_global = X_global[..., np.newaxis]

    X_local = None
    if local_path:
        print(f"Loading local view from {local_path} ...")
        l = pd.read_csv(local_path)
        lcols = [c for c in l.columns if c != label_col]
        X_local = _center_rows(l[lcols].values.astype(np.float32))[..., np.newaxis]

    print("\nClass balance:")
    info = report_balance(y)

    return X_global, X_local, y, info


def _center_rows(X: np.ndarray) -> np.ndarray:
    """Per-row baseline centring: subtract the median of each curve.

    Folded curves sit near 1.0; subtracting the per-row median puts the
    baseline at 0 so the transit dip becomes a clear negative deviation.
    """
    med = np.nanmedian(X, axis=1, keepdims=True)
    X = X - med
    X = np.nan_to_num(X, nan=0.0)
    return X


def load_tabular(koi_path: str,
                 label_col: str = "koi_disposition",
                 feature_cols: list[str] | None = None):
    """Load the KOI catalogue for the classical models.

    Default features are the standard KOI transit/stellar parameters.
    """
    df = pd.read_csv(koi_path)
    default_feats = [
        "koi_period", "koi_duration", "koi_depth", "koi_prad",
        "koi_impact", "koi_model_snr", "koi_steff", "koi_slogg",
        "koi_srad", "koi_kepmag",
    ]
    feature_cols = feature_cols or [c for c in default_feats if c in df.columns]
    X = df[feature_cols].copy()
    X = X.fillna(X.median(numeric_only=True))
    y = map_disposition(df[label_col]).values

    print(f"Tabular features used: {feature_cols}")
    print("\nClass balance:")
    info = report_balance(y)
    return X.values.astype(np.float32), y, feature_cols, info


if __name__ == "__main__":
    import sys
    # Usage: python src/data_loader.py all_global.csv [all_local.csv] [koi.csv]
    if len(sys.argv) < 2:
        print("Usage: python src/data_loader.py all_global.csv "
              "[all_local.csv] [lighkurve_KOI_dataset.csv]")
        sys.exit(0)

    g = sys.argv[1]
    l = sys.argv[2] if len(sys.argv) > 2 else None
    k = sys.argv[3] if len(sys.argv) > 3 else None
    Xg, Xl, y, info = load_light_curves(g, l, k)
    print(f"\nX_global shape: {Xg.shape}")
    if Xl is not None:
        print(f"X_local shape : {Xl.shape}")
