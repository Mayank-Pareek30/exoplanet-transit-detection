"""
preprocessing.py
----------------
Loads and preprocesses Kepler light curves and the stellar/KOI catalogue.

Two data branches, matching the project methodology:
  1. Light curves (Mendeley 2024 set)  -> deep learning models (CNN, LSTM, hybrid)
  2. Tabular catalogue (NASA stellar/KOI) -> classical models (RF, SVM, XGBoost)

The Mendeley dataset columns: kepid, koi_disposition, koi_period,
koi_time0bk, koi_duration, koi_quarters, plus the PDCSAP flux series.
Label = koi_disposition mapped to binary (CONFIRMED -> 1, else 0).
"""

import numpy as np
import pandas as pd
from scipy import interpolate


# ---------------------------------------------------------------------------
# Label handling
# ---------------------------------------------------------------------------
def disposition_to_binary(disposition: str) -> int:
    """Map a KOI disposition string to a binary planet label.

    CONFIRMED -> 1 (planet).  CANDIDATE / FALSE POSITIVE -> 0 (no confirmed planet).
    Adjust here if you decide to treat CANDIDATE differently.
    """
    if isinstance(disposition, str) and disposition.strip().upper() == "CONFIRMED":
        return 1
    return 0


# ---------------------------------------------------------------------------
# Light-curve preprocessing (for CNN / LSTM)
# ---------------------------------------------------------------------------
def normalise_flux(flux: np.ndarray) -> np.ndarray:
    """Divide by the median so the out-of-transit level sits near 1.0."""
    med = np.nanmedian(flux)
    if med == 0 or np.isnan(med):
        return flux
    return flux / med


def interpolate_gaps(flux: np.ndarray) -> np.ndarray:
    """Linear interpolation across NaN gaps (matches the Mendeley pipeline)."""
    flux = np.asarray(flux, dtype=float)
    n = len(flux)
    idx = np.arange(n)
    good = ~np.isnan(flux)
    if good.sum() < 2:
        return np.nan_to_num(flux, nan=1.0)
    f = interpolate.interp1d(
        idx[good], flux[good], kind="linear",
        bounds_error=False, fill_value=(flux[good][0], flux[good][-1])
    )
    return f(idx)


def remove_outliers(flux: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Clip points beyond `sigma` standard deviations (Mendeley used 2-sigma).

    Note: a transit is a real downward dip, so clipping symmetrically can erase
    shallow transits. We clip only positive spikes (cosmic rays) by default and
    leave downward excursions intact.
    """
    mean, std = np.nanmean(flux), np.nanstd(flux)
    upper = mean + sigma * std
    clipped = np.where(flux > upper, upper, flux)
    return clipped


def resample_to_length(flux: np.ndarray, target_len: int = 2000) -> np.ndarray:
    """Downsample / interpolate each curve to a fixed length for the network.

    ~60,000 raw points per curve is heavy; a fixed shorter length keeps the
    transit shape while making training tractable. Tune target_len as needed.
    """
    flux = np.asarray(flux, dtype=float)
    if len(flux) == target_len:
        return flux
    old_idx = np.linspace(0, 1, len(flux))
    new_idx = np.linspace(0, 1, target_len)
    return np.interp(new_idx, old_idx, flux)


def preprocess_light_curve(flux: np.ndarray,
                           sigma: float = 2.0,
                           target_len: int = 2000) -> np.ndarray:
    """Full single-curve pipeline: interpolate -> normalise -> declip -> resample."""
    flux = interpolate_gaps(flux)
    flux = normalise_flux(flux)
    flux = remove_outliers(flux, sigma=sigma)
    flux = resample_to_length(flux, target_len=target_len)
    return flux.astype(np.float32)


def phase_fold(time: np.ndarray,
               flux: np.ndarray,
               period: float,
               epoch: float,
               duration_hours: float | None = None,
               bins: int = 2000,
               duration_windows: float = 3.0):
    """Phase-fold a light curve on a candidate's period and transit epoch.

    This is the single most important step for the deep-learning branch. Raw
    Kepler light curves span ~4 years and contain transits at scattered times;
    a network sees the transit at a different position in every curve, which is
    very hard to learn. Folding stacks every orbit on top of the others so the
    transit always sits at the same phase (phase 0), turning a sparse, shifting
    signal into a single aligned dip the model can latch onto.

    The fold uses the linear ephemeris  t_center(i) = epoch + i * period,
    which is the standard way the Kepler catalogue defines transit times.

    Parameters
    ----------
    time : array
        Timestamps of the light curve, in BKJD (same units as `epoch`).
    flux : array
        Flux values (any normalisation; normalise separately if needed).
    period : float
        Orbital period in days  (koi_period).
    epoch : float
        Transit epoch in BKJD    (koi_time0bk) — the centre of a transit.
    duration_hours : float, optional
        Transit duration in hours (koi_duration). If given, the output is
        restricted to a window of `duration_windows` transit-durations around
        phase 0, so the network sees a zoomed-in view of the transit rather
        than mostly flat baseline.
    bins : int
        Number of points in the binned, folded output sequence.
    duration_windows : float
        Half-width of the kept window, in units of transit duration, when
        `duration_hours` is provided.

    Returns
    -------
    folded_flux : np.ndarray, shape (bins,)
        The binned, phase-folded flux, centred so the transit is at the middle.
    """
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)

    # phase in days since the nearest transit centre, in range [-period/2, period/2)
    phase = np.mod(time - epoch + 0.5 * period, period) - 0.5 * period

    # optionally zoom to a window around the transit
    if duration_hours is not None and duration_hours > 0:
        half_window = duration_windows * (duration_hours / 24.0)  # hours -> days
        keep = np.abs(phase) <= half_window
        if keep.sum() > bins // 4:        # only zoom if enough points remain
            phase, flux = phase[keep], flux[keep]
            lo, hi = -half_window, half_window
        else:
            lo, hi = -0.5 * period, 0.5 * period
    else:
        lo, hi = -0.5 * period, 0.5 * period

    # sort by phase so the dip is centred and contiguous
    order = np.argsort(phase)
    phase, flux = phase[order], flux[order]

    # bin into `bins` equal-width phase bins (median is robust to residual noise)
    edges = np.linspace(lo, hi, bins + 1)
    idx = np.digitize(phase, edges) - 1
    idx = np.clip(idx, 0, bins - 1)

    folded = np.full(bins, np.nan, dtype=float)
    for b in range(bins):
        vals = flux[idx == b]
        if len(vals):
            folded[b] = np.median(vals)

    # fill any empty bins by interpolation, then normalise to the out-of-transit level
    folded = interpolate_gaps(folded)
    folded = normalise_flux(folded)
    return folded.astype(np.float32)


def center_for_model(folded_flux: np.ndarray, scale: float = 100.0) -> np.ndarray:
    """Shift a folded curve so the baseline sits at 0, then scale.

    Networks train far better on inputs centred near zero. A folded curve sits
    around 1.0 with a shallow dip; subtracting 1.0 turns it into deviations
    from baseline, and scaling makes the ~1% dip a numerically meaningful
    signal. This step is what lifts deep-model performance from chance to high
    accuracy on folded data.
    """
    return ((np.asarray(folded_flux, dtype=np.float32) - 1.0) * scale)


def preprocess_folded(time, flux, period, epoch, duration_hours=None,
                      bins: int = 2000) -> np.ndarray:
    """Convenience wrapper: clean the raw flux, then phase-fold it.

    Use this for the deep-learning branch when you have per-cadence (time, flux)
    plus the candidate ephemeris. Returns a fixed-length folded sequence.
    """
    flux = interpolate_gaps(flux)
    flux = remove_outliers(flux, sigma=2.0)
    return phase_fold(time, flux, period, epoch,
                      duration_hours=duration_hours, bins=bins)


def build_folded_dataset(records: list[dict], bins: int = 2000,
                         center: bool = True):
    """Build (X, y) of phase-folded light curves for the deep models.

    Each record is a dict with keys:
        'time', 'flux'           -> arrays of the raw light curve
        'period', 'epoch'        -> koi_period, koi_time0bk
        'duration' (optional)    -> koi_duration in hours
        'label'                  -> 0/1

    If `center` is True (recommended), each folded curve is baseline-centred
    and scaled with center_for_model() so it is ready for a neural network.
    """
    folded = [
        preprocess_folded(r["time"], r["flux"], r["period"], r["epoch"],
                          r.get("duration"), bins=bins)
        for r in records
    ]
    if center:
        folded = [center_for_model(f) for f in folded]
    X = np.stack(folded)
    X = X[..., np.newaxis]
    y = np.asarray([r["label"] for r in records], dtype=np.int64)
    return X, y


def build_lightcurve_dataset(curves: list[np.ndarray],
                             labels: list[int],
                             target_len: int = 2000):
    """Apply preprocessing to a list of raw flux arrays.

    Returns X of shape (n_samples, target_len, 1) ready for a 1D CNN / LSTM,
    and y of shape (n_samples,).
    """
    X = np.stack([preprocess_light_curve(c, target_len=target_len) for c in curves])
    X = X[..., np.newaxis]          # add channel dim for Conv1D / LSTM
    y = np.asarray(labels, dtype=np.int64)
    return X, y


# ---------------------------------------------------------------------------
# Tabular preprocessing (for classical models)
# ---------------------------------------------------------------------------
TABULAR_FEATURES = [
    "kepmag", "teff", "logg", "feh", "radius", "mass", "dens",
    "koi_period", "koi_duration",       # present in the KOI/Mendeley catalogue
]


def load_stellar_catalogue(path: str) -> pd.DataFrame:
    """Load the NASA stellar CSV, skipping the '#' comment header."""
    df = pd.read_csv(path, comment="#")
    return df


def build_tabular_dataset(df: pd.DataFrame,
                          label_col: str = "nconfp",
                          feature_cols: list[str] | None = None):
    """Build (X, y) for the classical models from the catalogue table.

    label: nconfp > 0  ->  1 (has a confirmed planet), else 0.
    Missing feature values are median-imputed.
    """
    feature_cols = feature_cols or [c for c in TABULAR_FEATURES if c in df.columns]
    X = df[feature_cols].copy()
    # median imputation
    X = X.fillna(X.median(numeric_only=True))
    y = (df[label_col] > 0).astype(int).values
    return X.values.astype(np.float32), y, feature_cols


def check_class_balance(y, imbalance_threshold: float = 0.30) -> dict:
    """Report the class balance and recommend whether imbalance handling is needed.

    This dataset is built from KOIs (already-flagged candidates), so the
    CONFIRMED vs FALSE-POSITIVE split is often far more balanced than the
    1:99 ratio seen in the full stellar catalogue. Rather than assume, we
    measure and decide.

    Returns a dict with the counts, the minority fraction, and a boolean
    `needs_handling`: True when the minority class is below `imbalance_threshold`
    (default 30%), in which case SMOTE / class weighting is worth applying.
    """
    import numpy as np
    y = np.asarray(y)
    n = len(y)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    minority_frac = min(n_pos, n_neg) / n if n else 0.0
    needs = minority_frac < imbalance_threshold

    print(f"Class balance: {n_pos} positive ({100*n_pos/n:.1f}%), "
          f"{n_neg} negative ({100*n_neg/n:.1f}%)")
    if needs:
        print(f"-> Minority class is {100*minority_frac:.1f}% (< "
              f"{100*imbalance_threshold:.0f}%): applying SMOTE + class weighting.")
    else:
        print(f"-> Minority class is {100*minority_frac:.1f}% (>= "
              f"{100*imbalance_threshold:.0f}%): data is reasonably balanced, "
              f"imbalance handling not required.")
    return {
        "n_positive": n_pos, "n_negative": n_neg,
        "minority_fraction": minority_frac, "needs_handling": needs,
    }


def load_koi_lightcurve_csv(global_path: str,
                            label_col: str = "koi_disposition"):
    """Load the Mendeley global/local light-curve CSV for the deep models.

    These files are already phase-folded and binned (the global view is the
    full folded curve, the local view zooms on the transit), so no further
    folding is needed. Each row is one light curve; flux columns are every
    numeric column except the identifier/label columns.

    Returns (X, y): X shape (n_samples, n_bins, 1), y binary from disposition.
    """
    import numpy as np
    df = pd.read_csv(global_path)

    id_cols = {"kepid", "kepoi_name", "kepler_name", label_col,
               "koi_period", "koi_time0bk", "koi_duration", "koi_quarters"}
    flux_cols = [c for c in df.columns if c not in id_cols]

    X = df[flux_cols].to_numpy(dtype=np.float32)
    # baseline-centre each row (folded curves sit near 1.0); helps the network
    X = X - np.nanmedian(X, axis=1, keepdims=True)
    X = np.nan_to_num(X)[..., np.newaxis]

    if label_col in df.columns:
        y = (df[label_col].astype(str).str.upper() == "CONFIRMED").astype(int).to_numpy()
    else:
        y = None
    return X, y


if __name__ == "__main__":
    # quick smoke test on the stellar CSV
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/keplerstellar.csv"
    df = load_stellar_catalogue(path)
    X, y, cols = build_tabular_dataset(df)
    print(f"Tabular: X={X.shape}, y={y.shape}, positives={y.sum()} "
          f"({100*y.mean():.2f}%)")
    print("Features:", cols)
