"""
dual_view.py
------------
Loading, alignment and modelling for the paired global/local light-curve views.

The Macedo & Zalewski corpus supplies two representations of each candidate: a
global view spanning the whole phase-folded curve (2,001 bins) and a local view
zoomed on the transit (201 bins). Fusing the two is the design introduced by
Shallue & Vanderburg (2018) and is the standard architecture in this field.

ALIGNMENT WARNING
-----------------
The two files are NOT row-aligned as distributed. `all_global.csv` has 5,302
rows and `all_local.csv` has 5,300. The discrepancy is not a simple truncation:
global contains one extra record at index 3088 and one trailing record. Removing
index 3088 and truncating to 5,300 rows produces 100% label agreement, against
roughly 52% under random pairing, which establishes the correspondence beyond
reasonable doubt.

Pairing the files naively by position would silently attach the wrong local view
to every candidate after index 3088 - about 42% of the dataset - and the failure
would not be obvious from the training curves. `load_aligned_views` performs the
correction and asserts the result.
"""

from typing import Tuple

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    HAVE_TF = True
except ImportError:
    HAVE_TF = False


# index of the extra record in all_global.csv, determined empirically
GLOBAL_EXTRA_ROW = 3088
LABEL_COLUMN = "label"
POSITIVE_CLASS = "CONFIRMED"


def _centre(A: np.ndarray) -> np.ndarray:
    """Baseline-centre each curve and add a channel axis."""
    A = A - np.nanmedian(A, axis=1, keepdims=True)
    return np.nan_to_num(A, nan=0.0)[..., np.newaxis]


def load_aligned_views(global_path: str,
                       local_path: str,
                       centre: bool = True,
                       verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load both views, correct the row offset, and return (X_global, X_local, y).

    Raises AssertionError if the labels do not agree after correction, which
    would indicate the alignment assumption no longer holds for the input files.
    """
    gl = pd.read_csv(global_path)
    lo = pd.read_csv(local_path)

    g_flux = gl[[c for c in gl.columns if c != LABEL_COLUMN]].values.astype("float32")
    g_lab = gl[LABEL_COLUMN].values
    l_flux = lo[[c for c in lo.columns if c != LABEL_COLUMN]].values.astype("float32")
    l_lab = lo[LABEL_COLUMN].values

    n = len(l_lab)

    # The correction below is specific to the distributed files. Apply it only
    # when the inputs actually exhibit the expected shape mismatch; otherwise
    # pair by position and let the assertion below catch any disagreement.
    if len(g_lab) > GLOBAL_EXTRA_ROW and len(g_lab) != n:
        g_flux = np.delete(g_flux, GLOBAL_EXTRA_ROW, axis=0)
        g_lab = np.delete(g_lab, GLOBAL_EXTRA_ROW)
    g_flux, g_lab = g_flux[:n], g_lab[:n]

    if len(g_lab) != n:
        raise ValueError(
            f"Cannot align: global has {len(g_lab)} usable rows, local has {n}.")

    agreement = float((g_lab == l_lab).mean())
    assert agreement == 1.0, (
        f"Global/local alignment failed: label agreement {agreement:.4f}. "
        "Do not train on these pairs.")

    Xg = _centre(g_flux) if centre else g_flux[..., np.newaxis]
    Xl = _centre(l_flux) if centre else l_flux[..., np.newaxis]
    y = (pd.Series(g_lab).astype(str).str.strip().str.upper()
         == POSITIVE_CLASS).astype(int).values

    if verbose:
        print(f"Aligned {n:,} paired views "
              f"(global {Xg.shape[1]} bins, local {Xl.shape[1]} bins); "
              f"label agreement 100%.")
        print(f"Positives: {int(y.sum()):,} ({100*y.mean():.1f}%)")
    return Xg, Xl, y


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
def _branch(inp, name: str):
    """Convolutional stack for one view.

    Global max pooling is used rather than average pooling because a transit is
    a localised extremum; averaging over the sequence dilutes it.
    """
    h = layers.Conv1D(32, 5, activation="relu", padding="same", name=f"{name}_c1")(inp)
    h = layers.MaxPooling1D(4)(h)
    h = layers.Conv1D(64, 5, activation="relu", padding="same", name=f"{name}_c2")(h)
    h = layers.MaxPooling1D(4)(h)
    h = layers.Conv1D(64, 3, activation="relu", padding="same", name=f"{name}_c3")(h)
    return layers.GlobalMaxPooling1D()(h)


def build_dual_view(global_len: int = 2001, local_len: int = 201):
    """Two-branch CNN over the global and local views, fused before the head.

    Each branch learns filters appropriate to its own scale; the concatenated
    representation lets the classifier weigh broad context against fine detail.
    """
    g_in = layers.Input((global_len, 1), name="global_view")
    l_in = layers.Input((local_len, 1), name="local_view")
    h = layers.Concatenate(name="fusion")([_branch(g_in, "global"),
                                           _branch(l_in, "local")])
    h = layers.Dropout(0.4)(h)
    h = layers.Dense(64, activation="relu")(h)
    h = layers.Dropout(0.3)(h)
    out = layers.Dense(1, activation="sigmoid")(h)
    return models.Model([g_in, l_in], out, name="DualView_CNN")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python src/dual_view.py all_global.csv all_local.csv")
        sys.exit(0)
    Xg, Xl, y = load_aligned_views(sys.argv[1], sys.argv[2])
    if HAVE_TF:
        m = build_dual_view(Xg.shape[1], Xl.shape[1])
        print(f"DualView_CNN: {m.count_params():,} parameters")
