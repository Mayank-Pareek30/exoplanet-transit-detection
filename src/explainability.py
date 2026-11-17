"""
explainability.py
-----------------
Explainable AI for the deep models.

Plain saliency (raw input gradients) is noisy and can be misleading. This module
implements three complementary attribution methods so that a claim about "what
the model looks at" is supported by more than one technique:

  1. Integrated Gradients  - axiomatic attribution, satisfies completeness
  2. Grad-CAM (1D)         - class-discriminative, uses conv feature maps
  3. Occlusion sensitivity - model-agnostic, directly measures what matters

For a phase-folded light curve the transit sits at the centre of the sequence,
so a well-behaved model should concentrate attribution there. `transit_focus_score`
quantifies exactly that, turning a qualitative figure into a number that can be
reported and compared across models.
"""

from typing import Callable, Dict, Optional, Tuple

import numpy as np

try:
    import tensorflow as tf
    HAVE_TF = True
except ImportError:
    HAVE_TF = False


# ---------------------------------------------------------------------------
# 1. Integrated Gradients
# ---------------------------------------------------------------------------
def integrated_gradients(model,
                         x: np.ndarray,
                         baseline: Optional[np.ndarray] = None,
                         steps: int = 64) -> np.ndarray:
    """Integrated Gradients (Sundararajan et al., 2017).

    Attribution is the path integral of gradients from a baseline to the input.
    Unlike raw saliency this satisfies *completeness*: attributions sum to the
    difference in model output between input and baseline, so the numbers mean
    something rather than just indicating local sensitivity.

    Baseline here is a flat (all-zero) curve, i.e. "no transit signal", which is
    the natural null case for centred flux.
    """
    x = np.asarray(x, dtype=np.float32)
    single = (x.ndim == 2)
    if single:
        x = x[np.newaxis, ...]
    if baseline is None:
        baseline = np.zeros_like(x)

    alphas = np.linspace(0.0, 1.0, steps, dtype=np.float32)
    total = np.zeros_like(x, dtype=np.float32)

    for a in alphas:
        point = baseline + a * (x - baseline)
        t = tf.convert_to_tensor(point)
        with tf.GradientTape() as tape:
            tape.watch(t)
            out = model(t, training=False)
        g = tape.gradient(out, t).numpy()
        total += g

    avg_grad = total / steps
    attribution = (x - baseline) * avg_grad
    return attribution[0] if single else attribution


# ---------------------------------------------------------------------------
# 2. Grad-CAM for 1D convolutions
# ---------------------------------------------------------------------------
def _last_conv_layer(model):
    for layer in reversed(model.layers):
        if "conv" in layer.__class__.__name__.lower():
            return layer
    return None


def grad_cam_1d(model,
                x: np.ndarray,
                layer_name: Optional[str] = None) -> np.ndarray:
    """Grad-CAM adapted to 1D sequences.

    Weights each convolutional feature map by the gradient of the output with
    respect to that map, then combines them. Because it works on feature maps
    rather than raw inputs it is far less noisy than pixel-level saliency, and
    it is class-discriminative.

    Returns a attribution vector upsampled to the input length, normalised to
    [0, 1].
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x[np.newaxis, ...]

    conv_layer = (model.get_layer(layer_name) if layer_name
                  else _last_conv_layer(model))
    if conv_layer is None:
        raise ValueError("No convolutional layer found for Grad-CAM.")

    # Keras 3 requires the model to have been called before layer outputs are
    # available, and Sequential models need an explicit functional rebuild.
    _ = model(tf.zeros((1,) + tuple(x.shape[1:]), dtype=tf.float32),
              training=False)

    # build a sub-model that returns the conv activations, then apply the
    # remaining layers manually so this works for Sequential and functional models
    layer_list = list(model.layers)
    conv_idx = layer_list.index(conv_layer)

    inp = tf.keras.Input(shape=tuple(x.shape[1:]))
    h = inp
    for layer in layer_list[:conv_idx + 1]:
        h = layer(h)
    conv_activations = h
    for layer in layer_list[conv_idx + 1:]:
        h = layer(h)
    grad_model = tf.keras.Model(inp, [conv_activations, h])

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(tf.convert_to_tensor(x), training=False)
        tape.watch(conv_out)
        target = preds[:, 0]
    grads = tape.gradient(target, conv_out)

    # channel weights = global average of the gradients
    weights = tf.reduce_mean(grads, axis=1)                 # (batch, channels)
    cam = tf.reduce_sum(conv_out * weights[:, None, :], axis=-1).numpy()[0]

    # Standard Grad-CAM applies ReLU to keep only positive evidence. That
    # assumption comes from image classification, where the evidence is an
    # additive feature. A transit is a *downward* dip, so the class evidence
    # here is often negative-going and a plain ReLU would zero the whole map.
    # We therefore take the magnitude of the deviation from the map's own mean,
    # which captures evidence of either polarity.
    if np.all(cam <= 0) or np.all(cam >= 0):
        cam = np.abs(cam - cam.mean())
    else:
        cam = np.maximum(cam, 0.0)

    # upsample back to input length
    cam = np.interp(np.linspace(0, 1, x.shape[1]),
                    np.linspace(0, 1, len(cam)), cam)
    rng = cam.max() - cam.min()
    return (cam - cam.min()) / rng if rng > 0 else cam


# ---------------------------------------------------------------------------
# 3. Occlusion sensitivity
# ---------------------------------------------------------------------------
def occlusion_sensitivity(model,
                          x: np.ndarray,
                          window: int = 50,
                          stride: int = 25) -> np.ndarray:
    """Slide a mask across the curve and measure the drop in predicted score.

    This is model-agnostic and needs no gradients: it directly answers "if this
    region were absent, how much would the prediction change?". Slower, but it
    is the most literal notion of importance and a useful cross-check on the
    gradient-based methods.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x[np.newaxis, ...]
    seq_len = x.shape[1]

    base_score = float(model(tf.convert_to_tensor(x), training=False).numpy().ravel()[0])

    positions, variants = [], []
    for start in range(0, seq_len - window + 1, stride):
        occluded = x.copy()
        occluded[0, start:start + window, 0] = 0.0    # 0 == baseline flux
        variants.append(occluded[0])
        positions.append(start)

    if not variants:
        return np.zeros(seq_len, dtype=np.float32)

    scores = model(tf.convert_to_tensor(np.stack(variants)),
                   training=False).numpy().ravel()
    drops = base_score - scores

    # spread each window's drop back over the sequence
    importance = np.zeros(seq_len, dtype=np.float32)
    counts = np.zeros(seq_len, dtype=np.float32)
    for start, d in zip(positions, drops):
        importance[start:start + window] += d
        counts[start:start + window] += 1
    counts[counts == 0] = 1
    return importance / counts


# ---------------------------------------------------------------------------
# Quantifying whether attribution lands on the transit
# ---------------------------------------------------------------------------
def transit_focus_score(attribution: np.ndarray,
                        centre_fraction: float = 0.05) -> float:
    """Fraction of total attribution mass falling in the central transit window.

    Phase-folded curves place the transit at the centre. A model that has
    learned the physics should put most of its attribution there. If the curve
    length is L and centre_fraction is 0.05, the window is the middle 5% of L.

    A value near `centre_fraction` means attribution is essentially uniform
    (no focus); values well above it indicate genuine concentration on the
    transit. The ratio of the two is reported by `focus_ratio`.
    """
    a = np.abs(np.asarray(attribution).ravel())
    total = a.sum()
    if total == 0:
        return 0.0
    L = len(a)
    half = int(L * centre_fraction / 2)
    centre = L // 2
    window = a[max(0, centre - half): min(L, centre + half)]
    return float(window.sum() / total)


def focus_ratio(attribution: np.ndarray, centre_fraction: float = 0.05) -> float:
    """How many times more attribution the transit window gets than chance.

    1.0 means no better than uniform; 5.0 means the transit window receives
    five times the attribution mass a uniform distribution would give it.
    """
    return transit_focus_score(attribution, centre_fraction) / centre_fraction


# ---------------------------------------------------------------------------
# Convenience: run all methods and score them
# ---------------------------------------------------------------------------
def explain_sample(model, x: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute all three attributions for one curve."""
    out = {}
    try:
        out["integrated_gradients"] = integrated_gradients(model, x, steps=32)[..., 0]
    except Exception as e:                                   # pragma: no cover
        out["integrated_gradients"] = None
    try:
        out["grad_cam"] = grad_cam_1d(model, x)
    except Exception:
        out["grad_cam"] = None
    try:
        out["occlusion"] = occlusion_sensitivity(model, x)
    except Exception:
        out["occlusion"] = None
    return {k: v for k, v in out.items() if v is not None}


def attribution_report(model, X: np.ndarray, y: np.ndarray,
                       n_samples: int = 20) -> "pd.DataFrame":
    """Average focus ratios across several confirmed planets.

    Reporting a mean focus ratio over many samples is far stronger evidence than
    showing three hand-picked curves: it demonstrates the behaviour is systematic.
    """
    import pandas as pd
    idx = np.where(y == 1)[0][:n_samples]
    rows = []
    for method in ("integrated_gradients", "grad_cam", "occlusion"):
        ratios = []
        for i in idx:
            expl = explain_sample(model, X[i])
            if method in expl:
                ratios.append(focus_ratio(expl[method]))
        if ratios:
            rows.append({
                "Method": method.replace("_", " ").title(),
                "Mean focus ratio": round(float(np.mean(ratios)), 2),
                "Std": round(float(np.std(ratios)), 2),
                "N samples": len(ratios),
            })
    return pd.DataFrame(rows)
