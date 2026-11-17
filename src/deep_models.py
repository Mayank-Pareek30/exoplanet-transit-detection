"""
deep_models.py
--------------
Deep architectures for transit classification.

Four models are provided:
  build_cnn     - 1D CNN, local shape detection
  build_lstm    - stacked LSTM, long-range temporal structure
  build_hybrid  - CNN feature extractor feeding a bidirectional LSTM
  build_dual_view - two-branch network reading the global and local views
                    together, following the design of Shallue & Vanderburg (2018)

The dual-view model is the reason the dataset ships both `all_global.csv` and
`all_local.csv`: the global view supplies context (is there anything else
periodic in this curve?) while the local view supplies resolution at the transit
itself. Fusing them is the standard approach in the literature and is included
here so the comparison is not limited to single-view models.
"""

from typing import Callable, Dict

import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks
    HAVE_TF = True
except ImportError:
    HAVE_TF = False


# ---------------------------------------------------------------------------
# Single-view architectures
# ---------------------------------------------------------------------------
def build_cnn(seq_len: int = 2001):
    m = models.Sequential([
        layers.Input((seq_len, 1)),
        layers.Conv1D(32, 5, activation="relu", padding="same"),
        layers.MaxPooling1D(4),
        layers.Conv1D(64, 5, activation="relu", padding="same"),
        layers.MaxPooling1D(4),
        layers.Conv1D(128, 3, activation="relu", padding="same"),
        layers.MaxPooling1D(4),
        layers.Flatten(),
        layers.Dropout(0.4),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ], name="CNN")
    return m


def build_lstm(seq_len: int = 2001):
    """Average pooling first keeps the recurrence tractable on long sequences."""
    m = models.Sequential([
        layers.Input((seq_len, 1)),
        layers.AveragePooling1D(8),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(32),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ], name="LSTM")
    return m


def build_hybrid(seq_len: int = 2001):
    m = models.Sequential([
        layers.Input((seq_len, 1)),
        layers.Conv1D(32, 5, activation="relu", padding="same"),
        layers.MaxPooling1D(4),
        layers.Conv1D(64, 5, activation="relu", padding="same"),
        layers.MaxPooling1D(4),
        layers.Bidirectional(layers.LSTM(32)),
        layers.Dropout(0.4),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ], name="CNN_LSTM_Hybrid")
    return m


# ---------------------------------------------------------------------------
# Dual-view architecture
# ---------------------------------------------------------------------------
def _view_branch(inp, name: str):
    """Convolutional stack applied to one view."""
    h = layers.Conv1D(32, 5, activation="relu", padding="same",
                      name=f"{name}_conv1")(inp)
    h = layers.MaxPooling1D(4)(h)
    h = layers.Conv1D(64, 5, activation="relu", padding="same",
                      name=f"{name}_conv2")(h)
    h = layers.MaxPooling1D(4)(h)
    h = layers.Conv1D(64, 3, activation="relu", padding="same",
                      name=f"{name}_conv3")(h)
    h = layers.GlobalMaxPooling1D()(h)      # max, not average: a transit is a
    return h                                # localised extremum, not an average


def build_dual_view(global_len: int = 2001, local_len: int = 201):
    """Two-branch CNN over the global and local views, fused before the head.

    Each branch learns filters suited to its own scale, and the concatenated
    representation lets the classifier weigh broad context against fine detail.
    """
    g_in = layers.Input((global_len, 1), name="global_view")
    l_in = layers.Input((local_len, 1), name="local_view")

    g = _view_branch(g_in, "global")
    l = _view_branch(l_in, "local")

    h = layers.Concatenate(name="fusion")([g, l])
    h = layers.Dropout(0.4)(h)
    h = layers.Dense(64, activation="relu")(h)
    h = layers.Dropout(0.3)(h)
    out = layers.Dense(1, activation="sigmoid")(h)

    return models.Model([g_in, l_in], out, name="DualView_CNN")


# ---------------------------------------------------------------------------
# Registry and training
# ---------------------------------------------------------------------------
def get_builders() -> Dict[str, Callable]:
    return {"CNN": build_cnn, "LSTM": build_lstm, "CNN_LSTM_Hybrid": build_hybrid}


def compile_model(model, name: str, cfg=None):
    """Attach the optimiser, using the safer schedule for the LSTM.

    Without gradient clipping and a lower learning rate the LSTM suffers
    exploding gradients on these sequences and collapses to chance within a few
    epochs, which would make any comparison against it meaningless.
    """
    lr = 1e-3
    clipnorm = None
    if name == "LSTM":
        lr, clipnorm = 3e-4, 1.0
    if cfg is not None:
        lr = cfg.lstm_learning_rate if name == "LSTM" else cfg.learning_rate
        clipnorm = cfg.lstm_clipnorm if name == "LSTM" else None

    opt = (tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=clipnorm)
           if clipnorm else tf.keras.optimizers.Adam(learning_rate=lr))
    model.compile(optimizer=opt, loss="binary_crossentropy",
                  metrics=[tf.keras.metrics.AUC(name="auc")])
    return model


def train_model(model, name, X_tr, y_tr, X_te, y_te, cfg=None, verbose=2):
    """Train with early stopping and LR reduction on plateau."""
    epochs = cfg.epochs if cfg else 50
    batch = cfg.batch_size if cfg else 64
    es_pat = cfg.early_stopping_patience if cfg else 8
    rlr_pat = cfg.reduce_lr_patience if cfg else 4
    factor = cfg.reduce_lr_factor if cfg else 0.5
    min_lr = cfg.min_lr if cfg else 1e-5

    compile_model(model, name, cfg)
    cb = [
        callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                patience=es_pat, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max",
                                    factor=factor, patience=rlr_pat,
                                    min_lr=min_lr),
    ]
    hist = model.fit(X_tr, y_tr, validation_data=(X_te, y_te),
                     epochs=epochs, batch_size=batch,
                     callbacks=cb, verbose=verbose)
    return model, hist
