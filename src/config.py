"""
config.py
---------
Central configuration for the exoplanet detection study.

Keeping every tunable in one place makes runs reproducible and makes the
experimental setup auditable: a reader can see the exact settings that produced
the reported numbers without reading through the training code.
"""

from dataclasses import dataclass, field, asdict
from typing import List
import json
import os
import random

import numpy as np


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_global_seed(seed: int) -> None:
    """Seed every source of randomness we can control.

    Note: full bit-for-bit determinism on GPU is not guaranteed even with this,
    because some cuDNN kernels are non-deterministic. That residual variance is
    exactly why results are reported over multiple seeds.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


@dataclass
class DataConfig:
    global_path: str = "all_global.csv"
    local_path: str = "all_local.csv"
    label_column: str = "label"
    positive_class: str = "CONFIRMED"
    # centring turns the ~1.0 baseline into 0 so the transit becomes a clear
    # negative deviation; this materially affects deep model convergence
    centre_curves: bool = True
    test_size: float = 0.2


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3
    # the LSTM needs a gentler schedule and gradient clipping: at 1e-3 without
    # clipping it suffers exploding gradients and collapses to chance
    lstm_learning_rate: float = 3e-4
    lstm_clipnorm: float = 1.0
    early_stopping_patience: int = 8
    reduce_lr_patience: int = 4
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-5


@dataclass
class ExperimentConfig:
    name: str = "exoplanet_comparison"
    seeds: List[int] = field(default_factory=lambda: [42, 7, 123, 2024, 99])
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    output_dir: str = "results"
    figure_dir: str = "figures"

    def save(self, path: str = None) -> str:
        path = path or os.path.join(self.output_dir, f"{self.name}_config.json")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        return path

    def summary(self) -> str:
        return (f"Experiment '{self.name}': {len(self.seeds)} seeds, "
                f"{self.train.epochs} max epochs, batch {self.train.batch_size}")


# Feature names for the classical branch, kept here so figures and tables
# always agree with the feature extraction code.
CLASSICAL_FEATURE_NAMES = [
    "Transit depth (min)",
    "Variability (std)",
    "Mean flux",
    "Robust depth (5th pct)",
    "Phase of deepest point",
    "Dip width",
]

DEFAULT_CONFIG = ExperimentConfig()
