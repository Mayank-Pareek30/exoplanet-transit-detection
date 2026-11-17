"""
classical_models.py
--------------------
Baseline classical models for the tabular branch:
Random Forest, SVM (RBF), and XGBoost.

Handles the 1:99 class imbalance with SMOTE + class weighting, and reports
precision / recall / F1 / ROC-AUC (not accuracy, which is misleading here).
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False

try:
    from imblearn.over_sampling import SMOTE
    HAVE_SMOTE = True
except ImportError:
    HAVE_SMOTE = False


def get_models():
    """Return the dict of classical models to compare.

    class_weight='balanced' makes RF and SVM penalise missing the rare
    planet class more heavily. XGBoost uses scale_pos_weight for the same.
    """
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=None, class_weight="balanced",
            n_jobs=-1, random_state=42,
        ),
        "SVM_RBF": SVC(
            kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
            probability=True, random_state=42,
        ),
    }
    if HAVE_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            # Ratio of negatives to positives in the KOI light-curve dataset
            # (3,107 : 2,195). An earlier value of 99 was carried over from the
            # full stellar catalogue, where the imbalance is far more severe;
            # applied here it over-weighted the positive class by roughly 70x
            # and pushed the operating point badly off, inflating recall at the
            # cost of precision and calibration.
            scale_pos_weight=1.4,
            eval_metric="logloss", random_state=42, n_jobs=-1,
        )
    return models


def evaluate_model(model, X, y, n_splits: int = 5, use_smote: bool = True):
    """Stratified k-fold evaluation. Returns mean metrics across folds."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = {"precision": [], "recall": [], "f1": [], "roc_auc": []}

    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        scaler = StandardScaler().fit(X_tr)
        X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)

        if use_smote and HAVE_SMOTE:
            X_tr, y_tr = SMOTE(random_state=42).fit_resample(X_tr, y_tr)

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        y_prob = (model.predict_proba(X_te)[:, 1]
                  if hasattr(model, "predict_proba") else y_pred)

        scores["precision"].append(precision_score(y_te, y_pred, zero_division=0))
        scores["recall"].append(recall_score(y_te, y_pred, zero_division=0))
        scores["f1"].append(f1_score(y_te, y_pred, zero_division=0))
        scores["roc_auc"].append(roc_auc_score(y_te, y_prob))

    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in scores.items()}


def run_all(X, y, n_splits: int = 5, use_smote: bool | None = None):
    """Evaluate every classical model and print a comparison table.

    If `use_smote` is None, it is decided automatically from the class balance:
    SMOTE is applied only when the minority class is under 30%. Pass True/False
    to force it on or off.
    """
    if use_smote is None:
        try:
            import sys
            sys.path.append(".")
            from src.preprocessing import check_class_balance
            use_smote = check_class_balance(y)["needs_handling"]
        except Exception:
            use_smote = True       # safe default

    results = {}
    print(f"\n{'Model':<14}{'Precision':>12}{'Recall':>10}{'F1':>10}{'ROC-AUC':>10}")
    print("-" * 56)
    for name, model in get_models().items():
        m = evaluate_model(model, X, y, n_splits=n_splits, use_smote=use_smote)
        results[name] = m
        print(f"{name:<14}"
              f"{m['precision'][0]:>12.3f}"
              f"{m['recall'][0]:>10.3f}"
              f"{m['f1'][0]:>10.3f}"
              f"{m['roc_auc'][0]:>10.3f}")
    return results


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.preprocessing import load_stellar_catalogue, build_tabular_dataset

    path = sys.argv[1] if len(sys.argv) > 1 else "data/keplerstellar.csv"
    df = load_stellar_catalogue(path)

    # For a quick run, subsample to keep SVM tractable (it scales poorly).
    if len(df) > 20000:
        import pandas as pd
        pos = df[df["nconfp"] > 0]
        neg = df[df["nconfp"] == 0].sample(n=min(len(pos) * 20, len(df)),
                                           random_state=42)
        df = pd.concat([pos, neg]).sample(frac=1, random_state=42)
        print(f"Subsampled to {len(df)} rows for a quick demo run.")

    X, y, cols = build_tabular_dataset(df)
    print(f"Features: {cols}")
    run_all(X, y)
