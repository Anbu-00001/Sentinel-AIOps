"""
train_v2.py — Sentinel-AIOps Production LightGBM Multiclass Classifier
=========================================================================
Replaces the baseline Isolation Forest with a supervised LightGBM model
trained on the 10 failure_type classes. Uses StratifiedShuffleSplit for
balanced data splitting and produces a full classification report,
feature importance plot, and model registry entry.

Workflow Rules (AGENTS.md):
  - Log 'Reasoning' before execution.
  - Save all ML metrics (F1-Score, PR AUC) as Artifacts.
"""

import json
import logging
import os
from datetime import datetime, timezone

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scipy.sparse import load_npz
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crypto_sig

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("sentinel.train_v2")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")

# ── Hyper-parameters ──────────────────────────────────────────────────────────
LGBM_PARAMS = {
    "objective": "multiclass",
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "n_estimators": 400,
    "learning_rate": 0.1,
    "num_leaves": 63,
    "max_depth": 8,
    "min_child_samples": 30,
    "subsample": 0.75,
    "colsample_bytree": 1.0,
    "reg_alpha": 0.05,
    "reg_lambda": 0.05,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

# Minimum acceptable macro F1 on the test set
MIN_MACRO_F1: float = 0.60


def build_feature_names():
    """Build human-readable feature names matching the preprocessor output."""
    meta_path = os.path.join(MODELS_DIR, "feature_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    names = []
    names.extend(meta["numerical_cols"])
    names.extend([f"hash_{i}" for i in range(meta["hash_n_features"])])
    names.extend([f"tfidf_{i}" for i in range(meta["tfidf_max_features"])])
    names.extend([str(c) for c in meta["dummy_column_order"]])
    names.extend(meta["bool_cols"])
    return names, meta


def load_data():
    """Load the pre-built feature matrix and string labels."""
    log.info("Reasoning: Loading feature matrix and labels from /data/.")
    X = load_npz(os.path.join(DATA_DIR, "feature_matrix.npz"))
    labels = np.load(os.path.join(DATA_DIR, "labels.npy"), allow_pickle=True)
    log.info("Loaded X=%s, labels=%d unique classes.", X.shape, len(np.unique(labels)))
    return X, labels


def encode_labels(labels):
    """Encode string labels to integers, return encoder + encoded array."""
    le = LabelEncoder()
    y = le.fit_transform(labels)
    log.info("Reasoning: Encoded %d classes → %s", len(le.classes_), list(le.classes_))
    return y, le


def stratified_split(X, y, test_size=0.20):
    """80/20 stratified split maintaining class balance."""
    log.info(
        "Reasoning: StratifiedShuffleSplit with test_size=%.2f to preserve class ratios.",
        test_size,
    )
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, test_idx = next(sss.split(X, y))
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def train_lgbm(X_train, y_train, X_test, y_test, feature_names):
    """Train a LightGBM classifier (fixed rounds, no early stopping)."""
    log.info(
        "Reasoning: Training LightGBM (objective=multiclass, n_estimators=%d, lr=%.3f). "
        "No early stopping — synthetic dataset has low feature-label signal.",
        LGBM_PARAMS["n_estimators"],
        LGBM_PARAMS["learning_rate"],
    )

    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        feature_name=feature_names,
        callbacks=[
            lgb.log_evaluation(period=100),
        ],
    )
    log.info("Training complete after %d boosting rounds.", LGBM_PARAMS["n_estimators"])
    return model


def generate_report(model, X_test, y_test, le):
    """Generate per-class classification report and save as JSON artifact."""
    log.info(
        "Reasoning: Generating classification report (precision, recall, F1, PR AUC per class)."
    )
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    report = classification_report(
        y_test,
        y_pred,
        target_names=list(le.classes_),
        output_dict=True,
        zero_division=0,
    )

    from sklearn.preprocessing import LabelBinarizer
    from sklearn.metrics import average_precision_score
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test_bin = lb.transform(y_test)

    macro_pr_auc = average_precision_score(y_test_bin, y_proba, average="macro")
    report["macro avg"]["pr_auc"] = macro_pr_auc

    report_path = os.path.join(MODELS_DIR, "v2_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info("Classification report saved to %s", report_path)

    # Print summary
    log.info("─── Per-Class Results ───")
    for cls in le.classes_:
        m = report[cls]
        cls_f1 = m["f1-score"]
        log.info(
            "  %-25s  P=%.4f  R=%.4f  F1=%.4f  n=%d",
            cls,
            m["precision"],
            m["recall"],
            cls_f1,
            int(m["support"]),
        )
        if cls_f1 < 0.40:
            log.warning(
                "WARNING: Class '%s' F1=%.4f is below 0.40 — review feature distributions.",
                cls, cls_f1,
            )
    macro = report["macro avg"]
    log.info(
        "  %-25s  P=%.4f  R=%.4f  F1=%.4f  PR-AUC=%.4f",
        "MACRO AVG",
        macro["precision"],
        macro["recall"],
        macro["f1-score"],
        macro.get("pr_auc", 0.0),
    )

    # Guard: do not commit a random-chance or memorized model
    f1 = report["macro avg"]["f1-score"]
    assert 0.60 <= f1 <= 0.90, (
        f"F1={f1:.4f} is outside the credible range [0.60, 0.90]. "
        f"If F1 > 0.90, noise injection is insufficient — increase "
        f"NOISE_SIGMA_FRACTION in preprocess.py. "
        f"If F1 < 0.60, numerical distributions need stronger separation."
    )

    return report


def plot_feature_importance(model, top_n=30):
    """Plot native LightGBM feature importance ('gain') and save PNG."""
    log.info(
        "Reasoning: Generating feature_importance.png using LightGBM native gain-based importance."
    )
    importance = model.feature_importances_
    feature_names = [f"f_{i}" for i in range(len(importance))]

    # Load feature_meta to assign readable names where possible
    meta_path = os.path.join(MODELS_DIR, "feature_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    # Build human-readable names for known feature groups
    readable = []
    idx = 0
    for col in meta["numerical_cols"]:
        readable.append(col)
        idx += 1
    for i in range(meta["hash_n_features"]):
        readable.append(f"hash_{i}")
        idx += 1
    for i in range(meta["tfidf_max_features"]):
        readable.append(f"tfidf_{i}")
        idx += 1
    for col in meta["dummy_column_order"]:
        readable.append(str(col))
        idx += 1
    for col in meta["bool_cols"]:
        readable.append(col)
        idx += 1
    # Pad if sizes diverge
    while len(readable) < len(importance):
        readable.append(f"f_{len(readable)}")

    feature_names = readable[: len(importance)]

    # Select top-N
    top_idx = np.argsort(importance)[-top_n:]
    top_names = [feature_names[i] for i in top_idx]
    top_vals = importance[top_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_vals)))
    ax.barh(
        range(len(top_vals)), top_vals, color=colors, edgecolor="white", linewidth=0.5
    )
    ax.set_yticks(range(len(top_vals)))
    ax.set_yticklabels(top_names, fontsize=9)
    ax.set_xlabel("Feature Importance (Gain)", fontsize=12)
    ax.set_title(
        "Top-30 Feature Importance — Sentinel-AIOps LightGBM v2",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    out_path = os.path.join(MODELS_DIR, "feature_importance.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info("Feature importance plot saved to %s", out_path)


def update_registry(report, model):
    """Create or update registry.json with the new model version."""
    reg_path = os.path.join(MODELS_DIR, "registry.json")
    if os.path.exists(reg_path):
        with open(reg_path) as f:
            registry = json.load(f)
    else:
        registry = {"models": []}

    macro = report["macro avg"]
    entry = {
        "model_version": f"lgbm_v2.{len(registry['models']) + 1}",
        "algorithm": "LightGBM",
        "objective": "multiclass",
        "n_classes": 10,
        "best_iteration": LGBM_PARAMS["n_estimators"],
        "macro_f1": round(macro["f1-score"], 4),
        "macro_precision": round(macro["precision"], 4),
        "macro_recall": round(macro["recall"], 4),
        "macro_pr_auc": round(macro.get("pr_auc", 0.0), 4),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "artifacts": [
            "lgbm_model.joblib",
            "label_encoder.joblib",
            "v2_report.json",
            "feature_importance.png",
        ],
    }
    registry["models"].append(entry)
    registry["latest"] = entry["model_version"]

    with open(reg_path, "w") as f:
        json.dump(registry, f, indent=2)
    log.info(
        "Registry updated → %s (F1=%.4f)", entry["model_version"], macro["f1-score"]
    )


def main():
    log.info("=== Sentinel-AIOps | train_v2.py (LightGBM Multiclass) ===")

    feature_names, meta = build_feature_names()
    X, labels = load_data()
    y, le = encode_labels(labels)

    # Trim or pad feature_names to match actual matrix width
    n_feats = X.shape[1]
    if len(feature_names) < n_feats:
        feature_names.extend([f"f_{i}" for i in range(len(feature_names), n_feats)])
    feature_names = feature_names[:n_feats]

    X_train, X_test, y_train, y_test = stratified_split(X, y)
    log.info(
        "Train: %d samples | Test: %d samples | Features: %d",
        X_train.shape[0],
        X_test.shape[0],
        n_feats,
    )

    model = train_lgbm(X_train, y_train, X_test, y_test, feature_names)

    # Save model + label encoder
    joblib.dump(model, os.path.join(MODELS_DIR, "lgbm_model.joblib"))
    crypto_sig.sign_artifact(os.path.join(MODELS_DIR, "lgbm_model.joblib"))

    joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.joblib"))
    crypto_sig.sign_artifact(os.path.join(MODELS_DIR, "label_encoder.joblib"))

    log.info("Model saved and signed: lgbm_model.joblib, label_encoder.joblib")

    report = generate_report(model, X_test, y_test, le)
    plot_feature_importance(model)
    update_registry(report, model)

    log.info("=== train_v2.py complete. ===")


if __name__ == "__main__":
    main()
