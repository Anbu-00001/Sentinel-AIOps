"""
train_detector.py — Sentinel-AIOps Isolation Forest Anomaly Detector
=====================================================================
Trains an unsupervised Isolation Forest on the feature matrix produced by
preprocess.py. Labels (failure_type) are used ONLY for validation metrics,
never seen during training.

Anomaly Score Formula (consistent with original IF paper):
    s(x, n) = 2 ^ ( -E[h(x)] / c(n) )
where c(n) = 2*H(n-1) - 2*(n-1)/n  (expected path length for n samples)

Artifacts saved to /models/:
  - isolation_forest.joblib
  - anomaly_scores.npy
  - pr_curve.png
  - confusion_matrix.png
  - metrics.json

Workflow Rules (AGENTS.md):
  - Log 'Reasoning' before execution.
  - Save all ML metrics (F1-Score, PR AUC) as Artifacts.
"""

import json
import logging
import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import load_npz
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    ConfusionMatrixDisplay,
)

# ── Logging ──────────────────────────────────────────────────────────────────
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import sentinel_logging
sentinel_logging.configure_logging()
log = logging.getLogger("sentinel.train_detector")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")

N_ESTIMATORS = 200
CONTAMINATION = (
    0.1  # Conservative prior: ~10% of CI/CD runs exhibit anomalous behaviour
)


def c_factor(n: int) -> float:
    """
    Expected path length c(n) for an Isolation Forest of n samples.
    c(n) = 2 * H(n-1) - 2*(n-1)/n
    H(i) ≈ ln(i) + 0.5772156649 (Euler-Mascheroni constant)
    """
    if n <= 1:
        return 1.0
    harmonic = np.log(n - 1) + 0.5772156649
    return 2.0 * harmonic - (2.0 * (n - 1) / n)


def compute_anomaly_scores(model: IsolationForest, X) -> np.ndarray:
    """
    Compute anomaly scores using the original IF formula:
        s(x, n) = 2 ^ ( -E[h(x)] / c(n) )

    sklearn's score_samples(X) returns:
        -E[h(x)] / c(n_max_samples)  (shifted so that 0.5 = boundary)
    We recover E[h(x)] and apply the raw formula for full interpretability.
    """
    log.info(
        "Reasoning: Computing anomaly scores via s(x,n) = 2^(-E[h(x)]/c(n)) "
        "using sklearn score_samples (vectorized, sparse-safe)."
    )
    n_train = model.max_samples_
    c_n = c_factor(n_train)

    # sklearn score_samples = -mean_path_length / c(n) + offset(0.5)
    # raw_score = score_samples + 0.5  →  raw_score = -E[h(x)] / c(n)
    # E[h(x)] = -raw_score * c(n)
    raw = model.score_samples(X)  # shape (n_samples,)
    e_h_x = -(raw) * c_n  # recover mean path length
    scores = np.power(2.0, -e_h_x / c_n)  # s(x, n) ∈ (0, 1]
    return scores


def plot_pr_curve(y_true_binary: np.ndarray, scores: np.ndarray, out_path: str):
    """Generate and save a Precision-Recall curve."""
    log.info("Reasoning: Generating Precision-Recall curve artifact.")
    precision, recall, thresholds = precision_recall_curve(y_true_binary, scores)
    pr_auc = average_precision_score(y_true_binary, scores)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        recall,
        precision,
        lw=2,
        color="#4C72B0",
        label=f"Isolation Forest (PR AUC = {pr_auc:.4f})",
    )
    ax.fill_between(recall, precision, alpha=0.15, color="#4C72B0")
    ax.axhline(
        y=CONTAMINATION,
        color="red",
        linestyle="--",
        lw=1.2,
        label=f"Baseline (contamination = {CONTAMINATION})",
    )
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(
        "Precision-Recall Curve — Sentinel-AIOps Anomaly Detector",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.grid(alpha=0.35)
    fig.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info("PR curve saved to %s  |  PR AUC = %.4f", out_path, pr_auc)
    return pr_auc


def plot_confusion_matrix(
    y_true_binary: np.ndarray, y_pred_binary: np.ndarray, out_path: str
):
    """Generate and save a Confusion Matrix."""
    log.info("Reasoning: Generating Confusion Matrix artifact.")
    cm = confusion_matrix(y_true_binary, y_pred_binary)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Normal", "Anomaly"]
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title("Confusion Matrix — Sentinel-AIOps", fontsize=13, fontweight="bold")
    fig.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info("Confusion matrix saved to %s", out_path)


def main():
    log.info("=== Sentinel-AIOps | train_detector.py ===")

    # Load feature matrix and labels
    log.info("Reasoning: Loading feature matrix produced by preprocess.py.")
    X = load_npz(os.path.join(DATA_DIR, "feature_matrix.npz"))
    labels = np.load(os.path.join(DATA_DIR, "labels.npy"), allow_pickle=True)
    log.info("Feature matrix: %s | Labels: %d classes", X.shape, len(np.unique(labels)))

    # ── Model Training (Unsupervised — labels NOT used) ──────────────────────
    log.info(
        "Reasoning: Training Isolation Forest (%d estimators, contamination=%.2f). Labels are EXCLUDED from training.",
        N_ESTIMATORS,
        CONTAMINATION,
    )
    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X)
    log.info("Isolation Forest training complete.")

    # ── Anomaly Scoring ──────────────────────────────────────────────────────
    scores = compute_anomaly_scores(model, X)
    np.save(os.path.join(MODELS_DIR, "anomaly_scores.npy"), scores)
    log.info(
        "Score stats — min: %.4f | max: %.4f | mean: %.4f | std: %.4f",
        scores.min(),
        scores.max(),
        scores.mean(),
        scores.std(),
    )

    # IF predicts: -1 = anomaly, 1 = normal
    raw_preds = model.predict(X)
    y_pred_binary = (raw_preds == -1).astype(int)  # 1 = anomaly

    # ── Validation: use labels as proxy for "what should be anomalies" ───────
    # Strategy: treat 'Timeout' and 'Resource Exhaustion' as the most
    # anomalous classes (operational definition); rest as normal.
    ANOMALY_CLASSES = {"Timeout", "Resource Exhaustion", "Security Scan Failure"}
    log.info(
        "Reasoning: Using %s as 'anomaly' proxy classes for validation only.",
        ANOMALY_CLASSES,
    )
    y_true_binary = np.isin(labels, list(ANOMALY_CLASSES)).astype(int)

    pr_auc = plot_pr_curve(
        y_true_binary, scores, os.path.join(MODELS_DIR, "pr_curve.png")
    )
    plot_confusion_matrix(
        y_true_binary, y_pred_binary, os.path.join(MODELS_DIR, "confusion_matrix.png")
    )

    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    log.info("Validation Metrics — F1-Score: %.4f | PR AUC: %.4f", f1, pr_auc)

    # ── Save model artifact ──────────────────────────────────────────────────
    model_path = os.path.join(MODELS_DIR, "isolation_forest.joblib")
    joblib.dump(model, model_path)
    log.info("Model artifact saved to %s", model_path)

    # ── Save metrics as Artifact (AGENTS.md protocol) ────────────────────────
    metrics = {
        "model": "IsolationForest",
        "n_estimators": N_ESTIMATORS,
        "contamination": CONTAMINATION,
        "training_samples": int(X.shape[0]),
        "feature_dim": int(X.shape[1]),
        "validation_proxy_classes": sorted(ANOMALY_CLASSES),
        "f1_score": round(float(f1), 4),
        "pr_auc": round(float(pr_auc), 4),
        "anomaly_score_stats": {
            "min": round(float(scores.min()), 4),
            "max": round(float(scores.max()), 4),
            "mean": round(float(scores.mean()), 4),
            "std": round(float(scores.std()), 4),
        },
    }
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Metrics artifact saved to %s", metrics_path)
    log.info("=== Training complete. ===")


if __name__ == "__main__":
    main()
