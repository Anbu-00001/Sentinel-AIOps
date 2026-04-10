"""
preprocess.py — Sentinel-AIOps Feature Engineering Pipeline
============================================================
Reads raw CI/CD logs, engineers three feature groups, fits transformers,
and saves all fitted artifacts to /models/ for use by the detector and
MCP inference server.

Workflow Rules (AGENTS.md):
  - Log 'Reasoning' before execution.
  - Save all ML artifacts as .joblib files.
"""

import logging
import os
from datetime import datetime, timedelta

import json

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("sentinel.preprocess")

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CSV = os.path.join(ROOT, "data", "raw", "ci_cd_pipeline_failure_logs_dataset.csv")
MODELS_DIR = os.path.join(ROOT, "models")
DATA_DIR = os.path.join(ROOT, "data")

# ── Feature column definitions ────────────────────────────────────────────────
NUMERICAL_COLS = [
    "build_duration_sec",
    "test_duration_sec",
    "deploy_duration_sec",
    "cpu_usage_pct",
    "memory_usage_mb",
    "retry_count",
]

# High-cardinality categoricals — hashed to fixed-size vectors
HIGH_CARD_COLS = ["repository", "author"]

# Low-cardinality categoricals encoded as dummies
LOW_CARD_COLS = ["ci_tool", "language", "os", "cloud_provider", "failure_stage", "severity"]

# Boolean columns (already 0/1 after cast)
BOOL_COLS = ["is_flaky_test", "rollback_triggered", "incident_created"]

# Unstructured text column
TEXT_COL = "error_message"

# Label column (kept aside — used only during validation, not training)
LABEL_COL = "failure_type"

HASH_N_FEATURES = 64   # bits for FeatureHasher (power-of-2 preferred)
TFIDF_MAX_FEATURES = 500


def generate_synthetic_baseline(num_samples: int = 1000) -> pd.DataFrame:
    """
    Generate mathematically sound synthetic data matching our schema.
    Used as fallback for CI/CD environments where raw Kaggle data is missing.
    """
    log.info("Reasoning: Generating %d synthetic baseline records for CI validation.", num_samples)
    
    ci_tools = ["Jenkins", "GitHub Actions", "GitLab CI", "CircleCI", "Travis CI"]
    languages = ["Python", "Java", "JavaScript", "Go", "Rust", "C++"]
    os_list = ["Linux", "Windows", "macOS"]
    cloud_providers = ["AWS", "GCP", "Azure", "On-Premise"]
    stages = ["build", "test", "deploy"]
    severities = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    failure_types = [
        "Build Failure", "Configuration Error", "Dependency Error",
        "Deployment Failure", "Network Error", "Permission Error",
        "Resource Exhaustion", "Security Scan Failure", "Test Failure", "Timeout"
    ]

    data = {
        "pipeline_id": [f"pipe-{i % 100}" for i in range(num_samples)],
        "run_id": [f"run-{i}" for i in range(num_samples)],
        "timestamp": [(datetime.now() - timedelta(minutes=i)).isoformat() for i in range(num_samples)],
        "ci_tool": np.random.choice(ci_tools, num_samples),
        "repository": [f"repo-{np.random.randint(0, 50)}" for _ in range(num_samples)],
        "branch": np.random.choice(["main", "develop", "feature", "hotfix"], num_samples),
        "commit_hash": [f"hash-{i}" for i in range(num_samples)],
        "author": [f"dev-{np.random.randint(0, 100)}" for _ in range(num_samples)],
        "language": np.random.choice(languages, num_samples),
        "os": np.random.choice(os_list, num_samples),
        "cloud_provider": np.random.choice(cloud_providers, num_samples),
        
        # Gaussian / Normal distributions for numerical features
        "build_duration_sec": np.random.normal(1800, 500, num_samples).clip(10, 3600).astype(int),
        "test_duration_sec": np.random.normal(300, 100, num_samples).clip(5, 600).astype(int),
        "deploy_duration_sec": np.random.normal(150, 50, num_samples).clip(5, 300).astype(int),
        "cpu_usage_pct": np.random.normal(50, 15, num_samples).clip(0, 100),
        "memory_usage_mb": np.random.normal(8192, 2048, num_samples).clip(256, 16384).astype(int),
        "retry_count": np.random.randint(0, 6, num_samples),
        
        "failure_stage": np.random.choice(stages, num_samples),
        "failure_type": np.random.choice(failure_types, num_samples),
        "error_code": [f"ERR_{np.random.randint(100, 999)}" for _ in range(num_samples)],
        "error_message": ["Synthetic validation log entry" for _ in range(num_samples)],
        "severity": np.random.choice(severities, num_samples),
        "is_flaky_test": np.random.choice([True, False], num_samples),
        "rollback_triggered": np.random.choice([True, False], num_samples),
        "incident_created": np.random.choice([True, False], num_samples),
    }
    
    return pd.DataFrame(data)


def load_raw(path: str) -> pd.DataFrame:
    """Load the raw CSV or fall back to synthetic data in CI."""
    try:
        log.info("Reasoning: Loading raw dataset from %s", path)
        df = pd.read_csv(path)
    except FileNotFoundError:
        if os.getenv("CI") or os.getenv("GITHUB_ACTIONS"):
            log.warning("WARNING: Raw dataset not found. CI Environment detected. "
                        "Falling back to Synthetic Data Generation for pipeline validation.")
            return generate_synthetic_baseline(num_samples=1000)
        else:
            log.error("CRITICAL: Local dataset missing and non-CI environment detected.")
            raise

    log.info("Loaded %d rows × %d columns.", df.shape[0], df.shape[1])
    assert df[NUMERICAL_COLS].isnull().sum().sum() == 0, "Unexpected nulls in numerical cols"
    assert TEXT_COL in df.columns, f"Expected column '{TEXT_COL}' not found"
    return df


def build_feature_matrix(
    df: pd.DataFrame,
    scaler: StandardScaler = None,
    hasher: FeatureHasher = None,
    tfidf: TfidfVectorizer = None,
    fit: bool = True,
):
    """
    Construct the full feature matrix by concatenating three groups:
      1. Scaled numerical metrics
      2. Hashed high-cardinality categoricals
      3. TF-IDF text features from error_message
      4. One-hot low-cardinality categoricals + boolean flags

    Parameters
    ----------
    df     : Input DataFrame
    scaler : Pre-fitted StandardScaler (or None to fit a new one)
    hasher : Pre-fitted FeatureHasher (or None to fit a new one)
    tfidf  : Pre-fitted TfidfVectorizer (or None to fit a new one)
    fit    : If True, fit the transformers; otherwise transform only.

    Returns
    -------
    X      : scipy sparse matrix (n_samples, n_features)
    scaler, hasher, tfidf: fitted transformer objects
    """

    # 1. Numerical — StandardScaler
    log.info("Reasoning: Scaling numerical features via StandardScaler.")
    num_data = df[NUMERICAL_COLS].astype(float)
    if fit:
        scaler = StandardScaler()
        num_scaled = scaler.fit_transform(num_data)
    else:
        num_scaled = scaler.transform(num_data)
    X_num = csr_matrix(num_scaled)

    # 2. High-cardinality categoricals — FeatureHasher
    log.info("Reasoning: Hashing high-cardinality cols %s with %d features.", HIGH_CARD_COLS, HASH_N_FEATURES)
    hash_input = df[HIGH_CARD_COLS].astype(str).to_dict(orient="records")
    if fit:
        hasher = FeatureHasher(n_features=HASH_N_FEATURES, input_type="dict")
        X_hash = hasher.fit_transform(hash_input)
    else:
        X_hash = hasher.transform(hash_input)

    # 3. TF-IDF on error_message
    log.info("Reasoning: Vectorising '%s' with TF-IDF (max_features=%d).", TEXT_COL, TFIDF_MAX_FEATURES)
    texts = df[TEXT_COL].fillna("").astype(str)
    if fit:
        tfidf = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, sublinear_tf=True)
        X_text = tfidf.fit_transform(texts)
    else:
        X_text = tfidf.transform(texts)

    # 4. Low-cardinality categoricals + booleans (dense → sparse)
    log.info("Reasoning: One-hot encoding low-cardinality categoricals and boolean flags.")
    df_dummies = pd.get_dummies(df[LOW_CARD_COLS], drop_first=True)
    bool_data = df[BOOL_COLS].astype(int)
    dense_extra = pd.concat([df_dummies, bool_data], axis=1).astype(float)
    X_extra = csr_matrix(dense_extra.values)

    # Horizontal stack all groups
    X = hstack([X_num, X_hash, X_text, X_extra], format="csr")
    log.info("Final feature matrix shape: %s", X.shape)
    return X, scaler, hasher, tfidf


def save_artifacts(scaler, hasher, tfidf, column_order: list):
    """Persist fitted transformers and column metadata to /models/."""
    log.info("Reasoning: Saving fitted transformers as joblib artifacts.")
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    joblib.dump(hasher, os.path.join(MODELS_DIR, "hasher.joblib"))
    joblib.dump(tfidf, os.path.join(MODELS_DIR, "tfidf.joblib"))
    meta = {
        "numerical_cols": NUMERICAL_COLS,
        "high_card_cols": HIGH_CARD_COLS,
        "low_card_cols": LOW_CARD_COLS,
        "bool_cols": BOOL_COLS,
        "text_col": TEXT_COL,
        "label_col": LABEL_COL,
        "hash_n_features": HASH_N_FEATURES,
        "tfidf_max_features": TFIDF_MAX_FEATURES,
        "dummy_column_order": column_order,
    }
    meta_path = os.path.join(MODELS_DIR, "feature_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Artifacts written: scaler.joblib, hasher.joblib, tfidf.joblib, feature_meta.json")


def main():
    log.info("=== Sentinel-AIOps | preprocess.py ===")
    df = load_raw(RAW_CSV)

    # Build column order for dummies (needed at inference time)
    df_dummies = pd.get_dummies(df[LOW_CARD_COLS], drop_first=True)
    column_order = list(df_dummies.columns)

    X, scaler, hasher, tfidf = build_feature_matrix(df, fit=True)

    # Save labels separately for validation use
    labels = df[LABEL_COL].values
    labels_path = os.path.join(DATA_DIR, "labels.npy")
    np.save(labels_path, labels)
    log.info("Labels saved to %s", labels_path)

    # Persist feature matrix
    matrix_path = os.path.join(DATA_DIR, "feature_matrix.npz")
    from scipy.sparse import save_npz
    save_npz(matrix_path, X)
    log.info("Feature matrix saved to %s", matrix_path)

    save_artifacts(scaler, hasher, tfidf, column_order)
    log.info("=== Preprocessing complete. ===")


if __name__ == "__main__":
    main()
