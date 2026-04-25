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

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crypto_sig

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
LOW_CARD_COLS = [
    "ci_tool",
    "language",
    "os",
    "cloud_provider",
    "failure_stage",
    "severity",
]

# Boolean columns (already 0/1 after cast)
BOOL_COLS = ["is_flaky_test", "rollback_triggered", "incident_created"]

# Unstructured text column
TEXT_COL = "error_message"

# Label column (kept aside — used only during validation, not training)
LABEL_COL = "failure_type"

HASH_N_FEATURES = 64  # bits for FeatureHasher (power-of-2 preferred)
TFIDF_MAX_FEATURES = 600


# Per-feature noise sigma fractions.
# deploy_duration_sec uses a smaller fraction because its
# physical range (5-300s) is narrow relative to class means
# near the upper boundary. Global sigma would cause >25% of
# rows to hit the 300s clip ceiling.
FEATURE_NOISE_SIGMA = {
    "build_duration_sec": 0.65,
    "test_duration_sec": 0.65,
    "deploy_duration_sec": 0.12,   # narrow range, reduce noise
    "cpu_usage_pct": 0.65,
    "memory_usage_mb": 0.65,
    "retry_count": 0.40,   # discrete int, less noise
}

CLIP_BOUNDS = {
    "build_duration_sec": (10, 3600),
    "test_duration_sec": (5, 600),
    "deploy_duration_sec": (5, 300),
    "cpu_usage_pct": (0, 100),
    "memory_usage_mb": (256, 16384),
    "retry_count": (0, 10),
}


def generate_synthetic_baseline(num_samples: int = 10000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic CI/CD log data with signal-injected feature distributions.

    Each of the 10 failure types has statistically distinct, realistic feature
    distributions so that labels are *derived from* features rather than randomly
    assigned. This gives LightGBM learnable signal even when the real Kaggle
    dataset is unavailable (e.g. in CI).

    - Generic error messages are used intentionally so TF-IDF serves only as a weak auxiliary signal.
    - Each failure class has 2-3 PRIMARY numerical features that distinguish it:
      * Resource Exhaustion: cpu_usage_pct HIGH + memory_usage_mb HIGH
      * Timeout: build_duration_sec VERY HIGH + test_duration_sec HIGH
      * Build Failure: build_duration_sec LOW + test_duration_sec VERY LOW
      * Test Failure: test_duration_sec MODERATE-HIGH + failure_stage=test
      * Deployment Failure: deploy_duration_sec HIGH + rollback_triggered HIGH
      * Network Error: cpu_usage_pct VERY LOW + retry_count HIGH
      * Dependency Error: build_duration_sec VERY LOW + retry_count LOW + cpu_usage_pct MODERATE-LOW
      * Configuration Error: build_duration_sec VERY LOW + retry_count ZERO OR ONE + deploy_duration_sec VERY LOW
      * Permission Error: cpu_usage_pct LOW-MODERATE + incident_created HIGH + failure_stage=deploy HIGH
      * Security Scan Failure: build_duration_sec HIGH + test_duration_sec HIGH + incident_created VERY HIGH + severity=CRITICAL HIGH
    - Gaussian noise (8% of feature std) is applied post-generation to create realistic boundary ambiguity.
    - Expected F1 range after training is [0.60, 0.90].

    Note: ML metrics (F1-Score, PR AUC, ablation delta) are saved as Artifacts per AGENTS.md workflow rules.

    Parameters
    ----------
    num_samples : int
        Total number of rows to generate. Each class receives
        ``num_samples // 10`` rows for balanced representation.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame matching the full Sentinel-AIOps schema.
    """
    log.info(
        "Reasoning: Generating %d synthetic baseline records with per-class "
        "signal injection for CI validation.",
        num_samples,
    )
    np.random.seed(seed)

    GENERIC_ERROR_TEMPLATES = [
        "Process exited with non-zero status code {code} during pipeline execution",
        "Job failed at step {step} after {elapsed}s of runtime",
        "Stage did not complete within the allocated runner time",
        "Pipeline runner reported an unrecoverable error on attempt {attempt}",
        "Execution terminated unexpectedly — check runner logs for details",
        "Task aborted: upstream dependency returned exit code {code}",
        "Agent lost connection to coordinator after {elapsed}s",
        "Workflow step {step} produced no output and exited",
        "Internal runner error: process killed by signal {signal}",
        "Job exceeded allocated resources and was forcibly stopped",
        "Artifact upload failed after job completion",
        "Health check did not pass before pipeline timeout",
        "Workspace cleanup triggered premature job termination",
        "Runner capacity exceeded — job queued then dropped",
        "Step {step} returned unexpected exit status {code}",
        "Pipeline orchestrator lost heartbeat from worker node",
        "Job cancelled by scheduler due to resource contention",
        "Dependency resolution did not complete before stage timeout",
        "Retry limit reached after {attempt} consecutive failures",
        "Configuration could not be validated before pipeline start",
    ]

    def _make_error_message() -> str:
        tpl = np.random.choice(GENERIC_ERROR_TEMPLATES)
        return tpl.format(
            code=np.random.choice([1, 2, 126, 127, 137, 139, 143]),
            step=np.random.randint(1, 21),
            elapsed=np.random.randint(30, 3501),
            attempt=np.random.randint(1, 6),
            signal=np.random.choice([9, 11, 15]),
        )

    # ── Shared categorical pools ────────────────────────────────────────────
    ci_tools = ["Jenkins", "GitHub Actions", "GitLab CI", "CircleCI", "Travis CI"]
    languages = ["Python", "Java", "JavaScript", "Go", "Rust", "C++"]
    os_list = ["Linux", "Windows", "macOS"]
    cloud_providers = ["AWS", "GCP", "Azure", "On-Premise"]

    n_per_class: int = num_samples // 10

    # ── Per-class distribution specs ────────────────────────────────────────
    class_specs: dict[str, dict] = {
        "Resource Exhaustion": {
            "cpu_usage_pct": ("normal", 85, 12, 60, 100),
            "memory_usage_mb": ("normal", 12500, 2000, 7000, 16384),
            "build_duration_sec": ("normal", 2400, 500, 1000, 3600),
            "test_duration_sec": ("normal", 280, 80, 100, 500),
            "deploy_duration_sec": ("normal", 130, 50, 40, 280),
            "retry_count": ("randint", 2, 6),
            "failure_stage": (["deploy", "build", "test"], [0.50, 0.35, 0.15]),
            "severity": (["CRITICAL", "HIGH"], [0.65, 0.35]),
            "rollback_override": (True, 0.65),
            "incident_override": (True, 0.70),
            "flaky_override": (False, 0.90),
        },
        "Timeout": {
            "build_duration_sec": ("normal", 3300, 200, 2600, 3600),
            "test_duration_sec": ("normal", 550, 40, 440, 600),
            "deploy_duration_sec": ("normal", 260, 30, 180, 300),
            "cpu_usage_pct": ("normal", 45, 15, 15, 75),
            "memory_usage_mb": ("normal", 6000, 2000, 1500, 12000),
            "retry_count": ("randint", 3, 6),
            "failure_stage": (["build", "test", "deploy"], [0.50, 0.30, 0.20]),
            "severity": (["HIGH", "CRITICAL"], [0.60, 0.40]),
            "rollback_override": (False, 0.70),
            "incident_override": (True, 0.55),
            "flaky_override": (False, 0.80),
        },
        "Build Failure": {
            "build_duration_sec": ("normal", 380, 120, 60, 700),
            "test_duration_sec": ("normal", 18, 8, 5, 45),
            "deploy_duration_sec": ("normal", 10, 4, 5, 25),
            "cpu_usage_pct": ("normal", 68, 10, 50, 88),
            "memory_usage_mb": ("normal", 6000, 1500, 2000, 10000),
            "retry_count": ("randint", 0, 2),
            "failure_stage": (["build", "test", "deploy"], [0.82, 0.12, 0.06]),
            "severity": (["HIGH", "MEDIUM", "CRITICAL"], [0.50, 0.30, 0.20]),
            "rollback_override": (False, 0.85),
            "incident_override": (False, 0.70),
            "flaky_override": (False, 0.85),
        },
        "Test Failure": {
            "build_duration_sec": ("normal", 1750, 220, 1200, 2400),
            "test_duration_sec": ("normal", 460, 55, 300, 590),
            "deploy_duration_sec": ("normal", 14, 5, 5, 35),
            "cpu_usage_pct": ("normal", 52, 12, 30, 78),
            "memory_usage_mb": ("normal", 7000, 1800, 2500, 12000),
            "retry_count": ("randint", 0, 3),
            "failure_stage": (["test", "build", "deploy"], [0.78, 0.15, 0.07]),
            "severity": (["MEDIUM", "HIGH"], [0.65, 0.35]),
            "rollback_override": (False, 0.80),
            "incident_override": (False, 0.65),
            "flaky_override": (True, 0.45),
        },
        "Deployment Failure": {
            "build_duration_sec": ("normal", 1900, 300, 1100, 2800),
            "test_duration_sec": ("normal", 340, 70, 180, 520),
            "deploy_duration_sec": ("normal", 270, 25, 200, 300),
            "cpu_usage_pct": ("normal", 58, 14, 30, 85),
            "memory_usage_mb": ("normal", 8000, 2000, 2500, 14000),
            "retry_count": ("randint", 1, 4),
            "failure_stage": (["deploy", "test", "build"], [0.80, 0.12, 0.08]),
            "severity": (["HIGH", "CRITICAL"], [0.50, 0.50]),
            "rollback_override": (True, 0.80),
            "incident_override": (True, 0.70),
            "flaky_override": (False, 0.90),
        },
        "Network Error": {
            "build_duration_sec": ("normal", 1100, 400, 300, 2200),
            "test_duration_sec": ("normal", 180, 70, 50, 420),
            "deploy_duration_sec": ("normal", 90, 40, 20, 220),
            "cpu_usage_pct": ("normal", 22, 10, 5, 50),
            "memory_usage_mb": ("normal", 4000, 1500, 500, 8500),
            "retry_count": ("randint", 4, 6),
            "failure_stage": (["build", "deploy", "test"], [0.40, 0.40, 0.20]),
            "severity": (["MEDIUM", "HIGH", "LOW"], [0.50, 0.40, 0.10]),
            "rollback_override": (False, 0.60),
            "incident_override": (True, 0.50),
            "flaky_override": (False, 0.75),
        },
        "Dependency Error": {
            "build_duration_sec": ("normal", 280, 90, 80, 550),
            "test_duration_sec": ("normal", 22, 9, 5, 55),
            "deploy_duration_sec": ("normal", 10, 4, 5, 25),
            "cpu_usage_pct": ("normal", 38, 9, 18, 58),
            "memory_usage_mb": ("normal", 4000, 900, 1000, 7000),
            "retry_count": ("randint", 0, 2),
            "failure_stage": (["build", "test"], [0.80, 0.20]),
            "severity": (["MEDIUM", "HIGH", "LOW"], [0.60, 0.30, 0.10]),
            "rollback_override": (False, 0.90),
            "incident_override": (False, 0.80),
            "flaky_override": (False, 0.88),
        },
        "Configuration Error": {
            "build_duration_sec": ("normal", 220, 70, 60, 420),
            "test_duration_sec": ("normal", 12, 5, 5, 30),
            "deploy_duration_sec": ("normal", 35, 12, 12, 70),
            "cpu_usage_pct": ("normal", 22, 7, 8, 42),
            "memory_usage_mb": ("normal", 3000, 700, 800, 5500),
            "retry_count": ("randint", 0, 1),
            "failure_stage": (["build", "deploy"], [0.50, 0.50]),
            "severity": (["MEDIUM", "LOW", "HIGH"], [0.70, 0.20, 0.10]),
            "rollback_override": (False, 0.95),
            "incident_override": (False, 0.85),
            "flaky_override": (False, 0.98),
        },
        "Permission Error": {
            "build_duration_sec": ("normal", 580, 140, 200, 1000),
            "test_duration_sec": ("normal", 45, 18, 10, 100),
            "deploy_duration_sec": ("normal", 45, 14, 15, 90),
            "cpu_usage_pct": ("normal", 32, 8, 14, 52),
            "memory_usage_mb": ("normal", 4500, 1000, 1500, 7500),
            "retry_count": ("randint", 0, 2),
            "failure_stage": (["deploy", "build"], [0.65, 0.35]),
            "severity": (["HIGH", "CRITICAL", "MEDIUM"], [0.60, 0.30, 0.10]),
            "rollback_override": (False, 0.70),
            "incident_override": (True, 0.75),
            "flaky_override": (False, 0.95),
        },
        "Security Scan Failure": {
            "build_duration_sec": ("normal", 2300, 180, 1900, 2800),
            "test_duration_sec": ("normal", 420, 45, 320, 530),
            "deploy_duration_sec": ("normal", 12, 5, 5, 28),
            "cpu_usage_pct": ("normal", 62, 10, 40, 82),
            "memory_usage_mb": ("normal", 9000, 1500, 5000, 13000),
            "retry_count": ("randint", 0, 1),
            "failure_stage": (["test", "build"], [0.70, 0.30]),
            "severity": (["CRITICAL", "HIGH"], [0.70, 0.30]),
            "rollback_override": (False, 0.85),
            "incident_override": (True, 0.85),
            "flaky_override": (False, 0.92),
        },
    }

    def _sample_normal(spec: tuple, n: int) -> np.ndarray:
        _, mu, sigma, lo, hi = spec
        return np.random.normal(mu, sigma, n).clip(lo, hi)

    def _sample_randint(spec: tuple, n: int) -> np.ndarray:
        _, lo, hi = spec
        return np.random.randint(lo, hi + 1, n)

    # ── Build per-class DataFrames ──────────────────────────────────────────
    frames: list[pd.DataFrame] = []
    for ft, spec in class_specs.items():
        log.info("Reasoning: Generating synthetic rows for class '%s'.", ft)
        n = n_per_class
        row: dict[str, np.ndarray] = {}

        # Numerical features
        for col in ["build_duration_sec", "test_duration_sec",
                    "deploy_duration_sec", "cpu_usage_pct", "memory_usage_mb"]:
            s = spec[col]
            row[col] = _sample_normal(s, n).astype(int) if col != "cpu_usage_pct" \
                else np.round(_sample_normal(s, n), 1)

        rc_spec = spec["retry_count"]
        row["retry_count"] = _sample_randint(rc_spec, n)

        # Weighted categoricals
        stage_opts, stage_weights = spec["failure_stage"]
        row["failure_stage"] = np.random.choice(stage_opts, n, p=stage_weights)

        sev_opts, sev_weights = spec["severity"]
        row["severity"] = np.random.choice(sev_opts, n, p=sev_weights)

        # Error messages will be generated globally later

        # Boolean overrides
        if "rollback_override" in spec:
            val, prob = spec["rollback_override"]
            row["rollback_triggered"] = np.random.choice(
                [val, not val], n, p=[prob, 1 - prob]
            )
        else:
            row["rollback_triggered"] = np.random.choice([True, False], n)

        if "incident_override" in spec:
            val, prob = spec["incident_override"]
            row["incident_created"] = np.random.choice(
                [val, not val], n, p=[prob, 1 - prob]
            )
        else:
            row["incident_created"] = np.random.choice([True, False], n)

        if "flaky_override" in spec:
            val, prob = spec["flaky_override"]
            row["is_flaky_test"] = np.random.choice(
                [val, not val], n, p=[prob, 1 - prob]
            )
        else:
            row["is_flaky_test"] = np.random.choice([True, False], n)

        # Global-distribution fields (not class-specific)
        row["pipeline_id"] = np.array([f"pipe-{i % 100}" for i in range(n)])
        row["run_id"] = np.array([f"run-{ft[:3].lower()}-{i}" for i in range(n)])
        row["timestamp"] = np.array([
            (datetime.now() - timedelta(minutes=i)).isoformat() for i in range(n)
        ])
        row["ci_tool"] = np.random.choice(ci_tools, n)
        row["repository"] = np.array([f"repo-{np.random.randint(0, 50)}" for _ in range(n)])
        row["branch"] = np.random.choice(["main", "develop", "feature", "hotfix"], n)
        row["commit_hash"] = np.array([f"hash-{ft[:3].lower()}-{i}" for i in range(n)])
        row["author"] = np.array([f"dev-{np.random.randint(0, 100)}" for _ in range(n)])
        row["language"] = np.random.choice(languages, n)
        row["os"] = np.random.choice(os_list, n)
        row["cloud_provider"] = np.random.choice(cloud_providers, n)
        row["error_code"] = np.array([f"ERR_{np.random.randint(100, 999)}" for _ in range(n)])
        row["failure_type"] = np.array([ft] * n)

        frames.append(pd.DataFrame(row))

    df = pd.concat(frames, ignore_index=True)

    log.info("Reasoning: Injecting per-feature Gaussian noise and re-clipping.")
    for feat, sigma_frac in FEATURE_NOISE_SIGMA.items():
        sigma = df[feat].std() * sigma_frac
        rng_noise = np.random.default_rng(seed + hash(feat) & 0xFFFF)
        noise = rng_noise.normal(0, sigma, size=len(df))
        df[feat] = df[feat] + noise

        # Re-clip to physical bounds
        lo, hi = CLIP_BOUNDS[feat]
        df[feat] = df[feat].clip(lo, hi)
        if feat != "cpu_usage_pct":
            df[feat] = np.round(df[feat]).astype(int)

    log.info("Reasoning: Shuffling final dataset.")
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    log.info("Reasoning: Generating error_message for all rows from a single global sequence.")
    df["error_message"] = [_make_error_message() for _ in range(len(df))]
    log.info("Synthetic baseline generated: %d rows × %d columns.", df.shape[0], df.shape[1])
    return df


def load_raw(path: str) -> pd.DataFrame:
    """Load the raw CSV or fall back to synthetic data in CI."""
    try:
        log.info("Reasoning: Loading raw dataset from %s", path)
        df = pd.read_csv(path)
    except FileNotFoundError:
        if os.getenv("CI") or os.getenv("GITHUB_ACTIONS"):
            log.warning(
                "WARNING: Raw dataset not found. CI Environment detected. "
                "Falling back to Synthetic Data Generation for pipeline validation."
            )
            return generate_synthetic_baseline(num_samples=10000)
        else:
            log.error(
                "CRITICAL: Local dataset missing and non-CI environment detected."
            )
            raise

    log.info("Loaded %d rows × %d columns.", df.shape[0], df.shape[1])
    assert (
        df[NUMERICAL_COLS].isnull().sum().sum() == 0
    ), "Unexpected nulls in numerical cols"
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
    log.info(
        "Reasoning: Hashing high-cardinality cols %s with %d features.",
        HIGH_CARD_COLS,
        HASH_N_FEATURES,
    )
    hash_input = df[HIGH_CARD_COLS].astype(str).to_dict(orient="records")
    if fit:
        hasher = FeatureHasher(n_features=HASH_N_FEATURES, input_type="dict")
        X_hash = hasher.fit_transform(hash_input)
    else:
        X_hash = hasher.transform(hash_input)

    # 3. TF-IDF on error_message
    log.info(
        "Reasoning: Vectorising '%s' with TF-IDF (max_features=%d).",
        TEXT_COL,
        TFIDF_MAX_FEATURES,
    )
    texts = df[TEXT_COL].fillna("").astype(str)
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    custom_stop_words = list(ENGLISH_STOP_WORDS.union({"job", "step", "runner", "pipeline", "process", "error"}))
    if fit:
        tfidf = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, sublinear_tf=True, stop_words=custom_stop_words)
        X_text = tfidf.fit_transform(texts)
    else:
        X_text = tfidf.transform(texts)

    # 4. Low-cardinality categoricals + booleans (dense → sparse)
    log.info(
        "Reasoning: One-hot encoding low-cardinality categoricals and boolean flags."
    )
    df_dummies = pd.get_dummies(df[LOW_CARD_COLS], drop_first=True)
    bool_data = df[BOOL_COLS].astype(int)
    dense_extra = pd.concat([df_dummies, bool_data], axis=1).astype(float)
    X_extra = csr_matrix(dense_extra.values)

    # Horizontal stack all groups
    X = hstack([X_num, X_hash, X_text, X_extra], format="csr")
    log.info("Final feature matrix shape: %s", X.shape)
    return X, scaler, hasher, tfidf


def save_artifacts(scaler, hasher, tfidf, column_order: list, total_features: int):
    """Persist fitted transformers and column metadata to /models/."""
    log.info("Reasoning: Saving fitted transformers as joblib artifacts.")
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    crypto_sig.sign_artifact(os.path.join(MODELS_DIR, "scaler.joblib"))

    joblib.dump(hasher, os.path.join(MODELS_DIR, "hasher.joblib"))
    crypto_sig.sign_artifact(os.path.join(MODELS_DIR, "hasher.joblib"))

    joblib.dump(tfidf, os.path.join(MODELS_DIR, "tfidf.joblib"))
    crypto_sig.sign_artifact(os.path.join(MODELS_DIR, "tfidf.joblib"))

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
        "total_features": total_features,
        "artifacts_timestamp": datetime.now().isoformat(),
    }
    meta_path = os.path.join(MODELS_DIR, "feature_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info(
        "Artifacts written & signed: scaler, hasher, tfidf, feature_meta.json"
    )


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

    save_artifacts(scaler, hasher, tfidf, column_order, X.shape[1])
    log.info("=== Preprocessing complete. ===")


if __name__ == "__main__":
    main()
