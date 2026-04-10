"""
drift_monitor.py — Sentinel-AIOps Feature Drift Analytics
============================================================
Compares a sliding window of live logs against the training distribution
using:
  - Population Stability Index (PSI) for numerical features
  - Chi-Square test for categorical feature drift

Outputs:
  - drift_report.json in /models/
  - Updates registry.json with retrain_suggested flag

Workflow Rules (AGENTS.md):
  - Log 'Reasoning' before execution.
  - Save all ML metrics as Artifacts.
  - Type hinting + Pydantic validation throughout.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional


import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy.stats import chi2_contingency

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("sentinel.drift")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
DATA_DIR = os.path.join(ROOT, "data")

# ── PSI threshold ─────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, ROOT)
from config import PSI_SEVERE_THRESHOLD, PSI_BINS

PSI_THRESHOLD: float = PSI_SEVERE_THRESHOLD

# ── Feature lists (consistent with preprocess.py) ─────────────────────────────
NUMERICAL_COLS: list[str] = [
    "build_duration_sec", "test_duration_sec", "deploy_duration_sec",
    "cpu_usage_pct", "memory_usage_mb", "retry_count",
]

CATEGORICAL_COLS: list[str] = [
    "ci_tool", "language", "os", "cloud_provider",
    "failure_stage", "severity",
]


# ── Pydantic models ──────────────────────────────────────────────────────────

class FeatureDrift(BaseModel):
    """Drift result for a single feature."""
    feature: str
    method: str  # "PSI" or "Chi-Square"
    score: float
    p_value: Optional[float] = None
    is_drifted: bool
    severity: str  # "none", "moderate", "severe"


class DriftReport(BaseModel):
    """Complete drift report across all features."""
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    window_size: int
    features_analyzed: int
    features_drifted: int
    retrain_suggested: bool
    psi_threshold: float = PSI_THRESHOLD
    feature_results: list[FeatureDrift]


# ── PSI Calculation ───────────────────────────────────────────────────────────

def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = PSI_BINS,
) -> float:
    """
    Compute Population Stability Index between reference and current distributions.

    PSI = Σ (P_i - Q_i) * ln(P_i / Q_i)
    where P = current proportions, Q = reference proportions

    Interpretation:
      PSI < 0.1  → No significant change
      0.1 ≤ PSI < 0.2  → Moderate drift
      PSI ≥ 0.2  → Significant drift (retrain required)
    """
    # Create bins from reference distribution
    min_val = min(reference.min(), current.min())
    max_val = max(reference.max(), current.max())
    bins = np.linspace(min_val - 1e-6, max_val + 1e-6, n_bins + 1)

    ref_counts, _ = np.histogram(reference, bins=bins)
    cur_counts, _ = np.histogram(current, bins=bins)

    # ── Epsilon Smoothing ──
    # Epsilon Smoothing prevents mathematical asymptotes during OOD data injection.
    EPSILON: float = 1e-4

    # Calculate raw proportions
    ref_pct = ref_counts / ref_counts.sum()
    cur_pct = cur_counts / cur_counts.sum()

    # Apply smoothing: value = max(value, EPSILON)
    ref_pct = np.maximum(ref_pct, EPSILON)
    cur_pct = np.maximum(cur_pct, EPSILON)

    # Re-normalize so they sum to 1.0
    ref_pct /= ref_pct.sum()
    cur_pct /= cur_pct.sum()

    psi: float = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return psi


def classify_psi(psi: float) -> str:
    """Map PSI score to severity level."""
    if psi < 0.1:
        return "none"
    elif psi < PSI_THRESHOLD:
        return "moderate"
    else:
        return "severe"


# ── Chi-Square for categoricals ───────────────────────────────────────────────

def compute_chi_square(
    reference: pd.Series,
    current: pd.Series,
) -> tuple[float, float]:
    """
    Compute Chi-Square statistic for categorical drift.
    Returns (chi2_statistic, p_value).
    """
    # Build contingency table
    all_cats = set(reference.unique()) | set(current.unique())
    ref_counts = reference.value_counts().reindex(list(all_cats), fill_value=0)
    cur_counts = current.value_counts().reindex(list(all_cats), fill_value=0)

    # Contingency table: rows = [reference, current], cols = categories
    contingency = np.array([ref_counts.values, cur_counts.values])

    # Remove columns that are all zeros
    non_zero_mask = contingency.sum(axis=0) > 0
    contingency = contingency[:, non_zero_mask]

    if contingency.shape[1] < 2:
        return 0.0, 1.0

    chi2, p_value, _, _ = chi2_contingency(contingency)
    return float(chi2), float(p_value)


# ── Main Drift Analysis ──────────────────────────────────────────────────────

def load_training_data() -> pd.DataFrame:
    """Load the original training dataset for reference distribution."""
    log.info("Reasoning: Loading training data as reference distribution.")
    csv_path = os.path.join(DATA_DIR, "raw", "ci_cd_pipeline_failure_logs_dataset.csv")
    df = pd.read_csv(csv_path)
    log.info("Training reference: %d rows.", len(df))
    return df


def analyze_drift(
    live_logs: list[dict],
    reference_df: Optional[pd.DataFrame] = None,
) -> DriftReport:
    """
    Compare live logs against training distribution.

    Parameters
    ----------
    live_logs   : List of log dicts (from stream_simulator or real pipeline)
    reference_df: Training DataFrame (loaded automatically if None)

    Returns
    -------
    DriftReport with per-feature PSI/Chi-Square results.
    """
    log.info("Reasoning: Starting drift analysis on %d live logs.", len(live_logs))

    if reference_df is None:
        reference_df = load_training_data()

    live_df = pd.DataFrame(live_logs)
    results: list[FeatureDrift] = []

    # ── Numerical features: PSI ───────────────────────────────────────────────
    log.info("Reasoning: Computing PSI for %d numerical features.", len(NUMERICAL_COLS))
    for col in NUMERICAL_COLS:
        if col not in live_df.columns:
            log.warning("Skipping %s — not in live data.", col)
            continue

        ref_vals = reference_df[col].dropna().values.astype(float)
        cur_vals = live_df[col].dropna().values.astype(float)

        psi_score = compute_psi(ref_vals, cur_vals)
        severity = classify_psi(psi_score)

        results.append(FeatureDrift(
            feature=col,
            method="PSI",
            score=round(psi_score, 4),
            p_value=None,
            is_drifted=psi_score >= PSI_THRESHOLD,
            severity=severity,
        ))
        log.info("  %s: PSI=%.4f → %s", col, psi_score, severity)

    # ── Categorical features: Chi-Square ─────────────────────────────────────
    log.info("Reasoning: Computing Chi-Square for %d categorical features.", len(CATEGORICAL_COLS))
    for col in CATEGORICAL_COLS:
        if col not in live_df.columns:
            log.warning("Skipping %s — not in live data.", col)
            continue

        chi2, p_val = compute_chi_square(reference_df[col], live_df[col])
        is_drifted = p_val < 0.05
        severity = "severe" if p_val < 0.001 else ("moderate" if p_val < 0.05 else "none")

        results.append(FeatureDrift(
            feature=col,
            method="Chi-Square",
            score=round(chi2, 4),
            p_value=round(p_val, 6),
            is_drifted=is_drifted,
            severity=severity,
        ))
        log.info("  %s: χ²=%.4f, p=%.6f → %s", col, chi2, p_val, severity)

    # ── Build report ──────────────────────────────────────────────────────────
    n_drifted = sum(1 for r in results if r.is_drifted)
    retrain_needed = any(r.method == "PSI" and r.score >= PSI_THRESHOLD for r in results)

    report = DriftReport(
        window_size=len(live_logs),
        features_analyzed=len(results),
        features_drifted=n_drifted,
        retrain_suggested=retrain_needed,
        feature_results=results,
    )

    # ── Save drift_report.json ────────────────────────────────────────────────
    report_path = os.path.join(MODELS_DIR, "drift_report.json")
    with open(report_path, "w") as f:
        f.write(report.model_dump_json(indent=2))
    log.info("Drift report saved to %s", report_path)

    # ── Update registry.json ──────────────────────────────────────────────────
    _update_registry(retrain_needed)

    return report


def _update_registry(retrain_suggested: bool) -> None:
    """Update registry.json with retrain_suggested flag."""
    reg_path = os.path.join(MODELS_DIR, "registry.json")
    if os.path.exists(reg_path):
        with open(reg_path) as f:
            registry = json.load(f)
    else:
        registry = {"models": []}

    registry["retrain_suggested"] = retrain_suggested
    registry["last_drift_check"] = datetime.now(timezone.utc).isoformat()

    with open(reg_path, "w") as f:
        json.dump(registry, f, indent=2)
    log.info("Registry updated → retrain_suggested=%s", retrain_suggested)


# ── CLI Entry Point ───────────────────────────────────────────────────────────

def main() -> None:
    """Run drift analysis using stream simulator data."""
    log.info("=== Sentinel-AIOps | drift_monitor.py ===")

    # Import stream simulator
    import sys
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from stream_simulator import ChaosLevel, log_generator

    # Generate a window of 200 live logs with moderate chaos
    log.info("Reasoning: Generating 200 live logs with MEDIUM chaos for drift test.")
    live_logs = [record.model_dump() for record in log_generator(200, ChaosLevel.MEDIUM)]

    report = analyze_drift(live_logs)

    log.info("=== Drift Analysis Complete ===")
    log.info("Features drifted: %d/%d | Retrain suggested: %s",
             report.features_drifted, report.features_analyzed, report.retrain_suggested)


if __name__ == "__main__":
    main()
