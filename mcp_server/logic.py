"""
logic.py — Shared ML Inference Logic for Sentinel-AIOps
=======================================================
Lightweight module for feature validation and transformation,
decoupling ML core from FastMCP and Prometheus infrastructure.
"""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack

log = logging.getLogger("sentinel.logic")

# ── Schema Configuration ─────────────────────────────────────────────────────
REQUIRED_KEYS: list[str] = [
    "build_duration_sec",
    "test_duration_sec",
    "deploy_duration_sec",
    "cpu_usage_pct",
    "memory_usage_mb",
    "retry_count",
    "error_message",
]

OPTIONAL_KEYS: dict[str, Any] = {
    "repository": "",
    "author": "",
    "failure_stage": "build",
    "severity": "MEDIUM",
    "ci_tool": "Jenkins",
    "language": "Python",
    "os": "Linux",
    "cloud_provider": "AWS",
    "is_flaky_test": False,
    "rollback_triggered": False,
    "incident_created": False,
}


def validate_input(features: Dict[str, Any]) -> list[str]:
    """Validate required keys and numeric types."""
    errors: list[str] = []
    for key in REQUIRED_KEYS:
        if key not in features:
            errors.append(f"Missing required field: '{key}'")
    for key in [
        "build_duration_sec", "test_duration_sec",
        "deploy_duration_sec", "cpu_usage_pct",
        "memory_usage_mb", "retry_count",
    ]:
        if key in features:
            try:
                float(features[key])
            except (ValueError, TypeError):
                errors.append(
                    f"Field '{key}' must be numeric, got: {
                        type(
                            features[key]).__name__}")
    return errors


def transform_input(features: Dict[str, Any],
                    scaler, hasher, tfidf, meta: Dict[str, Any]):
    """Apply the 4-group transformation pipeline (Num, Hash, Text, Extra)."""
    # Fill defaults for optional keys
    feats = dict(features)
    for key, default in OPTIONAL_KEYS.items():
        if key not in feats:
            feats[key] = default

    df = pd.DataFrame([feats])

    # 1. Numerical scaling
    X_num = csr_matrix(
        scaler.transform(df[meta["numerical_cols"]].astype(float))
    )

    # 2. High-cardinality hashing
    hash_input = (
        df[meta["high_card_cols"]].astype(str).to_dict(orient="records")
    )
    X_hash = hasher.transform(hash_input)

    # 3. Text vectorization (TF-IDF)
    X_text = tfidf.transform(
        df[meta["text_col"]].fillna("").astype(str)
    )

    # 4. Low-cardinality dummies + Boolean flags
    df_dummies = pd.get_dummies(df[meta["low_card_cols"]], drop_first=True)
    df_dummies = df_dummies.reindex(
        columns=meta["dummy_column_order"], fill_value=0
    )
    bool_data = df[meta["bool_cols"]].astype(int)
    X_extra = csr_matrix(
        pd.concat([df_dummies, bool_data], axis=1).astype(float).values
    )

    return hstack([X_num, X_hash, X_text, X_extra], format="csr")


def get_top_features(
        X_row, feature_names: list[str], n: int = 5) -> list[dict[str, Any]]:
    """Return top-N contributing feature names by absolute value."""
    if hasattr(X_row, "toarray"):
        arr = np.abs(X_row.toarray().flatten())
    else:
        arr = np.abs(X_row.flatten())

    top_idx = np.argsort(arr)[-n:][::-1]
    results: list[dict[str, Any]] = []
    for idx in top_idx:
        name = (
            feature_names[idx]
            if idx < len(feature_names)
            else f"f_{idx}"
        )
        results.append(
            {"feature": name, "value": round(float(arr[idx]), 4)}
        )
    return results


def run_prediction(features: Dict[str,
                                  Any],
                   model,
                   le,
                   scaler,
                   hasher,
                   tfidf,
                   meta: Dict[str,
                              Any],
                   feature_names: list[str]):
    """Execute end-to-end prediction logic."""
    X = transform_input(features, scaler, hasher, tfidf, meta)

    proba = model.predict_proba(X)[0]
    pred_idx = int(np.argmax(proba))
    confidence = float(proba[pred_idx])
    prediction: str = le.inverse_transform([pred_idx])[0]

    top_feats = get_top_features(X, feature_names, n=5)

    return {
        "prediction": prediction,
        "confidence": confidence,
        "top_features": top_feats,
        "transformed_matrix": X
    }
