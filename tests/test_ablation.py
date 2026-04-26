"""
test_ablation.py — Sentinel-AIOps Ablation CI Gate
====================================================
Verifies that numerical telemetry features carry the primary
classification signal. If this test fails, error_message templates
likely contain class-specific keywords, making the model a text
classifier rather than a telemetry-driven AIOps system.

Thresholds:
  WITHOUT_TFIDF_F1_MIN = 0.55  (numerical signal floor)
  FULL_MODEL_F1_MAX    = 0.90  (memorization ceiling)
  DELTA_MAX            = 0.30  (max text contribution)

These thresholds are intentionally conservative. The current
system achieves without-TF-IDF F1 ≈ 0.89. A drop below 0.55
indicates feature signal regression, not normal variance.
"""

import os
import sys
import json
import pytest
import numpy as np  # noqa: F401

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("TESTING", "1")
os.environ.setdefault("CI", "1")

WITHOUT_TFIDF_F1_MIN = 0.55
FULL_MODEL_F1_MAX = 0.90
DELTA_MAX = 0.30
ABLATION_SAMPLE_SIZE = 2000   # smaller than full 10k for CI speed


@pytest.fixture(scope="module")
def ablation_results():
    """
    Build feature matrices with and without TF-IDF, train two
    LightGBM models on identical splits, and return their F1 scores.
    Uses a small sample size for CI speed while remaining
    statistically meaningful for the threshold checks.
    """
    import joblib
    import os
    MODELS_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models"
    )
    REQUIRED_ARTIFACTS = [
        "scaler.joblib",
        "hasher.joblib",
        "tfidf.joblib",
        "lgbm_model.joblib",
        "label_encoder.joblib",
        "feature_meta.json",
    ]
    missing = [
        a for a in REQUIRED_ARTIFACTS
        if not os.path.exists(os.path.join(MODELS_DIR, a))
    ]
    if missing:
        pytest.skip(
            f"Ablation test requires trained model artifacts. "
            f"Missing: {missing}. "
            f"Run 'make train' first, then re-run this test. "
            f"In CI, add a training step before pytest in ci.yml."
        )

    import lightgbm as lgb
    import pandas as pd
    from scipy.sparse import hstack, csr_matrix
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import f1_score

    # Load fitted transformers from Sprint 1/2 pipeline
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    hasher = joblib.load(os.path.join(MODELS_DIR, "hasher.joblib"))
    tfidf = joblib.load(os.path.join(MODELS_DIR, "tfidf.joblib"))

    with open(os.path.join(MODELS_DIR, "feature_meta.json")) as f:
        meta = json.load(f)

    # Generate fresh data — not the same seed as training
    sys.path.insert(0, MODELS_DIR)
    from preprocess import generate_synthetic_baseline
    df = generate_synthetic_baseline(
        num_samples=ABLATION_SAMPLE_SIZE, seed=99
    )

    NUMERICAL_COLS = meta["numerical_cols"]
    LABEL_COL = meta["label_col"]

    # ── Build WITHOUT TF-IDF ────────────────────────────────────────
    X_num = csr_matrix(
        scaler.transform(df[NUMERICAL_COLS].astype(float))
    )
    X_hash = hasher.transform(
        df[meta["high_card_cols"]].astype(str).to_dict(orient="records")
    )
    df_dummies = pd.get_dummies(
        df[meta["low_card_cols"]], drop_first=True
    ).reindex(columns=meta["dummy_column_order"], fill_value=0)
    bool_data = df[meta["bool_cols"]].astype(int)
    X_extra = csr_matrix(
        pd.concat([df_dummies, bool_data], axis=1).astype(float).values
    )
    X_no_tfidf = hstack([X_num, X_hash, X_extra], format="csr")

    # ── Build FULL (with TF-IDF) ────────────────────────────────────
    X_text = tfidf.transform(
        df[meta["text_col"]].fillna("").astype(str)
    )
    X_full = hstack([X_num, X_hash, X_text, X_extra], format="csr")

    # ── Encode labels and split ─────────────────────────────────────
    le = LabelEncoder()
    y = le.fit_transform(df[LABEL_COL].values)

    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=0.20, random_state=42
    )
    train_idx, test_idx = next(sss.split(X_full, y))

    # ── Train both models with identical hyperparameters ────────────
    PARAMS = dict(
        objective="multiclass",
        n_estimators=200,       # fewer rounds for CI speed
        num_leaves=63,
        min_child_samples=30,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )

    model_full = lgb.LGBMClassifier(**PARAMS)
    model_full.fit(X_full[train_idx], y[train_idx])
    f1_full = f1_score(
        y[test_idx],
        model_full.predict(X_full[test_idx]),
        average="macro",
    )

    model_no_tfidf = lgb.LGBMClassifier(**PARAMS)
    model_no_tfidf.fit(X_no_tfidf[train_idx], y[train_idx])
    f1_no_tfidf = f1_score(
        y[test_idx],
        model_no_tfidf.predict(X_no_tfidf[test_idx]),
        average="macro",
    )

    return {
        "f1_full": float(f1_full),
        "f1_no_tfidf": float(f1_no_tfidf),
        "delta": float(f1_full - f1_no_tfidf),
    }


class TestAblationGates:
    """CI gates that enforce telemetry-driven model behaviour."""

    def test_without_tfidf_f1_above_floor(
        self, ablation_results
    ) -> None:
        """
        Without TF-IDF features, macro F1 must exceed 0.55.
        Failure means numerical distributions have lost their
        discriminative signal — check generate_synthetic_baseline().
        """
        f1 = ablation_results["f1_no_tfidf"]
        assert f1 >= WITHOUT_TFIDF_F1_MIN, (
            f"Numerical feature F1 = {f1:.4f} is below the "
            f"floor of {WITHOUT_TFIDF_F1_MIN}. The model has lost "
            f"telemetry signal. Check per-class distributions in "
            f"models/preprocess.py::generate_synthetic_baseline()."
        )

    def test_full_model_f1_below_memorization_ceiling(
        self, ablation_results
    ) -> None:
        """
        Full model macro F1 must not exceed 0.90.
        Exceeding 0.90 indicates memorization — likely class-
        specific keywords re-introduced into error templates.
        """
        f1 = ablation_results["f1_full"]
        assert f1 <= FULL_MODEL_F1_MAX, (
            f"Full model F1 = {f1:.4f} exceeds memorization "
            f"ceiling of {FULL_MODEL_F1_MAX}. Check error_message "
            f"templates in generate_synthetic_baseline() for "
            f"class-specific keywords."
        )

    def test_text_contribution_delta_below_threshold(
        self, ablation_results
    ) -> None:
        """
        The F1 delta (full minus no-TF-IDF) must be below 0.30.
        A large positive delta means TF-IDF is doing most of the
        work — the model is a text classifier, not an AIOps system.
        """
        delta = ablation_results["delta"]
        assert delta < DELTA_MAX, (
            f"F1 delta = {delta:.4f} exceeds maximum of {DELTA_MAX}. "
            f"TF-IDF features are contributing too much signal. "
            f"Error messages may contain class-identifying keywords."
        )

    def test_ablation_results_are_logged(
        self, ablation_results, capsys
    ) -> None:
        """
        Print ablation results so they appear in CI logs for
        every test run. Not a pass/fail gate — always passes.
        """
        print(
            f"\nABLATION RESULTS:"
            f"\n  Full model F1      : "
            f"{ablation_results['f1_full']:.4f}"
            f"\n  Without TF-IDF F1  : "
            f"{ablation_results['f1_no_tfidf']:.4f}"
            f"\n  Delta              : "
            f"{ablation_results['delta']:+.4f}"
        )
        assert True  # always passes — logging test
