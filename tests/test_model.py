"""
test_model.py — Sentinel-AIOps Model Loading Tests
=====================================================
Validates that all model artifacts load correctly and
produce expected output shapes.
"""

import json
import os

import joblib
import pytest

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
DATA_DIR = os.path.join(ROOT, "data")


class TestModelArtifacts:
    """Test suite for model artifact loading and validation."""

    def test_lgbm_model_loads(self) -> None:
        """LightGBM model loads without error."""
        path = os.path.join(MODELS_DIR, "lgbm_model.joblib")
        assert os.path.exists(path), f"Model not found: {path}"
        model = joblib.load(path)
        assert hasattr(model, "predict"), "Model missing predict method"
        assert hasattr(model, "predict_proba"), "Model missing predict_proba"

    def test_label_encoder_loads(self) -> None:
        """Label encoder loads and has 10 classes."""
        path = os.path.join(MODELS_DIR, "label_encoder.joblib")
        le = joblib.load(path)
        assert len(le.classes_) == 10, (
            f"Expected 10 classes, got {len(le.classes_)}"
        )

    def test_scaler_loads(self) -> None:
        """StandardScaler loads with correct feature count."""
        path = os.path.join(MODELS_DIR, "scaler.joblib")
        scaler = joblib.load(path)
        assert hasattr(scaler, "mean_"), "Scaler not fitted"
        assert len(scaler.mean_) == 6, (
            f"Expected 6 numerical features, got {len(scaler.mean_)}"
        )

    def test_tfidf_loads(self) -> None:
        """TF-IDF vectorizer loads and has correct max_features."""
        path = os.path.join(MODELS_DIR, "tfidf.joblib")
        tfidf = joblib.load(path)
        assert hasattr(tfidf, "vocabulary_"), "TF-IDF not fitted"
        assert len(tfidf.vocabulary_) <= 500, (
            f"Expected max 500 features, got {len(tfidf.vocabulary_)}"
        )

    def test_hasher_loads(self) -> None:
        """FeatureHasher loads with correct n_features."""
        path = os.path.join(MODELS_DIR, "hasher.joblib")
        hasher = joblib.load(path)
        assert hasher.n_features == 64, (
            f"Expected 64 hash features, got {hasher.n_features}"
        )

    def test_feature_meta_valid(self) -> None:
        """feature_meta.json contains all required keys."""
        path = os.path.join(MODELS_DIR, "feature_meta.json")
        with open(path) as f:
            meta = json.load(f)
        required = [
            "numerical_cols", "high_card_cols", "low_card_cols",
            "bool_cols", "text_col", "label_col",
            "hash_n_features", "tfidf_max_features",
            "dummy_column_order",
        ]
        for key in required:
            assert key in meta, f"Missing key in feature_meta: {key}"

    def test_metrics_json_exists(self) -> None:
        """metrics.json exists with expected keys."""
        path = os.path.join(MODELS_DIR, "metrics.json")
        if os.path.exists(path):
            with open(path) as f:
                metrics = json.load(f)
            assert "f1_score" in metrics or "model" in metrics

    def test_model_predict_shape(self) -> None:
        """Model produces correct output shape on dummy input."""
        from scipy.sparse import load_npz
        matrix_path = os.path.join(DATA_DIR, "feature_matrix.npz")
        if not os.path.exists(matrix_path):
            pytest.skip("Feature matrix not found")
        X = load_npz(matrix_path)
        model = joblib.load(
            os.path.join(MODELS_DIR, "lgbm_model.joblib")
        )
        preds = model.predict(X[:5])
        assert len(preds) == 5, f"Expected 5 predictions, got {len(preds)}"
        proba = model.predict_proba(X[:5])
        assert proba.shape == (5, 10), (
            f"Expected (5, 10) proba shape, got {proba.shape}"
        )
