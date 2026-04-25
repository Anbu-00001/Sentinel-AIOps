"""
test_model.py — Sentinel-AIOps Real Logic Tests
=================================================
Tests the actual inference logic in mcp_server/logic.py using real
function calls with real assertions. No mocking of project code.

Covers:
  - validate_input()    : schema validation for required keys and types
  - get_top_features()  : top-N feature extraction from sparse rows
  - transform_input()   : 4-group transformation pipeline
  - run_prediction()    : end-to-end prediction with fitted LightGBM
"""

import os

os.environ.setdefault("TESTING", "1")

import numpy as np
import pytest
from scipy.sparse import csr_matrix, issparse

from mcp_server.logic import (
    REQUIRED_KEYS,
    OPTIONAL_KEYS,
    validate_input,
    get_top_features,
    transform_input,
    run_prediction,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_sparse_row():
    """10-element sparse row with values 1..10 in known positions."""
    data = np.array([1.0, 5.0, 3.0, 9.0, 2.0,
                     7.0, 4.0, 8.0, 6.0, 10.0])
    return csr_matrix(data)


@pytest.fixture
def sample_feature_names():
    """Feature names matching the 10-element sparse row."""
    return [f"feature_{i}" for i in range(10)]


@pytest.fixture(scope="module")
def fitted_transformers():
    """Fitted sklearn transformers on minimal in-memory data."""
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_extraction import FeatureHasher
    from sklearn.feature_extraction.text import TfidfVectorizer

    meta = {
        "numerical_cols": ["build_duration_sec", "test_duration_sec",
                           "deploy_duration_sec", "cpu_usage_pct",
                           "memory_usage_mb", "retry_count"],
        "high_card_cols": ["repository", "author"],
        "low_card_cols": ["ci_tool", "language", "os",
                          "cloud_provider", "failure_stage",
                          "severity"],
        "bool_cols": ["is_flaky_test", "rollback_triggered",
                      "incident_created"],
        "text_col": "error_message",
        "hash_n_features": 8,
        "tfidf_max_features": 10,
        "dummy_column_order": [],
    }

    # Fit scaler on 20 random rows
    rng = np.random.default_rng(42)
    num_data = rng.uniform(0, 100, size=(20, 6))
    scaler = StandardScaler().fit(num_data)

    # Fit hasher (stateless, but instantiated)
    hasher = FeatureHasher(n_features=8, input_type="dict")

    # Fit tfidf on generic sentences
    texts = [
        "job failed at step one",
        "process exited with code two",
        "stage did not complete successfully",
        "pipeline runner error occurred",
        "task aborted upstream dependency",
    ] * 4
    tfidf = TfidfVectorizer(max_features=10).fit(texts)

    return scaler, hasher, tfidf, meta


@pytest.fixture
def valid_features():
    """Full valid features dict matching the Sentinel-AIOps schema."""
    return {
        "build_duration_sec": 1800.0,
        "test_duration_sec": 300.0,
        "deploy_duration_sec": 150.0,
        "cpu_usage_pct": 55.0,
        "memory_usage_mb": 8192.0,
        "retry_count": 2.0,
        "error_message": "job failed at step one",
        "repository": "org/repo",
        "author": "dev-42",
        "failure_stage": "build",
        "severity": "HIGH",
        "ci_tool": "GitHub Actions",
        "language": "Python",
        "os": "Linux",
        "cloud_provider": "AWS",
        "is_flaky_test": False,
        "rollback_triggered": False,
        "incident_created": False,
    }


@pytest.fixture(scope="module")
def fitted_lgbm(fitted_transformers):
    """Tiny LightGBM model fitted on 60 synthetic training rows."""
    import numpy as np
    import lightgbm as lgb
    from sklearn.preprocessing import LabelEncoder
    from scipy.sparse import vstack

    scaler, hasher, tfidf, meta = fitted_transformers

    classes = [
        "Build Failure", "Configuration Error", "Dependency Error",
        "Deployment Failure", "Network Error", "Permission Error",
        "Resource Exhaustion", "Security Scan Failure",
        "Test Failure", "Timeout",
    ]
    rows, labels = [], []
    rng = np.random.default_rng(99)
    base_features = {
        "build_duration_sec": 1800.0,
        "test_duration_sec": 300.0,
        "deploy_duration_sec": 150.0,
        "cpu_usage_pct": 55.0,
        "memory_usage_mb": 8192.0,
        "retry_count": 2.0,
        "error_message": "job failed at step one",
    }
    for cls in classes:
        for _ in range(6):
            f = dict(base_features)
            f["cpu_usage_pct"] += float(rng.uniform(-5, 5))
            f["memory_usage_mb"] += float(rng.uniform(-500, 500))
            rows.append(f)
            labels.append(cls)

    X = vstack([transform_input(r, scaler, hasher, tfidf, meta)
                for r in rows])

    le = LabelEncoder()
    y = le.fit_transform(labels)

    model = lgb.LGBMClassifier(
        n_estimators=50,
        num_leaves=10,
        random_state=42,
        verbose=-1,
    )
    model.fit(X, y)

    feature_names = (
        meta["numerical_cols"]
        + [f"hash_{i}" for i in range(meta["hash_n_features"])]
        + [f"tfidf_{i}" for i in range(meta["tfidf_max_features"])]
        + [str(c) for c in meta["dummy_column_order"]]
        + meta["bool_cols"]
    )
    return model, le, feature_names


# ═══════════════════════════════════════════════════════════════════════════════
# TestValidateInput
# ═══════════════════════════════════════════════════════════════════════════════


class TestValidateInput:
    """Tests for validate_input() — schema validation on feature dicts."""

    def test_valid_input_returns_empty_list(self):
        """A fully valid features dict produces zero validation errors."""
        features = {
            "build_duration_sec": 1800,
            "test_duration_sec": 300,
            "deploy_duration_sec": 150,
            "cpu_usage_pct": 55.0,
            "memory_usage_mb": 8192,
            "retry_count": 2,
            "error_message": "job failed at step one",
        }
        assert validate_input(features) == []

    @pytest.mark.parametrize("key", REQUIRED_KEYS)
    def test_missing_single_required_key(self, key):
        """Removing one required key produces exactly one error mentioning it."""
        features = {
            "build_duration_sec": 1800,
            "test_duration_sec": 300,
            "deploy_duration_sec": 150,
            "cpu_usage_pct": 55.0,
            "memory_usage_mb": 8192,
            "retry_count": 2,
            "error_message": "job failed at step one",
        }
        del features[key]
        errors = validate_input(features)
        assert len(errors) == 1
        assert key in errors[0]

    def test_missing_multiple_required_keys(self):
        """An empty dict produces one error per required key."""
        errors = validate_input({})
        assert len(errors) == len(REQUIRED_KEYS) == 7

    def test_wrong_type_numeric_field_string(self):
        """A non-numeric string for memory_usage_mb triggers exactly one error."""
        features = {
            "build_duration_sec": 1800,
            "test_duration_sec": 300,
            "deploy_duration_sec": 150,
            "cpu_usage_pct": 55.0,
            "memory_usage_mb": "not_a_number",
            "retry_count": 2,
            "error_message": "job failed",
        }
        errors = validate_input(features)
        assert len(errors) == 1
        assert "memory_usage_mb" in errors[0]

    def test_wrong_type_numeric_field_none(self):
        """None for cpu_usage_pct triggers exactly one error."""
        features = {
            "build_duration_sec": 1800,
            "test_duration_sec": 300,
            "deploy_duration_sec": 150,
            "cpu_usage_pct": None,
            "memory_usage_mb": 8192,
            "retry_count": 2,
            "error_message": "job failed",
        }
        errors = validate_input(features)
        assert len(errors) == 1

    def test_wrong_type_numeric_field_list(self):
        """A list for retry_count triggers exactly one error."""
        features = {
            "build_duration_sec": 1800,
            "test_duration_sec": 300,
            "deploy_duration_sec": 150,
            "cpu_usage_pct": 55.0,
            "memory_usage_mb": 8192,
            "retry_count": [1, 2, 3],
            "error_message": "job failed",
        }
        errors = validate_input(features)
        assert len(errors) == 1

    def test_numeric_field_as_string_int_passes(self):
        """String '3' for retry_count passes because float('3') succeeds."""
        features = {
            "build_duration_sec": 1800,
            "test_duration_sec": 300,
            "deploy_duration_sec": 150,
            "cpu_usage_pct": 55.0,
            "memory_usage_mb": 8192,
            "retry_count": "3",
            "error_message": "job failed",
        }
        errors = validate_input(features)
        assert len(errors) == 0

    def test_zero_values_are_valid(self):
        """All numeric fields as 0 or 0.0 are valid."""
        features = {
            "build_duration_sec": 0,
            "test_duration_sec": 0,
            "deploy_duration_sec": 0,
            "cpu_usage_pct": 0.0,
            "memory_usage_mb": 0,
            "retry_count": 0,
            "error_message": "job failed",
        }
        assert validate_input(features) == []

    def test_negative_values_are_valid(self):
        """Negative numeric values pass validation (type check only)."""
        features = {
            "build_duration_sec": 1800,
            "test_duration_sec": 300,
            "deploy_duration_sec": 150,
            "cpu_usage_pct": -5.0,
            "memory_usage_mb": 8192,
            "retry_count": -1,
            "error_message": "job failed",
        }
        assert validate_input(features) == []


# ═══════════════════════════════════════════════════════════════════════════════
# TestGetTopFeatures
# ═══════════════════════════════════════════════════════════════════════════════


class TestGetTopFeatures:
    """Tests for get_top_features() — top-N feature extraction."""

    def test_returns_correct_count(self, sample_sparse_row, sample_feature_names):
        """get_top_features returns exactly N items when N=5."""
        result = get_top_features(sample_sparse_row, sample_feature_names, n=5)
        assert len(result) == 5

    def test_returns_correct_count_n3(self, sample_sparse_row, sample_feature_names):
        """get_top_features returns exactly 3 items when N=3."""
        result = get_top_features(sample_sparse_row, sample_feature_names, n=3)
        assert len(result) == 3

    def test_top_feature_is_highest_value(self, sample_sparse_row, sample_feature_names):
        """The first result is feature_9 with value 10.0 (highest)."""
        result = get_top_features(sample_sparse_row, sample_feature_names, n=5)
        assert result[0]["feature"] == "feature_9"
        assert result[0]["value"] == pytest.approx(10.0, abs=1e-3)

    def test_second_feature_is_second_highest(self, sample_sparse_row, sample_feature_names):
        """The second result is feature_3 with value 9.0."""
        result = get_top_features(sample_sparse_row, sample_feature_names, n=5)
        assert result[1]["feature"] == "feature_3"

    def test_values_are_rounded_to_4_decimal_places(self):
        """Returned values are rounded to 4 decimal places."""
        data = np.array([1.0 / 3.0])
        row = csr_matrix(data)
        names = ["f0"]
        result = get_top_features(row, names, n=1)
        assert result[0]["value"] == pytest.approx(0.3333, abs=1e-3)

    def test_result_is_sorted_descending(self, sample_sparse_row, sample_feature_names):
        """All consecutive pairs are in non-increasing order."""
        result = get_top_features(sample_sparse_row, sample_feature_names, n=5)
        for i in range(len(result) - 1):
            assert result[i]["value"] >= result[i + 1]["value"]

    def test_handles_dense_array_input(self, sample_feature_names):
        """A plain numpy array (not sparse) works without exception."""
        arr = np.array([[1.0, 5.0, 3.0, 9.0, 2.0,
                         7.0, 4.0, 8.0, 6.0, 10.0]])
        result = get_top_features(arr, sample_feature_names, n=3)
        assert len(result) == 3

    def test_feature_name_fallback_when_index_out_of_range(self):
        """Out-of-range feature indices use f_{idx} fallback name."""
        data = np.array([1.0, 5.0, 3.0, 9.0, 2.0])
        row = csr_matrix(data)
        short_names = ["a", "b"]  # Only 2 names for 5-element row
        result = get_top_features(row, short_names, n=5)
        fallback_names = [r["feature"] for r in result if r["feature"].startswith("f_")]
        assert len(fallback_names) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# TestTransformInput
# ═══════════════════════════════════════════════════════════════════════════════


class TestTransformInput:
    """Tests for transform_input() — 4-group feature transformation pipeline."""

    def test_transform_returns_sparse_matrix(self, valid_features, fitted_transformers):
        """transform_input returns a scipy sparse matrix."""
        scaler, hasher, tfidf, meta = fitted_transformers
        result = transform_input(valid_features, scaler, hasher, tfidf, meta)
        assert issparse(result)

    def test_transform_output_shape_columns(self, valid_features, fitted_transformers):
        """Output has 6+8+10+0+3=27 columns."""
        scaler, hasher, tfidf, meta = fitted_transformers
        result = transform_input(valid_features, scaler, hasher, tfidf, meta)
        assert result.shape == (1, 27)

    def test_transform_output_shape_rows(self, valid_features, fitted_transformers):
        """Output has exactly 1 row."""
        scaler, hasher, tfidf, meta = fitted_transformers
        result = transform_input(valid_features, scaler, hasher, tfidf, meta)
        assert result.shape[0] == 1

    def test_transform_handles_missing_optional_keys(self, fitted_transformers):
        """Only the 7 required keys — optional keys are filled with defaults."""
        scaler, hasher, tfidf, meta = fitted_transformers
        features = {
            "build_duration_sec": 1800.0,
            "test_duration_sec": 300.0,
            "deploy_duration_sec": 150.0,
            "cpu_usage_pct": 55.0,
            "memory_usage_mb": 8192.0,
            "retry_count": 2.0,
            "error_message": "job failed at step one",
        }
        result = transform_input(features, scaler, hasher, tfidf, meta)
        assert result.shape[0] == 1

    def test_transform_bool_fields_are_zero_or_one(self, fitted_transformers):
        """Boolean columns (last 3) contain only 0.0 or 1.0."""
        scaler, hasher, tfidf, meta = fitted_transformers
        features = {
            "build_duration_sec": 1800.0,
            "test_duration_sec": 300.0,
            "deploy_duration_sec": 150.0,
            "cpu_usage_pct": 55.0,
            "memory_usage_mb": 8192.0,
            "retry_count": 2.0,
            "error_message": "job failed at step one",
            "is_flaky_test": True,
            "rollback_triggered": False,
            "incident_created": True,
        }
        result = transform_input(features, scaler, hasher, tfidf, meta)
        dense = result.toarray()
        bool_vals = dense[0, -3:]
        for v in bool_vals:
            assert v in {0.0, 1.0}

    def test_transform_is_deterministic(self, valid_features, fitted_transformers):
        """Two identical calls produce element-wise equal outputs."""
        scaler, hasher, tfidf, meta = fitted_transformers
        r1 = transform_input(valid_features, scaler, hasher, tfidf, meta)
        r2 = transform_input(valid_features, scaler, hasher, tfidf, meta)
        assert np.allclose(r1.toarray(), r2.toarray())


# ═══════════════════════════════════════════════════════════════════════════════
# TestRunPrediction
# ═══════════════════════════════════════════════════════════════════════════════


class TestRunPrediction:
    """Tests for run_prediction() — end-to-end inference with fitted model."""

    KNOWN_CLASSES = [
        "Build Failure", "Configuration Error", "Dependency Error",
        "Deployment Failure", "Network Error", "Permission Error",
        "Resource Exhaustion", "Security Scan Failure",
        "Test Failure", "Timeout",
    ]

    def _predict(self, valid_features, fitted_transformers, fitted_lgbm):
        scaler, hasher, tfidf, meta = fitted_transformers
        model, le, feature_names = fitted_lgbm
        return run_prediction(
            valid_features, model, le, scaler, hasher, tfidf, meta, feature_names
        )

    def test_run_prediction_returns_dict(self, valid_features, fitted_transformers, fitted_lgbm):
        """run_prediction returns a dict."""
        result = self._predict(valid_features, fitted_transformers, fitted_lgbm)
        assert isinstance(result, dict)

    def test_run_prediction_has_required_keys(self, valid_features, fitted_transformers, fitted_lgbm):
        """Result contains prediction, confidence, top_features, transformed_matrix."""
        result = self._predict(valid_features, fitted_transformers, fitted_lgbm)
        assert {"prediction", "confidence", "top_features", "transformed_matrix"} <= result.keys()

    def test_prediction_is_string(self, valid_features, fitted_transformers, fitted_lgbm):
        """prediction value is a string."""
        result = self._predict(valid_features, fitted_transformers, fitted_lgbm)
        assert isinstance(result["prediction"], str)

    def test_prediction_is_known_class(self, valid_features, fitted_transformers, fitted_lgbm):
        """prediction is one of the 10 known class names."""
        result = self._predict(valid_features, fitted_transformers, fitted_lgbm)
        assert result["prediction"] in self.KNOWN_CLASSES

    def test_confidence_is_float_between_0_and_1(self, valid_features, fitted_transformers, fitted_lgbm):
        """confidence is a float in [0.0, 1.0]."""
        result = self._predict(valid_features, fitted_transformers, fitted_lgbm)
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_top_features_is_list_of_dicts(self, valid_features, fitted_transformers, fitted_lgbm):
        """top_features is a list of 5 dicts with 'feature' and 'value' keys."""
        result = self._predict(valid_features, fitted_transformers, fitted_lgbm)
        assert isinstance(result["top_features"], list)
        assert len(result["top_features"]) == 5
        for item in result["top_features"]:
            assert "feature" in item and "value" in item

    def test_top_feature_values_are_non_negative(self, valid_features, fitted_transformers, fitted_lgbm):
        """All top_features values are >= 0 (absolute values)."""
        result = self._predict(valid_features, fitted_transformers, fitted_lgbm)
        for item in result["top_features"]:
            assert item["value"] >= 0

    def test_transformed_matrix_is_sparse(self, valid_features, fitted_transformers, fitted_lgbm):
        """transformed_matrix in result is a scipy sparse matrix."""
        result = self._predict(valid_features, fitted_transformers, fitted_lgbm)
        assert issparse(result["transformed_matrix"])

    def test_run_prediction_is_deterministic(self, valid_features, fitted_transformers, fitted_lgbm):
        """Two identical calls produce identical prediction and confidence."""
        r1 = self._predict(valid_features, fitted_transformers, fitted_lgbm)
        r2 = self._predict(valid_features, fitted_transformers, fitted_lgbm)
        assert r1["prediction"] == r2["prediction"]
        assert r1["confidence"] == r2["confidence"]

    def test_run_prediction_with_all_optional_defaults(self, fitted_transformers, fitted_lgbm):
        """Only the 7 required keys — run_prediction fills defaults and succeeds."""
        scaler, hasher, tfidf, meta = fitted_transformers
        model, le, feature_names = fitted_lgbm
        features = {
            "build_duration_sec": 1800.0,
            "test_duration_sec": 300.0,
            "deploy_duration_sec": 150.0,
            "cpu_usage_pct": 55.0,
            "memory_usage_mb": 8192.0,
            "retry_count": 2.0,
            "error_message": "job failed at step one",
        }
        result = run_prediction(
            features, model, le, scaler, hasher, tfidf, meta, feature_names
        )
        assert "prediction" in result
