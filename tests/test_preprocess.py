from unittest.mock import patch
from models.preprocess import load_raw, build_feature_matrix


def test_build_feature_matrix(synthetic_dataframe):
    """
    Test that building the feature matrix from synthetic data produces correct shapes.
    """
    X, scaler, hasher, tfidf = build_feature_matrix(synthetic_dataframe, fit=True)

    # X shape: (rows, features)
    # rows should be 100
    assert X.shape[0] == 100
    # No obvious check for features without digging into preprocessor internals,
    # but it should be greater than 0.
    assert X.shape[1] > 0

    assert scaler is not None
    assert hasher is not None
    assert tfidf is not None
