from unittest.mock import patch
from models.preprocess import load_raw, build_feature_matrix


def test_load_raw_mocked(synthetic_dataframe):
    """
    Test that load_raw correctly handles mocked data and bypasses the physical file.
    """
    with patch("pandas.read_csv") as mock_read:
        mock_read.return_value = synthetic_dataframe

        # This path doesn't need to exist because we are mocking read_csv
        df = load_raw("dummy_path.csv")

        assert len(df) == 100
        assert "cpu_usage_pct" in df.columns
        mock_read.assert_called_once_with("dummy_path.csv")


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
