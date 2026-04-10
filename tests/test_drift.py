import numpy as np
from models.drift_monitor import compute_psi


def test_psi_zero_probability_handling():
    """
    Intentionally passes a probability of 0.0 (by having a bin with 0 counts)
    to the PSI calculator to prove it returns a valid drift score instead of crashing.
    """
    # Create reference distribution
    reference = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])

    # Create current distribution where values 4 and 5 are completely missing
    # This will ensure some bins have 0 actual counts, normally leading to log(0).
    current = np.array([1, 1, 2, 2, 3, 3])

    # Calculate PSI. Without Epsilon Smoothing, this would crash.
    psi_score = compute_psi(reference, current, n_bins=5)

    # Assert PSI score is a valid, high float number
    assert isinstance(psi_score, float)
    assert not np.isnan(psi_score)
    assert not np.isinf(psi_score)
    assert psi_score > 0.1  # Significant difference guarantees drift > 0.1
