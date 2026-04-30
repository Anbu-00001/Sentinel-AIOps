"""
conftest.py — Sentinel-AIOps Test Configuration
=================================================
Sets up environment and fixtures for the test suite.

IMPORTANT: os.environ["TESTING"] must be set BEFORE any project imports
so that models/crypto_sig.py uses the test-only fallback secret.
"""

import os

# ── Environment setup (must precede ALL project imports) ─────────────────────
os.environ["TESTING"] = "1"
os.environ["SENTINEL_DB_PATH"] = ":memory:"

# ── Clear secrets that may leak from the user's shell environment ────────────
# Individual tests that need these secrets set them explicitly and clean up.
os.environ.pop("GITHUB_WEBHOOK_SECRET", None)
os.environ.pop("SENTINEL_API_KEY", None)

import pytest  # noqa: E402
import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402
from datetime import datetime, timedelta  # noqa: E402


@pytest.fixture
def synthetic_dataframe():
    """
    Generates a deterministic 100-row DataFrame matching the Sentinel-AIOps schema.
    Used for unit testing without physical file dependencies.
    """
    num_samples = 100
    np.random.seed(42)  # Deterministic for tests

    ci_tools = ["Jenkins", "GitHub Actions", "GitLab CI", "CircleCI", "Travis CI"]
    languages = ["Python", "Java", "JavaScript", "Go", "Rust", "C++"]
    os_list = ["Linux", "Windows", "macOS"]
    cloud_providers = ["AWS", "GCP", "Azure", "On-Premise"]
    stages = ["build", "test", "deploy"]
    severities = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    failure_types = [
        "Build Failure",
        "Configuration Error",
        "Dependency Error",
        "Deployment Failure",
        "Network Error",
        "Permission Error",
        "Resource Exhaustion",
        "Security Scan Failure",
        "Test Failure",
        "Timeout",
    ]

    data = {
        "pipeline_id": [f"pipe-{i % 10}" for i in range(num_samples)],
        "run_id": [f"run-{i}" for i in range(num_samples)],
        "timestamp": [
            (datetime(2026, 1, 1) + timedelta(minutes=i)).isoformat()
            for i in range(num_samples)
        ],
        "ci_tool": np.random.choice(ci_tools, num_samples),
        "repository": [f"repo-{i % 5}" for i in range(num_samples)],
        "branch": np.random.choice(
            ["main", "develop", "feature", "hotfix"], num_samples
        ),
        "commit_hash": [f"hash-{i}" for i in range(num_samples)],
        "author": [f"dev-{i % 10}" for i in range(num_samples)],
        "language": np.random.choice(languages, num_samples),
        "os": np.random.choice(os_list, num_samples),
        "cloud_provider": np.random.choice(cloud_providers, num_samples),
        # Mathematical Distributions
        "build_duration_sec": np.random.normal(1800, 500, num_samples)
        .clip(10, 3600)
        .astype(int),
        "test_duration_sec": np.random.normal(300, 100, num_samples)
        .clip(5, 600)
        .astype(int),
        "deploy_duration_sec": np.random.normal(150, 50, num_samples)
        .clip(5, 300)
        .astype(int),
        "cpu_usage_pct": np.random.normal(50, 15, num_samples).clip(0, 100),
        "memory_usage_mb": np.random.normal(8192, 2048, num_samples)
        .clip(256, 16384)
        .astype(int),
        "retry_count": np.random.randint(0, 6, num_samples),
        "failure_stage": np.random.choice(stages, num_samples),
        "failure_type": np.random.choice(failure_types, num_samples),
        "error_code": [f"ERR_{i}" for i in range(num_samples)],
        "error_message": ["Test log entry" for _ in range(num_samples)],
        "severity": np.random.choice(severities, num_samples),
        "is_flaky_test": np.random.choice([True, False], num_samples),
        "rollback_triggered": np.random.choice([True, False], num_samples),
        "incident_created": np.random.choice([True, False], num_samples),
    }
    return pd.DataFrame(data)


from unittest.mock import patch, MagicMock  # noqa: E402


@pytest.fixture(autouse=True)
def mock_secure_load():
    with patch("dashboard.app._secure_load", return_value=MagicMock()):
        yield
