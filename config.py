"""
config.py — Sentinel-AIOps Central Configuration
==================================================
Centralized configuration block to avoid hardcoded magic numbers.
"""

import os

# ── Feature Drift & Anomalies ────────────────────────────────────────────────
PSI_SEVERE_THRESHOLD: float = float(os.getenv("PSI_SEVERE_THRESHOLD", "0.20"))
PSI_MODERATE_THRESHOLD: float = float(os.getenv("PSI_MODERATE_THRESHOLD", "0.10"))
PSI_BINS: int = int(os.getenv("PSI_BINS", "10"))

# Threshold of previous inferences before considering retrain checks
RETRAIN_THRESHOLD: int = int(os.getenv("RETRAIN_THRESHOLD", "100"))

# ── Database ──────────────────────────────────────────────────────────────────
# Default to SQLite at project level
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(ROOT_DIR, "database", "sentinel.db")
DATABASE_URI: str = os.getenv("DATABASE_URI", f"sqlite:///{DEFAULT_DB_PATH}")
