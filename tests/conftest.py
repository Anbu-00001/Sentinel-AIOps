"""
conftest.py — Sentinel-AIOps Pytest Global Configuration
==========================================================
Ensures a single shared in-memory database is used across all
test files, with the Phase 10 schema fully created before any
test runs. This prevents per-file engine re-creation issues.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Set the DB path BEFORE any test module imports database.session
os.environ.setdefault("SENTINEL_DB_PATH", ":memory:")

import database.session as _session_mod  # noqa: E402

# Force a fresh engine pointing at :memory: with all Phase 10 columns.
# This runs exactly once at pytest collection time.
_session_mod._engine = None
_session_mod._SessionLocal = None

from database import init_db  # noqa: E402
init_db()
