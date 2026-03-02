"""
schema.py — Sentinel-AIOps Database Schema (Phase 10)
=======================================================
Canonical re-export of the ORM models. Satisfies the Phase 10
requirement for a dedicated `database/schema.py` module.
"""

from database.models import Base, LogEntry  # noqa: F401

__all__ = ["Base", "LogEntry"]
