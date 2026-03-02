"""
test_database.py — Sentinel-AIOps Database Layer Tests
========================================================
Validates SQLAlchemy models, session management, and CRUD
operations against an in-memory SQLite database.
"""

import os
import sys

import pytest

# ── Allow imports from project root ──────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from database.models import Base, LogEntry


@pytest.fixture()
def db_session():
    """Provide a transactional in-memory SQLite session for tests."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    engine.dispose()


class TestDatabaseSchema:
    """Tests for database schema initialization."""

    def test_init_db_creates_tables(self) -> None:
        """Calling create_all produces the log_entries table."""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        assert "log_entries" in tables, (
            f"Expected 'log_entries' table, found: {tables}"
        )
        engine.dispose()

    def test_log_entries_columns(self) -> None:
        """log_entries table has the expected columns."""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)
        inspector = inspect(engine)
        columns = {c["name"] for c in inspector.get_columns("log_entries")}
        expected = {"id", "timestamp", "metrics_payload", "prediction", "confidence", "top_features"}
        assert expected.issubset(columns), (
            f"Missing columns: {expected - columns}"
        )
        engine.dispose()


class TestLogEntryCRUD:
    """Tests for LogEntry CRUD operations."""

    def test_insert_and_query(self, db_session) -> None:
        """Insert a LogEntry and query it back."""
        entry = LogEntry(
            metrics_payload={"cpu_usage_pct": 75.0, "error_message": "test"},
            prediction="Build Failure",
            confidence=0.8765,
            top_features=[{"feature": "cpu_usage_pct", "value": 0.95}],
        )
        db_session.add(entry)
        db_session.commit()

        result = db_session.query(LogEntry).first()
        assert result is not None
        assert result.prediction == "Build Failure"
        assert abs(result.confidence - 0.8765) < 1e-4
        assert result.metrics_payload["cpu_usage_pct"] == 75.0

    def test_timestamp_auto_populates(self, db_session) -> None:
        """Timestamp defaults to current UTC time."""
        entry = LogEntry(
            metrics_payload={"test": True},
            prediction="Timeout",
            confidence=0.5,
        )
        db_session.add(entry)
        db_session.commit()

        result = db_session.query(LogEntry).first()
        assert result.timestamp is not None

    def test_to_dict_serialization(self, db_session) -> None:
        """to_dict() returns the expected keys."""
        entry = LogEntry(
            metrics_payload={"key": "value"},
            prediction="Test Failure",
            confidence=0.42,
            top_features=[],
        )
        db_session.add(entry)
        db_session.commit()

        result = db_session.query(LogEntry).first()
        d = result.to_dict()
        assert set(d.keys()) == {
            "id", "timestamp", "metrics_payload",
            "prediction", "confidence", "confidence_score",
            "top_features", "event_source", "raw_payload",
            "psi_drift_stat",
        }
        assert d["prediction"] == "Test Failure"

    def test_multiple_inserts(self, db_session) -> None:
        """Multiple entries can be inserted and counted."""
        for i in range(5):
            db_session.add(LogEntry(
                metrics_payload={"i": i},
                prediction=f"type_{i}",
                confidence=0.1 * (i + 1),
            ))
        db_session.commit()

        count = db_session.query(LogEntry).count()
        assert count == 5

    def test_order_by_timestamp(self, db_session) -> None:
        """Entries can be ordered by timestamp descending."""
        from datetime import datetime, timezone, timedelta

        for i in range(3):
            entry = LogEntry(
                metrics_payload={"i": i},
                prediction=f"type_{i}",
                confidence=0.5,
            )
            entry.timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i)
            db_session.add(entry)
        db_session.commit()

        results = (
            db_session.query(LogEntry)
            .order_by(LogEntry.timestamp.desc())
            .all()
        )
        assert results[0].prediction == "type_2"
        assert results[-1].prediction == "type_0"
