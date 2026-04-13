"""
models.py — Sentinel-AIOps Database Models
=============================================
SQLAlchemy ORM models for persistent inference storage.

Phase 10 schema:
  - event_source:   origin tag ("mcp" | "github_webhook")
  - raw_payload:    original unmodified input for audit
  - psi_drift_stat: optional per-row drift statistic
"""

from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    JSON,
    String,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""

    pass


class LogEntry(Base):
    """Persisted inference result from analyze_log."""

    __tablename__ = "log_entries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        index=True,
        nullable=False,
    )
    # ── Phase 9 columns ─────────────────────────────────────────────────
    metrics_payload = Column(JSON, nullable=False)
    prediction = Column(String(64), nullable=False)
    confidence = Column(Float, nullable=False)
    top_features = Column(JSON, nullable=True)
    # ── Phase 10 columns ─────────────────────────────────────────────────
    event_source = Column(String(64), nullable=False, default="mcp", index=True)
    raw_payload = Column(JSON, nullable=True)
    psi_drift_stat = Column(Float, nullable=True)

    def __repr__(self) -> str:
        return (
            f"<LogEntry(id={self.id}, prediction={self.prediction!r}, "
            f"confidence={self.confidence:.4f}, source={self.event_source!r})>"
        )

    def to_dict(self) -> dict:
        """Serialize to plain dict for API responses."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "event_source": self.event_source,
            "metrics_payload": self.metrics_payload,
            "raw_payload": self.raw_payload,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "confidence_score": self.confidence,  # Phase 10 alias
            "top_features": self.top_features,
            "psi_drift_stat": self.psi_drift_stat,
        }
