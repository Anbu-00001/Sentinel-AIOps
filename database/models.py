"""
models.py — Sentinel-AIOps Database Models
=============================================
SQLAlchemy ORM models for persistent inference storage.
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
    metrics_payload = Column(JSON, nullable=False)
    prediction = Column(String(64), nullable=False)
    confidence = Column(Float, nullable=False)
    top_features = Column(JSON, nullable=True)

    def __repr__(self) -> str:
        return (
            f"<LogEntry(id={self.id}, prediction={self.prediction!r}, "
            f"confidence={self.confidence:.4f})>"
        )

    def to_dict(self) -> dict:
        """Serialize to plain dict for API responses."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metrics_payload": self.metrics_payload,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "top_features": self.top_features,
        }
