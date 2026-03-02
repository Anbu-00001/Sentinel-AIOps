"""
session.py — Sentinel-AIOps Database Session Management
=========================================================
Engine creation, session factory, and initialization helpers.

For :memory: databases (tests), uses SQLAlchemy StaticPool so that
all connections share the same in-memory database instance.
"""

import logging
import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from database.models import Base

# ── Logging ───────────────────────────────────────────────────────────────────
log = logging.getLogger("sentinel.database")

# ── Database path ─────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _db_path() -> str:
    """Return current DB path (respects SENTINEL_DB_PATH env override)."""
    return os.environ.get(
        "SENTINEL_DB_PATH",
        os.path.join(ROOT, "data", "sentinel.db"),
    )


def _make_engine():
    """
    Create a SQLAlchemy engine pointed at the current DB_PATH.

    For :memory: databases, uses StaticPool so ALL connections share
    the same in-memory database — critical for test isolation.
    """
    path = _db_path()
    if path == ":memory:":
        engine = create_engine(
            "sqlite:///:memory:",
            echo=False,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        engine = create_engine(
            f"sqlite:///{path}",
            echo=False,
            connect_args={"check_same_thread": False},
        )
    return engine


# ── Module-level singletons (lazily initialized) ─────────────────────────────
_engine = None
_SessionLocal = None


def _get_engine():
    global _engine
    if _engine is None:
        _engine = _make_engine()
    return _engine


def _get_session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            bind=_get_engine(),
            autocommit=False,
            autoflush=False,
        )
    return _SessionLocal


def init_db() -> None:
    """
    Create all tables if they do not exist.

    Rebuilds the engine/session so SENTINEL_DB_PATH overrides (e.g. :memory:
    for tests) are always honoured even after the module has been imported.
    """
    global _engine, _SessionLocal
    _engine = _make_engine()
    _SessionLocal = sessionmaker(
        bind=_engine,
        autocommit=False,
        autoflush=False,
    )
    Base.metadata.create_all(bind=_engine)
    log.info("Database initialized at %s", _db_path())


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Yield a transactional session, auto-closing on exit."""
    session = _get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
