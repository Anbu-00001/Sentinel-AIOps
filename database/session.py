"""
session.py — Sentinel-AIOps Database Session Management
=========================================================
Engine creation with Turso (persistent cloud SQLite) support
and local SQLite fallback for dev/CI.

Turso mode:  TURSO_DATABASE_URL + TURSO_AUTH_TOKEN set → embedded replica
Local mode:  No Turso credentials → plain SQLite with WAL
Test mode:   SENTINEL_DB_PATH=:memory: → in-memory with StaticPool
"""

import os
import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from database.models import Base

log = logging.getLogger("sentinel.database")


def _build_engine():
    """
    Create a SQLAlchemy engine.

    Reads TURSO_DATABASE_URL and TURSO_AUTH_TOKEN directly from os.getenv()
    (NOT from config module) to guarantee we see the latest env at call time.
    """
    # ── Diagnostic: log whether Turso env var is visible ──────────────
    turso_url_raw = os.environ.get("TURSO_DATABASE_URL", "NOT_SET")
    log.info(
        "Reasoning: DB engine init — TURSO_DATABASE_URL present: %s",
        "YES" if turso_url_raw and turso_url_raw != "NOT_SET" else "NO"
    )

    turso_url = os.getenv("TURSO_DATABASE_URL", "").strip()
    turso_token = os.getenv("TURSO_AUTH_TOKEN", "").strip()
    db_path = os.getenv("SENTINEL_DB_PATH", "/app/data/sentinel.db")

    # Ensure parent directory exists (safe no-op for :memory:)
    try:
        if db_path != ":memory:":
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
    except (OSError, PermissionError):
        pass

    if turso_url and turso_token:
        # Production: Turso embedded replica mode
        # Local file acts as replica; syncs to Turso on every write
        log.info(
            "Reasoning: TURSO_DATABASE_URL detected. "
            "Connecting via sqlalchemy-libsql embedded replica."
        )
        engine = create_engine(
            f"sqlite+libsql:///{db_path}",
            connect_args={
                "auth_token": turso_token,
                "sync_url": turso_url,
            },
        )
        log.info("Database connected to Turso (persistent).")
    else:
        # Local dev / CI: plain SQLite with WAL mode
        log.info(
            "Reasoning: No TURSO credentials found. "
            "Falling back to local SQLite at %s", db_path
        )
        kwargs = {"connect_args": {"check_same_thread": False}}
        if db_path == ":memory:":
            kwargs["poolclass"] = StaticPool

        engine = create_engine(
            f"sqlite:///{db_path}",
            **kwargs
        )
        # Enable WAL mode for local SQLite only
        # (Turso manages its own WAL — sending PRAGMA causes errors)
        @event.listens_for(engine, "connect")
        def set_wal_mode(dbapi_conn, _):
            dbapi_conn.execute("PRAGMA journal_mode=WAL")
            dbapi_conn.execute("PRAGMA synchronous=NORMAL")

        log.info("Database initialized at %s", db_path)

    return engine


# ── Module-level singletons ──────────────────────────────────────────────────
engine = _build_engine()
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def init_db() -> None:
    """
    Create all tables if they do not exist.

    Rebuilds the engine/session so SENTINEL_DB_PATH overrides (e.g. :memory:
    for tests) are always honoured even after the module has been imported.
    """
    global engine, SessionLocal
    engine = _build_engine()
    SessionLocal = sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
    )
    Base.metadata.create_all(bind=engine)
    db_path = os.getenv("SENTINEL_DB_PATH", "/app/data/sentinel.db")
    log.info("Database initialized at %s", db_path)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Yield a transactional session, auto-closing on exit."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
