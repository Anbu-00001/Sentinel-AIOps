"""
session.py — Sentinel-AIOps Database Session Management
=========================================================
Engine creation with Turso (persistent cloud SQLite) support
via libsql-experimental SDK, and local SQLite fallback for dev/CI.

Turso mode:  TURSO_DATABASE_URL + TURSO_AUTH_TOKEN set → libsql embedded replica
Local mode:  No Turso credentials → plain SQLite with WAL
Test mode:   SENTINEL_DB_PATH=:memory: → in-memory with StaticPool
"""

import os
import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from database.models import Base

log = logging.getLogger("sentinel.database")


def _build_engine():
    """
    Create a SQLAlchemy engine.

    Reads TURSO_DATABASE_URL and TURSO_AUTH_TOKEN directly from os.environ
    (NOT from config module) to guarantee we see the latest env at call time.

    Turso path uses libsql-experimental SDK with a custom creator function,
    bypassing the sqlalchemy-libsql dialect entirely (which fails to register
    its entry point in some environments like HF Spaces Python 3.12-slim).
    """
    turso_url = os.environ.get("TURSO_DATABASE_URL", "").strip()
    turso_token = os.environ.get("TURSO_AUTH_TOKEN", "").strip()
    db_path = os.environ.get("SENTINEL_DB_PATH", "/app/data/sentinel.db")

    log.info(
        "Reasoning: DB engine init — TURSO credentials present: %s",
        "YES" if turso_url and turso_token else "NO"
    )

    if turso_url and turso_token:
        # Production: Turso via libsql-experimental
        # Uses a custom SQLAlchemy creator that connects via libsql
        log.info("Reasoning: Connecting to Turso via libsql-experimental.")
        try:
            import libsql_experimental as libsql

            def _creator():
                return libsql.connect(
                    database=db_path,
                    sync_url=turso_url,
                    auth_token=turso_token,
                )

            engine = create_engine(
                "sqlite+pysqlite://",
                creator=_creator,
                connect_args={},
            )

            # Trigger a sync on connect
            @event.listens_for(engine, "connect")
            def do_sync(dbapi_conn, _):
                dbapi_conn.sync()

            log.info("Database connected to Turso (persistent).")
        except Exception as e:
            log.error(
                "Turso connection failed: %s — falling back to local SQLite", e
            )
            turso_url = ""  # trigger fallback

    if not turso_url or not turso_token:
        # Local dev / CI: plain SQLite with WAL mode
        if db_path == ":memory:":
            engine = create_engine(
                "sqlite:///:memory:",
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
            )
        else:
            try:
                os.makedirs(os.path.dirname(db_path), exist_ok=True)
            except (OSError, PermissionError):
                pass
            engine = create_engine(
                f"sqlite:///{db_path}",
                connect_args={"check_same_thread": False},
            )

            @event.listens_for(engine, "connect")
            def set_wal(dbapi_conn, _):
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
    db_path = os.environ.get("SENTINEL_DB_PATH", "/app/data/sentinel.db")
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
