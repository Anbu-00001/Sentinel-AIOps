"""
session.py — Sentinel-AIOps Database Session Management
=========================================================
Engine creation, session factory, and initialization helpers.
"""

import logging
import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from database.models import Base

# ── Logging ───────────────────────────────────────────────────────────────────
log = logging.getLogger("sentinel.database")

# ── Database path ─────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.environ.get(
    "SENTINEL_DB_PATH",
    os.path.join(ROOT, "data", "sentinel.db"),
)

# ── Engine & session factory ──────────────────────────────────────────────────
engine = create_engine(
    f"sqlite:///{DB_PATH}",
    echo=False,
    connect_args={"check_same_thread": False},
)
SessionLocal: sessionmaker[Session] = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)


def init_db() -> None:
    """Create all tables if they do not exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    Base.metadata.create_all(bind=engine)
    log.info("Database initialized at %s", DB_PATH)


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
