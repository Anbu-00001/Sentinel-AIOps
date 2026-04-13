"""
feedback_handler.py — Sentinel-AIOps Human-in-the-Loop Feedback
=================================================================
Enterprise-grade MCP tool for submitting human corrections.
Features:
  - Pydantic-validated input schema
  - Thread-safe JSON file persistence
  - Auto-triggers retrain_required=True in registry.json at >100 corrections

Workflow Rules (AGENTS.md):
  - Log 'Reasoning' before execution.
  - Type hinting + Pydantic validation throughout.
"""

import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field, field_validator

# ── Logging ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("sentinel.feedback_handler")

# ── Paths ───────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEEDBACK_DIR = os.path.join(ROOT, "data", "feedback")
FEEDBACK_FILE = os.path.join(FEEDBACK_DIR, "labels.json")
REGISTRY_FILE = os.path.join(ROOT, "models", "registry.json")

# ── Thread lock ─────────────────────────────────────────────────────────
_write_lock = threading.Lock()

# ── Retrain threshold ───────────────────────────────────────────────────
RETRAIN_THRESHOLD: int = 100

# ── Valid labels ────────────────────────────────────────────────────────
VALID_LABELS: set[str] = {
    "Build Failure", "Configuration Error", "Dependency Error",
    "Deployment Failure", "Network Error", "Permission Error",
    "Resource Exhaustion", "Security Scan Failure",
    "Test Failure", "Timeout",
}


# ── Pydantic Schemas ─────────────────────────────────────────────────────────

class CorrectionInput(BaseModel):
    """Input schema for human corrections."""
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    log_id: str = Field(..., min_length=1)
    original_pred: str = Field(...)
    corrected_label: str = Field(...)
    author_id: str = Field(..., min_length=1)

    @field_validator("corrected_label")
    @classmethod
    def validate_label(cls, v: str) -> str:
        if v not in VALID_LABELS:
            raise ValueError(
                f"Invalid label '{v}'. "
                f"Must be one of: {sorted(VALID_LABELS)}"
            )
        return v


# ── Core logic ──────────────────────────────────────────────────────────

def _ensure_file() -> None:
    """Create feedback dir and file if missing."""
    os.makedirs(FEEDBACK_DIR, exist_ok=True)
    if not os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "w") as f:
            json.dump([], f)


def _append_correction(record: CorrectionInput) -> int:
    """Thread-safe append. Returns total count."""
    _ensure_file()
    with _write_lock:
        with open(FEEDBACK_FILE, "r") as f:
            data: list[dict] = json.load(f)
        data.append(record.model_dump())
        with open(FEEDBACK_FILE, "w") as f:
            json.dump(data, f, indent=2)
    return len(data)


def _update_registry_retrain(count: int) -> bool:
    """Set retrain_required=True in registry if count > threshold."""
    retrain: bool = count > RETRAIN_THRESHOLD
    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE, "r") as f:
            registry = json.load(f)
    else:
        registry = {}

    registry["retrain_required"] = retrain
    registry["feedback_count"] = count
    registry["feedback_last_updated"] = (
        datetime.now(timezone.utc).isoformat()
    )

    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2)

    if retrain:
        log.warning(
            "RETRAIN REQUIRED: %d corrections exceed threshold of %d.",
            count, RETRAIN_THRESHOLD,
        )
    return retrain


def get_correction_count() -> int:
    """Return number of stored corrections."""
    _ensure_file()
    with open(FEEDBACK_FILE) as f:
        return len(json.load(f))


# ── FastMCP Application ──────────────────────────────────────────────────────

mcp = FastMCP(
    name="Sentinel-AIOps-Feedback-v2",
    instructions=(
        "Use 'submit_human_correction' to record label fixes. "
        "Retrain is auto-triggered when corrections exceed 100."
    ),
)


@mcp.tool()
def submit_human_correction(
    log_id: str,
    original_pred: str,
    corrected_label: str,
    author_id: str,
) -> Dict[str, Any]:
    """
    Submit a human ground-truth correction.

    Parameters
    ----------
    log_id          : Unique log record ID
    original_pred   : Model's original prediction
    corrected_label : Correct label (validated against known types)
    author_id       : Reviewer's ID (e.g., GitHub username)

    Returns
    -------
    Success: {status, message, record_count, retrain_required}
    Error:   {error, details}
    """
    log.info(
        "Reasoning: Received correction (log=%s, author=%s).",
        log_id, author_id,
    )

    try:
        record = CorrectionInput(
            log_id=log_id,
            original_pred=original_pred,
            corrected_label=corrected_label,
            author_id=author_id,
        )
    except Exception as exc:
        log.warning("Validation failed: %s", exc)
        return {"error": "Validation failed", "details": [str(exc)]}

    count = _append_correction(record)
    retrain = _update_registry_retrain(count)

    log.info(
        "Correction accepted (#%d). retrain_required=%s",
        count, retrain,
    )

    return {
        "status": "accepted",
        "message": (
            f"Correction recorded: '{corrected_label}' for log {log_id}"
        ),
        "record_count": count,
        "retrain_required": retrain,
    }


if __name__ == "__main__":
    log.info("Starting Feedback Handler (stdio transport).")
    mcp.run(transport="stdio")
