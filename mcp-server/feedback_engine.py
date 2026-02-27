"""
feedback_engine.py — Sentinel-AIOps Human Feedback Infrastructure
===================================================================
Provides a thread-safe mechanism for submitting ground-truth corrections
via the FastMCP protocol. All feedback is appended to:
    /data/feedback/human_labels.json

Workflow Rules (AGENTS.md):
  - Log 'Reasoning' before execution.
  - Type hinting + Pydantic validation throughout.
"""

import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field, field_validator

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("sentinel.feedback")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEEDBACK_DIR = os.path.join(ROOT, "data", "feedback")
FEEDBACK_FILE = os.path.join(FEEDBACK_DIR, "human_labels.json")

# ── Thread lock for file-safe writes ─────────────────────────────────────────
_write_lock = threading.Lock()

# ── Valid failure types (AGENTS.md + data_summary.json) ───────────────────────
VALID_LABELS: set[str] = {
    "Build Failure", "Configuration Error", "Dependency Error",
    "Deployment Failure", "Network Error", "Permission Error",
    "Resource Exhaustion", "Security Scan Failure", "Test Failure", "Timeout",
}


# ── Pydantic Schema ──────────────────────────────────────────────────────────

class FeedbackRecord(BaseModel):
    """Strict schema for human ground-truth corrections."""
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 UTC timestamp of feedback submission",
    )
    log_id: str = Field(
        ...,
        min_length=1,
        description="Unique ID of the original log record",
    )
    original_pred: str = Field(
        ...,
        description="The model's original prediction for this log",
    )
    corrected_label: str = Field(
        ...,
        description="The human-corrected ground-truth label",
    )
    author_id: str = Field(
        ...,
        min_length=1,
        description="ID of the human reviewer (e.g., GitHub username)",
    )

    @field_validator("corrected_label")
    @classmethod
    def validate_label(cls, v: str) -> str:
        if v not in VALID_LABELS:
            raise ValueError(
                f"Invalid label '{v}'. Must be one of: {sorted(VALID_LABELS)}"
            )
        return v


class FeedbackResponse(BaseModel):
    """Response schema for feedback submission."""
    status: str
    message: str
    record_count: int
    feedback_file: str


class FeedbackError(BaseModel):
    """Error response for invalid feedback."""
    error: str
    details: list[str]


# ── Core logic ────────────────────────────────────────────────────────────────

def _ensure_feedback_dir() -> None:
    """Create feedback directory and file if they don't exist."""
    os.makedirs(FEEDBACK_DIR, exist_ok=True)
    if not os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "w") as f:
            json.dump([], f)
        log.info("Created feedback file: %s", FEEDBACK_FILE)


def _append_feedback(record: FeedbackRecord) -> int:
    """Thread-safe append of a feedback record. Returns total record count."""
    _ensure_feedback_dir()

    with _write_lock:
        # Read existing
        with open(FEEDBACK_FILE, "r") as f:
            data: list[dict] = json.load(f)

        # Append
        data.append(record.model_dump())

        # Write back
        with open(FEEDBACK_FILE, "w") as f:
            json.dump(data, f, indent=2)

    return len(data)


def get_feedback_count() -> int:
    """Return the number of feedback records stored."""
    _ensure_feedback_dir()
    with open(FEEDBACK_FILE, "r") as f:
        data = json.load(f)
    return len(data)


def get_all_feedback() -> list[dict]:
    """Return all stored feedback records."""
    _ensure_feedback_dir()
    with open(FEEDBACK_FILE, "r") as f:
        return json.load(f)


# ── FastMCP Application ──────────────────────────────────────────────────────

mcp = FastMCP(
    name="Sentinel-AIOps-Feedback",
    instructions=(
        "You are the Sentinel-AIOps feedback engine. "
        "Use 'submit_ground_truth' to record human label corrections. "
        "Use 'get_feedback_stats' to view feedback summary."
    ),
)


@mcp.tool()
def submit_ground_truth(
    log_id: str,
    original_pred: str,
    corrected_label: str,
    author_id: str,
) -> Dict[str, Any]:
    """
    Submit a human ground-truth correction for a model prediction.

    Parameters
    ----------
    log_id          : Unique ID of the original CI/CD log record
    original_pred   : What the model predicted (e.g., 'Network Error')
    corrected_label : The correct label (must be a valid failure type)
    author_id       : GitHub username or ID of the reviewer

    Returns
    -------
    On success: {"status": "accepted", "message": ..., "record_count": N}
    On error:   {"error": "Validation failed", "details": [...]}
    """
    log.info("Reasoning: Received ground-truth submission (log_id=%s, author=%s).",
             log_id, author_id)

    try:
        record = FeedbackRecord(
            log_id=log_id,
            original_pred=original_pred,
            corrected_label=corrected_label,
            author_id=author_id,
        )
    except Exception as exc:
        log.warning("Validation failed: %s", exc)
        return FeedbackError(
            error="Validation failed",
            details=[str(exc)],
        ).model_dump()

    count = _append_feedback(record)
    log.info("Feedback accepted. Total records: %d", count)

    return FeedbackResponse(
        status="accepted",
        message=f"Ground truth recorded: '{corrected_label}' for log {log_id}",
        record_count=count,
        feedback_file=FEEDBACK_FILE,
    ).model_dump()


@mcp.tool()
def get_feedback_stats() -> Dict[str, Any]:
    """Return summary statistics about collected feedback."""
    log.info("Reasoning: Retrieving feedback stats.")
    records = get_all_feedback()

    if not records:
        return {"total_records": 0, "message": "No feedback collected yet."}

    # Compute stats
    labels = [r.get("corrected_label", "unknown") for r in records]
    authors = set(r.get("author_id", "unknown") for r in records)
    mismatches = sum(
        1 for r in records
        if r.get("original_pred") != r.get("corrected_label")
    )

    from collections import Counter
    label_dist = dict(Counter(labels))

    return {
        "total_records": len(records),
        "unique_authors": len(authors),
        "correction_rate": round(mismatches / len(records), 4) if records else 0,
        "label_distribution": label_dist,
    }


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("Starting Sentinel-AIOps Feedback Engine (stdio transport).")
    mcp.run(transport="stdio")
