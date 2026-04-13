"""
test_pipeline.py — Sentinel-AIOps End-to-End Pipeline Integration Test
========================================================================
Tests the complete Webhook → SQL → Dashboard data pipeline:

1. POST /webhook/github (actionable failure)        → 200, DB record created
2. GET  /api/history                                → new record visible
3. GET  /api/dashboard                              → inference_count updated
4. POST /webhook/github (ignored / non-failure)     → 200 ignored, no new DB row

Note: database initialization is handled by tests/conftest.py which creates
a single shared in-memory SQLite DB with the full Phase 10 schema.
"""

import os
import sys

import pytest
from fastapi.testclient import TestClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dashboard.app import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    """Module-scoped TestClient so DB accumulates between tests."""
    return TestClient(app)


# ── Shared payload ────────────────────────────────────────────────────────────

FAILURE_PAYLOAD = {
    "action": "completed",
    "repository": {"full_name": "org/sentinel-ci"},
    "workflow_run": {
        "name": "Integration CI",
        "conclusion": "failure",
        "run_started_at": "2026-03-02T14:00:00Z",
        "updated_at": "2026-03-02T14:07:30Z",
        "run_attempt": 1,
        "actor": {"login": "ci-bot"},
        "head_repository": {"language": "Python"},
    },
}

IGNORED_PAYLOAD = {
    "action": "completed",
    "repository": {"full_name": "org/sentinel-ci"},
    "workflow_run": {
        "conclusion": "success",  # <── not a failure
    },
}

IN_PROGRESS_PAYLOAD = {
    "action": "in_progress",  # <── not "completed"
    "workflow_run": {"conclusion": "failure"},
}


class TestWebhookToSQLToDashboard:
    """Full pipeline integration tests."""

    def test_webhook_failure_returns_200(self, client) -> None:
        """A genuine failure webhook returns 200."""
        response = client.post("/webhook/github", json=FAILURE_PAYLOAD)
        assert response.status_code == 200

    def test_webhook_failure_not_ignored(self, client) -> None:
        """A failure webhook is NOT returned as 'ignored'."""
        response = client.post("/webhook/github", json=FAILURE_PAYLOAD)
        data = response.json()
        assert data.get("status") != "ignored", f"Should not be ignored: {data}"

    def test_db_record_appears_in_history(self, client) -> None:
        """After posting a failure webhook, /api/history has at least one record."""
        # Post a webhook to ensure at least one entry
        client.post("/webhook/github", json=FAILURE_PAYLOAD)

        response = client.get("/api/history?limit=50")
        assert response.status_code == 200
        history = response.json()
        assert isinstance(history, list)
        assert len(history) >= 1, "Expected at least one DB record after webhook"

    def test_db_record_has_github_event_source(self, client) -> None:
        """Records from the webhook are tagged event_source=github_webhook."""
        # Post and then inspect
        client.post("/webhook/github", json=FAILURE_PAYLOAD)
        response = client.get("/api/history?limit=10")
        history = response.json()
        sources = {r.get("event_source") for r in history}
        assert (
            "github_webhook" in sources
        ), f"Expected 'github_webhook' in event sources, got: {sources}"

    def test_dashboard_inference_count_increments(self, client) -> None:
        """After webhook, /api/dashboard inference_count is > 0."""
        client.post("/webhook/github", json=FAILURE_PAYLOAD)

        response = client.get("/api/dashboard")
        assert response.status_code == 200
        data = response.json()
        assert (
            data["inference_count"] >= 1
        ), f"inference_count should be >= 1, got: {data['inference_count']}"

    def test_success_webhook_is_ignored(self, client) -> None:
        """A success-conclusion webhook returns {status: ignored}."""
        before = client.get("/api/history?limit=200").json()
        before_count = len(before)

        response = client.post("/webhook/github", json=IGNORED_PAYLOAD)
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "ignored", f"Expected ignored, got: {data}"

        after = client.get("/api/history?limit=200").json()
        assert (
            len(after) == before_count
        ), "Success webhook should not create a new DB record"

    def test_in_progress_webhook_is_ignored(self, client) -> None:
        """A non-completed action webhook returns {status: ignored}."""
        response = client.post("/webhook/github", json=IN_PROGRESS_PAYLOAD)
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "ignored"

    def test_timed_out_conclusion_is_processed(self, client) -> None:
        """A timed_out conclusion is treated as actionable."""
        payload = {
            "action": "completed",
            "workflow_run": {
                "conclusion": "timed_out",
                "run_attempt": 3,
            },
        }
        response = client.post("/webhook/github", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") != "ignored"

    def test_raw_payload_stored(self, client) -> None:
        """DB entries from webhook contain raw_payload for audit."""
        client.post("/webhook/github", json=FAILURE_PAYLOAD)
        response = client.get("/api/history?limit=10")
        history = response.json()
        webhook_entries = [
            r for r in history if r.get("event_source") == "github_webhook"
        ]
        assert len(webhook_entries) > 0
        latest = webhook_entries[0]
        assert (
            latest.get("raw_payload") is not None
        ), "webhook entries must have raw_payload for audit"
