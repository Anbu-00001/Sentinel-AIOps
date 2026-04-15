"""
test_webhook.py — Sentinel-AIOps Webhook Endpoint Tests
==========================================================
Tests the POST /webhook/github endpoint using FastAPI TestClient.
"""

import os
import sys

import pytest
from fastapi.testclient import TestClient

# ── Allow imports from project root ──────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Override DB path to use a temp in-memory database for tests
os.environ["SENTINEL_DB_PATH"] = ":memory:"

from dashboard.app import app  # noqa: E402


@pytest.fixture()
def client():
    """Provide a TestClient for the dashboard app."""
    return TestClient(app)


class TestGitHubWebhook:
    """Tests for the POST /webhook/github endpoint."""

    def test_webhook_valid_workflow_run(self, client) -> None:
        """A valid workflow_run payload returns 200 with expected fields."""
        payload = {
            "action": "completed",
            "repository": {
                "full_name": "org/sentinel-test",
            },
            "workflow_run": {
                "name": "CI Pipeline",
                "conclusion": "failure",
                "run_started_at": "2026-03-01T10:00:00Z",
                "updated_at": "2026-03-01T10:05:30Z",
                "run_attempt": 2,
                "actor": {"login": "dev-tester"},
                "head_repository": {"language": "Python"},
            },
        }
        response = client.post("/webhook/github", json=payload)
        assert response.status_code == 200
        data = response.json()
        # Should contain either a prediction or a queued status
        assert (
            "prediction" in data or "status" in data or "error" in data
        ), f"Unexpected response: {data}"

    def test_webhook_valid_check_run(self, client) -> None:
        """A check_run payload is accepted."""
        payload = {
            "action": "completed",
            "repository": {"full_name": "org/another-repo"},
            "check_run": {
                "conclusion": "timed_out",
                "output": {
                    "summary": "Job exceeded time limit on build step",
                },
            },
        }
        response = client.post("/webhook/github", json=payload)
        assert response.status_code == 200

    def test_webhook_minimal_payload_is_ignored(self, client) -> None:
        """A payload with only 'action' and no failure conclusion is ignored."""
        payload = {"action": "completed"}
        response = client.post("/webhook/github", json=payload)
        assert response.status_code == 200
        # No workflow_run → conclusion is empty → should be ignored
        data = response.json()
        assert data.get("status") == "ignored"

    def test_webhook_missing_action_returns_400(self, client) -> None:
        """Missing the required 'action' field triggers validation error (400)."""
        payload = {"repository": {"full_name": "org/test"}}
        response = client.post("/webhook/github", json=payload)
        assert response.status_code == 400

    def test_webhook_empty_body_returns_400(self, client) -> None:
        """An empty JSON body triggers validation error (400)."""
        response = client.post("/webhook/github", json={})
        assert response.status_code == 400

    def test_webhook_success_conclusion_ignored(self, client) -> None:
        """A success conclusion returns 200 with status: ignored."""
        payload = {
            "action": "completed",
            "workflow_run": {"conclusion": "success"},
        }
        response = client.post("/webhook/github", json=payload)
        assert response.status_code == 200
        assert response.json().get("status") == "ignored"


class TestDashboardEndpoints:
    """Smoke tests for core dashboard API endpoints."""

    def test_health_check(self, client) -> None:
        """GET /health returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_api_history(self, client) -> None:
        """GET /api/history returns a list."""
        response = client.get("/api/history?limit=10")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_dashboard_ui_renders(self, client) -> None:
        """GET / returns HTML with Sentinel-AIOps title."""
        response = client.get("/")
        assert response.status_code == 200
        assert "Sentinel-AIOps" in response.text
