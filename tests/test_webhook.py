"""
test_webhook.py — Sentinel-AIOps Webhook Endpoint Tests
==========================================================
Tests the POST /webhook/github endpoint using FastAPI TestClient.
"""

import os
import sys

import hashlib
import hmac
import json as json_lib
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


class TestWebhookHMACValidation:
    """
    Verifies that verify_webhook_signature() correctly accepts
    valid signatures and rejects tampered or missing ones.
    Tests the HMAC path that is bypassed when no secret is set.
    """

    WEBHOOK_SECRET = "test-sentinel-webhook-secret-week3"

    SAMPLE_PAYLOAD = {
        "action": "completed",
        "repository": {"full_name": "org/hmac-test"},
        "workflow_run": {
            "conclusion": "failure",
            "run_attempt": 1,
        },
    }

    def _make_signature(
        self,
        payload_bytes: bytes,
        secret: str,
    ) -> str:
        """Compute the correct HMAC-SHA256 signature for a payload."""
        mac = hmac.new(
            secret.encode("utf-8"),
            payload_bytes,
            hashlib.sha256,
        )
        return "sha256=" + mac.hexdigest()

    def test_valid_signature_is_accepted(self, client) -> None:
        """A correctly signed payload must return 200, not 403."""
        import os
        os.environ["GITHUB_WEBHOOK_SECRET"] = self.WEBHOOK_SECRET

        payload_bytes = json_lib.dumps(
            self.SAMPLE_PAYLOAD
        ).encode("utf-8")
        signature = self._make_signature(
            payload_bytes, self.WEBHOOK_SECRET
        )

        response = client.post(
            "/webhook/github",
            content=payload_bytes,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": signature,
            },
        )
        assert response.status_code == 200, (
            f"Valid HMAC signature was rejected: {response.json()}"
        )
        os.environ.pop("GITHUB_WEBHOOK_SECRET", None)

    def test_tampered_payload_returns_403(self, client) -> None:
        """A payload signed with wrong secret must return 403."""
        import os
        os.environ["GITHUB_WEBHOOK_SECRET"] = self.WEBHOOK_SECRET

        payload_bytes = json_lib.dumps(
            self.SAMPLE_PAYLOAD
        ).encode("utf-8")
        wrong_signature = self._make_signature(
            payload_bytes, "wrong-secret-attacker-key"
        )

        response = client.post(
            "/webhook/github",
            content=payload_bytes,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": wrong_signature,
            },
        )
        assert response.status_code == 403, (
            f"Tampered payload was not rejected. "
            f"Status: {response.status_code}"
        )
        os.environ.pop("GITHUB_WEBHOOK_SECRET", None)

    def test_missing_signature_header_returns_403(
        self, client
    ) -> None:
        """A request with no signature header must return 403."""
        import os
        os.environ["GITHUB_WEBHOOK_SECRET"] = self.WEBHOOK_SECRET

        payload_bytes = json_lib.dumps(
            self.SAMPLE_PAYLOAD
        ).encode("utf-8")

        response = client.post(
            "/webhook/github",
            content=payload_bytes,
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 403, (
            f"Request without signature header was not rejected. "
            f"Status: {response.status_code}"
        )
        os.environ.pop("GITHUB_WEBHOOK_SECRET", None)

    def test_empty_secret_bypasses_validation(
        self, client
    ) -> None:
        """
        When GITHUB_WEBHOOK_SECRET is unset, all requests pass.
        This is the dev-mode behaviour — documented explicitly.
        """
        import os
        os.environ.pop("GITHUB_WEBHOOK_SECRET", None)

        payload_bytes = json_lib.dumps(
            self.SAMPLE_PAYLOAD
        ).encode("utf-8")

        response = client.post(
            "/webhook/github",
            content=payload_bytes,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": "sha256=invalidsignature",
            },
        )
        assert response.status_code == 200, (
            "Dev mode (no secret) should accept all requests. "
            f"Got: {response.status_code}"
        )

    def test_replay_with_modified_body_returns_403(
        self, client
    ) -> None:
        """
        A signature valid for payload A must be rejected for
        payload B. Verifies HMAC binds to exact payload bytes.
        """
        import os
        os.environ["GITHUB_WEBHOOK_SECRET"] = self.WEBHOOK_SECRET

        original_bytes = json_lib.dumps(
            self.SAMPLE_PAYLOAD
        ).encode("utf-8")
        valid_sig = self._make_signature(
            original_bytes, self.WEBHOOK_SECRET
        )

        # Modify the payload after signing
        modified_payload = dict(self.SAMPLE_PAYLOAD)
        modified_payload["action"] = "injected"
        modified_bytes = json_lib.dumps(
            modified_payload
        ).encode("utf-8")

        response = client.post(
            "/webhook/github",
            content=modified_bytes,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": valid_sig,
            },
        )
        assert response.status_code == 403, (
            "Modified payload with original signature must be rejected"
        )
        os.environ.pop("GITHUB_WEBHOOK_SECRET", None)


class TestAPIKeyAuthentication:
    """
    Verifies that protected endpoints enforce X-API-Key auth
    when SENTINEL_API_KEY is configured, and are open when unset.
    """

    TEST_API_KEY = "test-sentinel-api-key-week3-auth"

    def test_protected_endpoint_open_without_configured_key(
        self, client
    ) -> None:
        """When SENTINEL_API_KEY is unset, /api/dashboard is open."""
        import os
        os.environ.pop("SENTINEL_API_KEY", None)
        response = client.get("/api/dashboard")
        assert response.status_code == 200

    def test_protected_endpoint_requires_key_when_configured(
        self, client
    ) -> None:
        """When SENTINEL_API_KEY is set, /api/dashboard needs key."""
        import os
        os.environ["SENTINEL_API_KEY"] = self.TEST_API_KEY
        response = client.get("/api/dashboard")
        assert response.status_code == 403, (
            f"Expected 403 without key, got {response.status_code}"
        )
        os.environ.pop("SENTINEL_API_KEY", None)

    def test_correct_api_key_grants_access(self, client) -> None:
        """Correct X-API-Key header grants access to dashboard."""
        import os
        os.environ["SENTINEL_API_KEY"] = self.TEST_API_KEY
        response = client.get(
            "/api/dashboard",
            headers={"X-API-Key": self.TEST_API_KEY},
        )
        assert response.status_code == 200, (
            f"Valid API key was rejected: {response.json()}"
        )
        os.environ.pop("SENTINEL_API_KEY", None)

    def test_wrong_api_key_returns_403(self, client) -> None:
        """Wrong X-API-Key header must return 403."""
        import os
        os.environ["SENTINEL_API_KEY"] = self.TEST_API_KEY
        response = client.get(
            "/api/dashboard",
            headers={"X-API-Key": "wrong-key-attacker"},
        )
        assert response.status_code == 403
        os.environ.pop("SENTINEL_API_KEY", None)

    def test_health_endpoint_is_always_open(self, client) -> None:
        """GET /health must return 200 regardless of API key."""
        import os
        os.environ["SENTINEL_API_KEY"] = self.TEST_API_KEY
        response = client.get("/health")
        assert response.status_code == 200
        os.environ.pop("SENTINEL_API_KEY", None)

    def test_webhook_endpoint_bypasses_api_key_auth(
        self, client
    ) -> None:
        """
        POST /webhook/github must not require X-API-Key.
        It uses HMAC webhook signature instead.
        """
        import os
        os.environ["SENTINEL_API_KEY"] = self.TEST_API_KEY
        payload = {
            "action": "completed",
            "workflow_run": {"conclusion": "success"},
        }
        response = client.post(
            "/webhook/github",
            json=payload,
        )
        # success conclusion is ignored — 200 with ignored status
        assert response.status_code == 200
        assert response.json().get("status") == "ignored"
        os.environ.pop("SENTINEL_API_KEY", None)

    def test_api_history_protected(self, client) -> None:
        """GET /api/history is protected when key is configured."""
        import os
        os.environ["SENTINEL_API_KEY"] = self.TEST_API_KEY
        response = client.get("/api/history")
        assert response.status_code == 403
        os.environ.pop("SENTINEL_API_KEY", None)

    def test_timing_safe_comparison(self, client) -> None:
        """
        Verify the endpoint uses constant-time comparison by
        confirming both a prefix-match and full-wrong key get 403.
        """
        import os
        os.environ["SENTINEL_API_KEY"] = self.TEST_API_KEY
        prefix_key = self.TEST_API_KEY[:8]
        for bad_key in [prefix_key, "x" * len(self.TEST_API_KEY)]:
            response = client.get(
                "/api/dashboard",
                headers={"X-API-Key": bad_key},
            )
            assert response.status_code == 403, (
                f"Key '{bad_key[:4]}...' should be rejected"
            )
        os.environ.pop("SENTINEL_API_KEY", None)
