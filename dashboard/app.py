"""
app.py — Sentinel-AIOps Observability Dashboard
=================================================
FastAPI-based dashboard providing:
  - System Health Badge (🟢 Healthy / 🟡 Drift Detected / 🔴 Training Required)
  - Drift Heatmap showing per-feature drift scores
  - Model registry info and feedback stats
  - Inference history powered by SQLite
  - GitHub Webhook ingestion endpoint

Workflow Rules (AGENTS.md):
  - Log 'Reasoning' before execution.
  - Type hinting + Pydantic validation throughout.
"""

import hashlib
import hmac
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Optional

from fastapi import Depends, FastAPI, Query, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from fastapi.concurrency import run_in_threadpool
import joblib

# ── Webhook Security ──────────────────────────────────────────────────────────


def verify_webhook_signature(payload_body: bytes, signature_header: str, secret: str) -> bool:
    """
    Verify GitHub webhook HMAC-SHA256 signature.

    Args:
        payload_body: Raw request body bytes.
        signature_header: Value of X-Hub-Signature-256 header.
        secret: The shared webhook secret.

    Returns:
        True if signature is valid, False otherwise.
    """
    if not secret:
        log_msg = "GITHUB_WEBHOOK_SECRET not configured — skipping signature verification."
        logging.getLogger("sentinel.dashboard").warning(log_msg)
        return True  # Allow in dev; block in prod by setting the env var
    if not signature_header:
        return False
    expected = "sha256=" + hmac.new(
        secret.encode("utf-8"), payload_body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature_header)


# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import LogEntry, get_session, init_db  # noqa: E402
from config import PSI_SEVERE_THRESHOLD, PSI_MODERATE_THRESHOLD, RETRAIN_THRESHOLD

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("sentinel.dashboard")

# ── Import Decoupled ML Logic ───────────────────────────────────────────────
from mcp_server.logic import run_prediction  # noqa: E402

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
FEEDBACK_FILE = os.path.join(ROOT, "data", "feedback", "human_labels.json")

# ── Initialize database ──────────────────────────────────────────────────────
log.info("Reasoning: Initializing SQLite database for dashboard queries.")
init_db()

# ── Load Mathematical Artifacts ──────────────────────────────────────────────
log.info(
    "Reasoning: Loading model artifacts for mathematical synchronization & inline inference."
)

import sys
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from models import crypto_sig


def _secure_load(artifact_name):
    filepath = os.path.join(MODELS_DIR, artifact_name)
    if not crypto_sig.verify_artifact(filepath):
        if os.getenv("TESTING") == "1":
            return None
        log.error("CRITICAL: Artifact signature validation failed for %s", filepath)
        raise SystemExit(1)
    return joblib.load(filepath)


_scaler = None
_model = None
_le = None
_hasher = None
_tfidf = None
_meta = None
_feature_names = []
_BASELINE_MEANS = {}
_INFERENCE_READY = False

# ── Pydantic Models ──────────────────────────────────────────────────────────


class HealthStatus(BaseModel):
    """System health status."""

    badge: str
    label: str
    color: str
    retrain_suggested: bool
    features_drifted: int
    total_features: int
    last_drift_check: Optional[str] = None


class DriftHeatmapEntry(BaseModel):
    """Single cell in the drift heatmap."""

    feature: str
    method: str
    score: float
    severity: str
    is_drifted: bool


class LogHistoryEntry(BaseModel):
    """A single inference history record."""

    id: int
    timestamp: Optional[str] = None
    prediction: str
    confidence: float
    top_features: Optional[list] = None


class DashboardData(BaseModel):
    """Complete dashboard payload."""

    model_config = {"protected_namespaces": ()}

    health: HealthStatus
    drift_heatmap: list[DriftHeatmapEntry]
    model_version: Optional[str] = None
    feedback_count: int = 0
    inference_count: int = 0


class GitHubWebhookPayload(BaseModel):
    """
    Accepts standard GitHub Actions workflow_run / check_run webhook events.
    Only the fields we need for mapping are required; the rest are ignored.
    """

    action: str = Field(..., description="Event action (e.g. 'completed')")
    repository: Optional[dict] = Field(default=None)
    workflow_run: Optional[dict] = Field(default=None)
    check_run: Optional[dict] = Field(default=None)


# ── FastAPI Application ──────────────────────────────────────────────────────

from starlette.middleware.base import BaseHTTPMiddleware


class PayloadSizeLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_upload_size: int):
        super().__init__(app)
        self.max_upload_size = max_upload_size

    async def dispatch(self, request: Request, call_next):
        if request.headers.get("content-length"):
            content_length = int(request.headers.get("content-length", 0))
            if content_length > self.max_upload_size:
                return JSONResponse(status_code=413, content={"detail": "Payload Too Large"})
        return await call_next(request)


from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan_handler(app: FastAPI):
    load_artifacts()
    yield


app = FastAPI(
    title="Sentinel-AIOps Dashboard",
    description="Observability dashboard for CI/CD anomaly detection platform.",
    version="2.0.0",
    lifespan=lifespan_handler,
)

app.add_middleware(PayloadSizeLimitMiddleware, max_upload_size=2_000_000)


# ── API Key Authentication ───────────────────────────────────────
_API_KEY_HEADER = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    description="API key for dashboard endpoint authentication.",
)


def _load_api_key() -> str:
    """
    Load SENTINEL_API_KEY from environment at call time.
    Returns empty string if not set (development mode).
    """
    return os.getenv("SENTINEL_API_KEY", "")


async def require_api_key(
    api_key_header: str = Depends(_API_KEY_HEADER),
) -> None:
    """
    FastAPI dependency that enforces API key authentication.

    Behaviour matrix:
      SENTINEL_API_KEY unset + no header  → 200 (dev mode, open)
      SENTINEL_API_KEY unset + any header → 200 (dev mode, open)
      SENTINEL_API_KEY set + correct key  → 200 (authenticated)
      SENTINEL_API_KEY set + wrong key    → 403 (rejected)
      SENTINEL_API_KEY set + no header    → 403 (rejected)

    Dev mode (SENTINEL_API_KEY unset) allows unauthenticated
    access so local development requires no configuration.
    Production MUST set SENTINEL_API_KEY via environment.
    """
    configured_key = _load_api_key()
    if not configured_key:
        # Dev mode: no key configured, allow all requests.
        log.debug(
            "SENTINEL_API_KEY not set — running in open dev mode. "
            "Set this variable before exposing to a network."
        )
        return

    if not api_key_header:
        log.warning(
            "API request rejected: X-API-Key header missing."
        )
        raise HTTPException(
            status_code=403,
            detail=(
                "X-API-Key header is required. "
                "Set SENTINEL_API_KEY in your environment and "
                "pass it as the X-API-Key request header."
            ),
        )

    if not hmac.compare_digest(
        api_key_header.encode("utf-8"),
        configured_key.encode("utf-8"),
    ):
        log.warning(
            "API request rejected: invalid X-API-Key value."
        )
        raise HTTPException(
            status_code=403,
            detail="Invalid API key.",
        )


def load_artifacts():
    global _scaler, _model, _le, _hasher, _tfidf, _meta, _feature_names, _BASELINE_MEANS, _INFERENCE_READY
    try:
        _scaler = _secure_load("scaler.joblib")
        _model = _secure_load("lgbm_model.joblib")
        _le = _secure_load("label_encoder.joblib")
        _hasher = _secure_load("hasher.joblib")
        _tfidf = _secure_load("tfidf.joblib")

        with open(os.path.join(MODELS_DIR, "feature_meta.json")) as f:
            _meta = json.load(f)

        _feature_names = (
            _meta["numerical_cols"]
            + [f"hash_{i}" for i in range(_meta["hash_n_features"])]
            + [f"tfidf_{i}" for i in range(_meta["tfidf_max_features"])]
            + _meta["dummy_column_order"]
            + _meta["bool_cols"]
        )

        # Derive dynamic baselines from fitted scaler means
        _BASELINE_MEANS = {
            col: float(_scaler.mean_[i]) for i, col in enumerate(_meta["numerical_cols"])
        }
        _INFERENCE_READY = True
        log.info("Dashboard Artifacts Loaded. Feature Dimension: %d", len(_feature_names))
    except Exception as exc:
        log.warning(
            "Failed to load artifacts for synchronization: %s. Using safe defaults.", exc
        )
        _INFERENCE_READY = False
        _BASELINE_MEANS = {
            "build_duration_sec": 1800.0,
            "test_duration_sec": 300.0,
            "deploy_duration_sec": 150.0,
            "cpu_usage_pct": 50.0,
            "memory_usage_mb": 8192.0,
            "retry_count": 2.5,
        }


def _load_drift_report() -> Optional[dict]:
    """Load the latest drift report."""
    path = os.path.join(MODELS_DIR, "drift_report.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _load_registry() -> Optional[dict]:
    """Load the model registry."""
    path = os.path.join(MODELS_DIR, "registry.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _load_feedback_count() -> int:
    """Count feedback records."""
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE) as f:
            return len(json.load(f))
    return 0


def _get_inference_count() -> int:
    """Query total inference count from database."""
    try:
        with get_session() as session:
            return session.query(LogEntry).count()
    except Exception:
        return 0


def _compute_health_from_db() -> Optional[dict]:
    """
    Derive a live drift signal from the last inference records.

    Returns a dict with keys:
      - db_mean_confidence: mean confidence of recent inferences
      - db_retrain_suggested: True if mean confidence < 0.45
      - db_sample_size: number of rows used
    Returns None if DB has no data.
    """
    try:
        with get_session() as session:
            rows = (
                session.query(LogEntry.confidence)
                .order_by(LogEntry.timestamp.desc())
                .limit(RETRAIN_THRESHOLD)
                .all()
            )
        if not rows:
            return None
        confidences = [r[0] for r in rows]
        mean_conf = sum(confidences) / len(confidences)
        return {
            "db_mean_confidence": round(mean_conf, 4),
            "db_retrain_suggested": mean_conf < 0.45,
            "db_sample_size": len(confidences),
        }
    except Exception:
        return None


def _compute_health(
    drift_report: Optional[dict],
    registry: Optional[dict],
    db_signal: Optional[dict] = None,
) -> HealthStatus:
    """
    Determine system health from drift_report (PSI) + live DB confidence signal.

    Priority: retrain flag from JSON report > DB mean-confidence drift > drifted features.
    """
    n_drifted: int = drift_report.get("features_drifted", 0) if drift_report else 0
    total: int = drift_report.get("features_analyzed", 0) if drift_report else 0
    last_check: Optional[str] = registry.get("last_drift_check") if registry else None

    retrain_json: bool = (
        drift_report.get("retrain_suggested", False) if drift_report else False
    )
    retrain_db: bool = (
        db_signal.get("db_retrain_suggested", False) if db_signal else False
    )
    retrain: bool = retrain_json or retrain_db

    if retrain:
        source = (
            "JSON report"
            if retrain_json
            else f"DB signal (mean_conf={db_signal['db_mean_confidence']:.3f})"
        )
        log.info("Training Required triggered by: %s", source)
        return HealthStatus(
            badge="🔴",
            label="Training Required",
            color="#ef4444",
            retrain_suggested=True,
            features_drifted=n_drifted,
            total_features=total,
            last_drift_check=last_check,
        )
    elif n_drifted > 0:
        return HealthStatus(
            badge="🟡",
            label="Drift Detected",
            color="#eab308",
            retrain_suggested=False,
            features_drifted=n_drifted,
            total_features=total,
            last_drift_check=last_check,
        )
    else:
        return HealthStatus(
            badge="🟢",
            label="Healthy",
            color="#22c55e",
            retrain_suggested=False,
            features_drifted=n_drifted,
            total_features=total,
            last_drift_check=last_check,
        )


def _build_heatmap(drift_report: Optional[dict]) -> list[DriftHeatmapEntry]:
    """Build drift heatmap entries from report."""
    if drift_report is None:
        return []
    return [
        DriftHeatmapEntry(
            feature=r["feature"],
            method=r["method"],
            score=r["score"],
            severity=r["severity"],
            is_drifted=r["is_drifted"],
        )
        for r in drift_report.get("feature_results", [])
    ]


# ── Dynamic Drift Configuration ──────────────────────────────────────────────
_DRIFT_FEATURES = [
    "build_duration_sec",
    "test_duration_sec",
    "deploy_duration_sec",
    "cpu_usage_pct",
    "memory_usage_mb",
    "retry_count",
]

# Epsilon Smoothing prevents mathematical asymptotes in drift deviation
EPSILON = 1e-4


def _compute_dynamic_psi() -> list[DriftHeatmapEntry]:
    """
    Compute a live PSI-like drift score for key numeric features
    using the recent inference records from SQLite.

    Method: for each feature, compare the mean of the recent
    observations against the training baseline mean. We compute
    a normalised deviation that mimics PSI severity thresholds.
    """
    try:
        with get_session() as session:
            rows = (
                session.query(LogEntry.metrics_payload)
                .order_by(LogEntry.timestamp.desc())
                .limit(RETRAIN_THRESHOLD)
                .all()
            )
    except Exception as exc:
        log.warning("Dynamic PSI query failed: %s", exc)
        return []

    if not rows:
        return []

    # Aggregate feature values from JSON payloads
    feature_values: dict[str, list[float]] = {f: [] for f in _DRIFT_FEATURES}
    for (payload,) in rows:
        if not isinstance(payload, dict):
            continue
        for feat in _DRIFT_FEATURES:
            val = payload.get(feat)
            if isinstance(val, (int, float)):
                feature_values[feat].append(float(val))

    entries: list[DriftHeatmapEntry] = []
    for feat in _DRIFT_FEATURES:
        vals = feature_values[feat]
        if not vals:
            continue
        live_mean = sum(vals) / len(vals)
        baseline = _BASELINE_MEANS.get(feat, 1.0)
        # Normalised absolute deviation with Epsilon Smoothing
        psi_score = abs(live_mean - baseline) / (baseline + EPSILON)
        psi_score = round(min(psi_score, 1.0), 4)

        if psi_score >= PSI_SEVERE_THRESHOLD:
            severity = "severe"
        elif psi_score >= PSI_MODERATE_THRESHOLD:
            severity = "moderate"
        else:
            severity = "stable"

        entries.append(
            DriftHeatmapEntry(
                feature=feat,
                method="PSI (live-DB)",
                score=psi_score,
                severity=severity,
                is_drifted=psi_score >= PSI_MODERATE_THRESHOLD,
            )
        )

    return entries


def _get_heatmap(drift_report: Optional[dict]) -> list[DriftHeatmapEntry]:
    """
    Return heatmap entries, preferring the static drift_report.json but
    falling back (and merging) with the live SQL-computed PSI scores.
    """
    static_entries = _build_heatmap(drift_report)
    dynamic_entries = _compute_dynamic_psi()

    if not static_entries:
        # No JSON report — use dynamic SQL-based PSI exclusively
        return dynamic_entries

    # Merge: prefer static entries but append any dynamic features
    # that are not already covered by the JSON report
    static_features = {e.feature for e in static_entries}
    extra = [e for e in dynamic_entries if e.feature not in static_features]
    return static_entries + extra


def _map_github_to_features(payload: GitHubWebhookPayload) -> dict[str, Any]:
    """
    Map a GitHub webhook payload into the analyze_log feature schema.

    Uses workflow_run or check_run data to synthesize CI/CD metrics.
    For cpu_usage_pct and memory_usage_mb, extracts from the payload
    if custom fields are present, otherwise uses deterministic estimates.
    """
    repo_name = ""
    author = ""
    failure_stage = "build"
    error_message = "GitHub Action workflow failure"

    if payload.repository:
        repo_name = payload.repository.get("full_name", "")

    wf = payload.workflow_run or {}
    cr = payload.check_run or {}

    # Compute durations from timestamps if available
    build_dur = 120
    test_dur = 60
    deploy_dur = 30

    if wf.get("run_started_at") and wf.get("updated_at"):
        try:
            started = datetime.fromisoformat(
                wf["run_started_at"].replace("Z", "+00:00")
            )
            ended = datetime.fromisoformat(wf["updated_at"].replace("Z", "+00:00"))
            total_sec = int((ended - started).total_seconds())
            build_dur = max(10, total_sec // 3)
            test_dur = max(5, total_sec // 3)
            deploy_dur = max(5, total_sec // 3)
        except (ValueError, TypeError):
            pass

    if wf.get("actor", {}).get("login"):
        author = wf["actor"]["login"]

    conclusion = wf.get("conclusion", cr.get("conclusion", "failure"))
    if conclusion in ("timed_out", "stale"):
        failure_stage = "deploy"
        error_message = f"GitHub Action timed out: {conclusion}"
    elif conclusion == "failure":
        failure_stage = "build"
        error_message = wf.get("name", "Workflow") + " failed"

    if cr.get("output", {}).get("summary"):
        error_message = cr["output"]["summary"][:200]

    return {
        "build_duration_sec": build_dur,
        "test_duration_sec": test_dur,
        "deploy_duration_sec": deploy_dur,
        # Extract cpu/memory from payload if client passes them in workflow metadata
        # (via repository dispatch client_payload or outputs). Fall back to
        # deterministic, mathematically sound means from _BASELINE_MEANS.
        "cpu_usage_pct": float(
            wf.get("cpu_usage_pct")  # custom field if present
            or (payload.repository or {}).get("cpu_usage_pct", None)
            or _BASELINE_MEANS.get("cpu_usage_pct", 50.0)  # mathematically robust baseline
        ),
        "memory_usage_mb": int(
            wf.get("memory_usage_mb")  # custom field if present
            or (payload.repository or {}).get("memory_usage_mb", None)
            or _BASELINE_MEANS.get("memory_usage_mb", 8192)  # mathematically robust baseline
        ),
        "retry_count": max(0, wf.get("run_attempt", 1) - 1),
        "error_message": error_message,
        "repository": repo_name,
        "author": author,
        "failure_stage": failure_stage,
        "severity": "HIGH" if conclusion in ("failure", "timed_out") else "MEDIUM",
        "ci_tool": "GitHub Actions",
        "language": wf.get("head_repository", {}).get("language", "Python") or "Python",
        "os": "Linux",
        "cloud_provider": "GitHub",
        "is_flaky_test": False,
        "rollback_triggered": False,
        "incident_created": False,
    }


# ── API Endpoints ──────────────────────────────────────────────────────────────


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Basic health check."""
    return {"status": "healthy", "service": "sentinel-dashboard"}


@app.get("/api/dashboard", dependencies=[Depends(require_api_key)])
async def get_dashboard_data() -> DashboardData:
    """Return complete dashboard payload as JSON."""
    log.info("Reasoning: Building dashboard data payload.")
    drift_report = _load_drift_report()
    registry = _load_registry()

    db_signal = _compute_health_from_db()
    return DashboardData(
        health=_compute_health(drift_report, registry, db_signal),
        drift_heatmap=_get_heatmap(drift_report),
        model_version=registry.get("latest") if registry else None,
        feedback_count=_load_feedback_count(),
        inference_count=_get_inference_count(),
    )


@app.get("/api/drift", dependencies=[Depends(require_api_key)])
async def get_drift_report() -> dict[str, Any]:
    """Return raw drift report."""
    report = _load_drift_report()
    if report is None:
        return {"message": "No drift report available. Run drift_monitor.py first."}
    return report


@app.get("/api/psi", dependencies=[Depends(require_api_key)])
async def get_psi_scores() -> dict[str, Any]:
    """
    Return live PSI drift scores computed from the last 100 SQL inference records.

    Compares the 'current window' (SQLite last-100 rows) against the
    'training baseline' stored in _BASELINE_MEANS. Returns per-feature scores,
    severity labels, and an overall system status.
    """
    log.info("Reasoning: Computing live PSI scores from SQL window.")
    entries = _compute_dynamic_psi()
    if not entries:
        return {
            "status": "no_data",
            "message": "No inference records in database yet.",
            "features": [],
        }

    severe = [e for e in entries if e.severity == "severe"]
    moderate = [e for e in entries if e.severity == "moderate"]

    if severe:
        overall = "Training Required"
    elif moderate:
        overall = "Drift Detected"
    else:
        overall = "Healthy"

    return {
        "status": overall,
        "sample_size": RETRAIN_THRESHOLD,
        "baseline_source": "training_means",
        "features": [
            {
                "feature": e.feature,
                "method": e.method,
                "psi_score": e.score,
                "severity": e.severity,
                "is_drifted": e.is_drifted,
            }
            for e in entries
        ],
    }


@app.get("/api/registry", dependencies=[Depends(require_api_key)])
async def get_registry() -> dict[str, Any]:
    """Return model registry."""
    registry = _load_registry()
    if registry is None:
        return {"message": "No registry found."}
    return registry


@app.get("/api/history", dependencies=[Depends(require_api_key)])
async def get_history(
    limit: int = Query(default=100, ge=1, le=1000, description="Max records to return"),
) -> list[dict]:
    """Return recent inference history from the database."""
    log.info("Reasoning: Querying inference history (limit=%d).", limit)
    try:
        with get_session() as session:
            rows = (
                session.query(LogEntry)
                .order_by(LogEntry.timestamp.desc())
                .limit(limit)
                .all()
            )
            return [r.to_dict() for r in rows]
    except Exception as exc:
        log.error("Failed to query history: %s", exc)
        return []


@app.post("/webhook/github")
async def github_webhook(request: Request) -> JSONResponse:
    """
    Receive a GitHub Actions webhook (workflow_run events).

    Only processes events where action=="completed" AND
    conclusion is "failure" or "timed_out". All other events
    return a 200 {"status": "ignored"} response.

    Security: Verifies X-Hub-Signature-256 HMAC when GITHUB_WEBHOOK_SECRET is set.

    Configure in GitHub → Settings → Webhooks:
      Payload URL:  http://<your-host>:8200/webhook/github
      Content type: application/json
      Events:       Workflow runs
      Secret:       <same value as GITHUB_WEBHOOK_SECRET env var>
    """
    # ── HMAC Signature Verification ───────────────────────────────────
    raw_body = await request.body()
    signature = request.headers.get("X-Hub-Signature-256", "")
    _webhook_secret = os.getenv("GITHUB_WEBHOOK_SECRET", "")
    if not verify_webhook_signature(raw_body, signature, _webhook_secret):
        log.warning("Webhook signature verification FAILED. Rejecting request.")
        raise HTTPException(status_code=403, detail="Invalid webhook signature.")

    # Parse and validate the payload
    try:
        body_json = json.loads(raw_body)
        payload = GitHubWebhookPayload(**body_json)
    except Exception as exc:
        log.error("Webhook payload parse error: %s", exc)
        raise HTTPException(status_code=400, detail="Invalid payload.") from exc

    wf = payload.workflow_run or {}
    cr = payload.check_run or {}
    conclusion = wf.get("conclusion", cr.get("conclusion", ""))
    action = payload.action

    # ── Filter: only process actionable failures ──────────────────────
    if action != "completed" or conclusion not in ("failure", "timed_out"):
        log.info("Webhook ignored (action=%s, conclusion=%s).", action, conclusion)
        return JSONResponse(
            content={"status": "ignored", "action": action, "conclusion": conclusion},
            status_code=200,
        )

    log.info(
        "Reasoning: Processing GitHub webhook failure (conclusion=%s).",
        conclusion,
    )

    # Capture raw payload for audit trail
    raw = payload.model_dump()
    features = _map_github_to_features(payload)

    # ── Try to call the MCP server inference function inline ──────────
    prediction = "pending"
    confidence = 0.0
    top_features: list = []
    result: dict

    if _INFERENCE_READY:
        try:
            res = run_prediction(
                features, _model, _le, _scaler, _hasher, _tfidf, _meta, _feature_names
            )
            prediction = res["prediction"]
            confidence = res["confidence"]
            top_features = res["top_features"]
            result = {"status": "success", "prediction": prediction}
        except Exception as exc:
            log.error("Inference execution failed: %s", exc)
            result = {"status": "error", "message": str(exc)}
    else:
        # Artifacts not loaded — store payload for later
        result = {
            "status": "queued",
            "message": "Payload stored for next inference cycle (artifacts not loaded).",
        }

    # ── Always persist to DB with event_source tag ────────────────────
    def _save_webhook():
        with get_session() as session:
            entry = LogEntry(
                event_source="github_webhook",
                metrics_payload=features,
                raw_payload=raw,
                prediction=prediction,
                confidence=confidence,
                top_features=top_features,
            )
            session.add(entry)

    try:
        await run_in_threadpool(_save_webhook)
        log.info(
            "Webhook entry persisted (conclusion=%s, prediction=%s).",
            conclusion,
            prediction,
        )
    except Exception as db_exc:
        log.error("Webhook DB write failed: %s", db_exc)
        result = {"error": "Failed to persist webhook", "details": str(db_exc)}

    return JSONResponse(content=result, status_code=200)


@app.get("/", response_class=HTMLResponse)
async def dashboard_ui() -> str:
    """Serve the observability dashboard as an HTML page."""
    log.info("Reasoning: Rendering dashboard UI.")
    data: DashboardData = await get_dashboard_data()

    # Build heatmap rows
    heatmap_rows = ""
    for entry in data.drift_heatmap:
        bg = {"none": "#1a1a2e", "moderate": "#44403c", "severe": "#7f1d1d"}
        border = {"none": "#334155", "moderate": "#d97706", "severe": "#dc2626"}
        heatmap_rows += f"""
        <tr style="background:{bg.get(entry.severity, '#1a1a2e')}; border-left: 3px solid {border.get(entry.severity, '#334155')};">
            <td style="padding:10px 14px; font-weight:500;">{entry.feature}</td>
            <td style="padding:10px 14px;">{entry.method}</td>
            <td style="padding:10px 14px; font-family:monospace; font-weight:bold;">{entry.score:.4f}</td>
            <td style="padding:10px 14px;">
                <span style="padding:3px 10px; border-radius:12px; font-size:12px; font-weight:600;
                    background:{'#dc2626' if entry.severity == 'severe' else '#d97706' if entry.severity == 'moderate' else '#22c55e'};
                    color:white;">
                    {entry.severity.upper()}
                </span>
            </td>
        </tr>"""

    if not heatmap_rows:
        heatmap_rows = '<tr><td colspan="4" style="padding:20px; text-align:center; color:#94a3b8;">No drift data. Run <code>python3 models/drift_monitor.py</code></td></tr>'

    # Build inference history rows from database
    history_rows = ""
    try:
        with get_session() as session:
            recent = (
                session.query(LogEntry)
                .order_by(LogEntry.timestamp.desc())
                .limit(20)
                .all()
            )
            for row in recent:
                ts = (
                    row.timestamp.strftime("%Y-%m-%d %H:%M:%S")
                    if row.timestamp
                    else "—"
                )
                conf_color = (
                    "#22c55e"
                    if row.confidence > 0.7
                    else "#eab308" if row.confidence > 0.4 else "#ef4444"
                )
                history_rows += f"""
        <tr style="background:#1a1a2e; border-left: 3px solid #6366f1;">
            <td style="padding:10px 14px; font-family:monospace; font-size:13px;">{ts}</td>
            <td style="padding:10px 14px; font-weight:500;">{row.prediction}</td>
            <td style="padding:10px 14px; font-family:monospace; font-weight:bold; color:{conf_color};">{row.confidence:.4f}</td>
        </tr>"""
    except Exception:
        pass

    if not history_rows:
        history_rows = '<tr><td colspan="3" style="padding:20px; text-align:center; color:#94a3b8;">No inferences recorded yet.</td></tr>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentinel-AIOps Dashboard</title>
    <meta name="description" content="Real-time observability dashboard for CI/CD anomaly detection.">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
            color: #e2e8f0;
            min-height: 100vh;
            padding: 24px;
        }}
        .header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 32px;
            padding-bottom: 16px;
            border-bottom: 1px solid #334155;
        }}
        .header h1 {{
            font-size: 28px;
            font-weight: 700;
            background: linear-gradient(90deg, #6366f1, #8b5cf6, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .badge {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 20px;
            border-radius: 24px;
            font-weight: 600;
            font-size: 15px;
            background: {data.health.color}22;
            border: 2px solid {data.health.color};
            color: {data.health.color};
        }}
        .cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 32px;
        }}
        .card {{
            background: #1e1e30;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 20px;
        }}
        .card .label {{ font-size: 12px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }}
        .card .value {{ font-size: 28px; font-weight: 700; margin-top: 4px; }}
        .section-title {{
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 16px;
            margin-top: 32px;
            padding-left: 12px;
            border-left: 3px solid #6366f1;
        }}
        table {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0 4px;
        }}
        th {{
            text-align: left;
            padding: 10px 14px;
            font-size: 12px;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        td {{ border: none; }}
        .footer {{
            margin-top: 40px;
            text-align: center;
            font-size: 12px;
            color: #64748b;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🛡️ Sentinel-AIOps</h1>
        <div class="badge">{data.health.badge} {data.health.label}</div>
    </div>

    <div class="cards">
        <div class="card">
            <div class="label">Model Version</div>
            <div class="value" style="color:#a78bfa;">{data.model_version or 'N/A'}</div>
        </div>
        <div class="card">
            <div class="label">Features Drifted</div>
            <div class="value" style="color:{data.health.color};">{data.health.features_drifted} / {data.health.total_features}</div>
        </div>
        <div class="card">
            <div class="label">Feedback Records</div>
            <div class="value" style="color:#38bdf8;">{data.feedback_count}</div>
        </div>
        <div class="card">
            <div class="label">Total Inferences</div>
            <div class="value" style="color:#a78bfa;">{data.inference_count}</div>
        </div>
        <div class="card">
            <div class="label">Retrain Suggested</div>
            <div class="value" style="color:{'#ef4444' if data.health.retrain_suggested else '#22c55e'};">
                {'Yes ⚠️' if data.health.retrain_suggested else 'No ✅'}
            </div>
        </div>
    </div>

    <div class="section-title">Feature Drift Heatmap</div>
    <table>
        <thead>
            <tr>
                <th>Feature</th>
                <th>Method</th>
                <th>Score</th>
                <th>Severity</th>
            </tr>
        </thead>
        <tbody>
            {heatmap_rows}
        </tbody>
    </table>

    <div class="section-title">Inference History (Last 20)</div>
    <table>
        <thead>
            <tr>
                <th>Timestamp</th>
                <th>Prediction</th>
                <th>Confidence</th>
            </tr>
        </thead>
        <tbody>
            {history_rows}
        </tbody>
    </table>

    <div class="footer">
        Sentinel-AIOps v2.0 &bull; Last drift check: {data.health.last_drift_check or 'Never'}
    </div>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn

    log.info("Starting Sentinel-AIOps Dashboard on http://0.0.0.0:8200")
    uvicorn.run(app, host="0.0.0.0", port=8200)
