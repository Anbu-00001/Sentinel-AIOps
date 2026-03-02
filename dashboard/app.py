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

import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Optional

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import LogEntry, get_session, init_db  # noqa: E402

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("sentinel.dashboard")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
FEEDBACK_FILE = os.path.join(ROOT, "data", "feedback", "human_labels.json")

# ── Initialize database ──────────────────────────────────────────────────────
log.info("Reasoning: Initializing SQLite database for dashboard queries.")
init_db()

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

app = FastAPI(
    title="Sentinel-AIOps Dashboard",
    description="Observability dashboard for CI/CD anomaly detection platform.",
    version="2.0.0",
)


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
    Derive a live drift signal from the last 100 inference records.

    Returns a dict with keys:
      - db_mean_confidence: mean confidence of last 100 inferences
      - db_retrain_suggested: True if mean confidence < 0.45
      - db_sample_size: number of rows used
    Returns None if DB has no data.
    """
    try:
        with get_session() as session:
            rows = (
                session.query(LogEntry.confidence)
                .order_by(LogEntry.timestamp.desc())
                .limit(100)
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

    retrain_json: bool = drift_report.get("retrain_suggested", False) if drift_report else False
    retrain_db: bool = db_signal.get("db_retrain_suggested", False) if db_signal else False
    retrain: bool = retrain_json or retrain_db

    if retrain:
        source = "JSON report" if retrain_json else f"DB signal (mean_conf={db_signal['db_mean_confidence']:.3f})"
        log.info("Training Required triggered by: %s", source)
        return HealthStatus(
            badge="🔴", label="Training Required", color="#ef4444",
            retrain_suggested=True, features_drifted=n_drifted,
            total_features=total, last_drift_check=last_check,
        )
    elif n_drifted > 0:
        return HealthStatus(
            badge="🟡", label="Drift Detected", color="#eab308",
            retrain_suggested=False, features_drifted=n_drifted,
            total_features=total, last_drift_check=last_check,
        )
    else:
        return HealthStatus(
            badge="🟢", label="Healthy", color="#22c55e",
            retrain_suggested=False, features_drifted=n_drifted,
            total_features=total, last_drift_check=last_check,
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


def _map_github_to_features(payload: GitHubWebhookPayload) -> dict[str, Any]:
    """
    Map a GitHub webhook payload into the analyze_log feature schema.

    Uses workflow_run or check_run data to synthesize CI/CD metrics.
    """
    import random

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
            ended = datetime.fromisoformat(
                wf["updated_at"].replace("Z", "+00:00")
            )
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
        "cpu_usage_pct": round(random.uniform(20.0, 80.0), 1),
        "memory_usage_mb": random.randint(512, 8192),
        "retry_count": wf.get("run_attempt", 1) - 1,
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


@app.get("/api/dashboard")
async def get_dashboard_data() -> DashboardData:
    """Return complete dashboard payload as JSON."""
    log.info("Reasoning: Building dashboard data payload.")
    drift_report = _load_drift_report()
    registry = _load_registry()

    db_signal = _compute_health_from_db()
    return DashboardData(
        health=_compute_health(drift_report, registry, db_signal),
        drift_heatmap=_build_heatmap(drift_report),
        model_version=registry.get("latest") if registry else None,
        feedback_count=_load_feedback_count(),
        inference_count=_get_inference_count(),
    )


@app.get("/api/drift")
async def get_drift_report() -> dict[str, Any]:
    """Return raw drift report."""
    report = _load_drift_report()
    if report is None:
        return {"message": "No drift report available. Run drift_monitor.py first."}
    return report


@app.get("/api/registry")
async def get_registry() -> dict[str, Any]:
    """Return model registry."""
    registry = _load_registry()
    if registry is None:
        return {"message": "No registry found."}
    return registry


@app.get("/api/history")
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
async def github_webhook(payload: GitHubWebhookPayload) -> JSONResponse:
    """
    Receive a GitHub Actions webhook (workflow_run events).

    Only processes events where action=="completed" AND
    conclusion is "failure" or "timed_out". All other events
    return a 200 {"status": "ignored"} response.

    Configure in GitHub → Settings → Webhooks:
      Payload URL:  http://<your-host>:8200/webhook/github
      Content type: application/json
      Events:       Workflow runs
    """
    wf = payload.workflow_run or {}
    cr = payload.check_run or {}
    conclusion = wf.get("conclusion", cr.get("conclusion", ""))
    action = payload.action

    # ── Filter: only process actionable failures ──────────────────────
    if action != "completed" or conclusion not in ("failure", "timed_out"):
        log.info(
            "Webhook ignored (action=%s, conclusion=%s).", action, conclusion
        )
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

    try:
        from importlib import import_module
        server_mod = import_module("mcp_server.server")
        result = server_mod.analyze_log(features)
        prediction = result.get("prediction", "pending")
        confidence = result.get("confidence", 0.0)
        top_features = result.get("top_contributing_features", [])
    except Exception:
        # MCP server runs in a separate process — store payload for later
        result = {
            "status": "queued",
            "message": "Payload stored for next inference cycle.",
        }

    # ── Always persist to DB with event_source tag ────────────────────
    try:
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
        log.info(
            "Webhook entry persisted (conclusion=%s, prediction=%s).",
            conclusion, prediction,
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
                ts = row.timestamp.strftime("%Y-%m-%d %H:%M:%S") if row.timestamp else "—"
                conf_color = "#22c55e" if row.confidence > 0.7 else "#eab308" if row.confidence > 0.4 else "#ef4444"
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
