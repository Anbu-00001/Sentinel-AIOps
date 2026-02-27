"""
app.py — Sentinel-AIOps Observability Dashboard
=================================================
FastAPI-based dashboard providing:
  - System Health Badge (🟢 Healthy / 🟡 Drift Detected / 🔴 Training Required)
  - Drift Heatmap showing per-feature drift scores
  - Model registry info and feedback stats

Workflow Rules (AGENTS.md):
  - Log 'Reasoning' before execution.
  - Type hinting + Pydantic validation throughout.
"""

import json
import logging
import os
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("sentinel.dashboard")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
FEEDBACK_FILE = os.path.join(ROOT, "data", "feedback", "human_labels.json")

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


class DashboardData(BaseModel):
    """Complete dashboard payload."""
    health: HealthStatus
    drift_heatmap: list[DriftHeatmapEntry]
    model_version: Optional[str] = None
    feedback_count: int = 0


# ── FastAPI Application ──────────────────────────────────────────────────────

app = FastAPI(
    title="Sentinel-AIOps Dashboard",
    description="Observability dashboard for CI/CD anomaly detection platform.",
    version="1.0.0",
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


def _compute_health(drift_report: Optional[dict], registry: Optional[dict]) -> HealthStatus:
    """Determine system health based on drift report and registry."""
    if drift_report is None:
        return HealthStatus(
            badge="🟢",
            label="Healthy",
            color="#22c55e",
            retrain_suggested=False,
            features_drifted=0,
            total_features=0,
        )

    n_drifted: int = drift_report.get("features_drifted", 0)
    retrain: bool = drift_report.get("retrain_suggested", False)
    total: int = drift_report.get("features_analyzed", 0)
    last_check: Optional[str] = registry.get("last_drift_check") if registry else None

    if retrain:
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

    return DashboardData(
        health=_compute_health(drift_report, registry),
        drift_heatmap=_build_heatmap(drift_report),
        model_version=registry.get("latest") if registry else None,
        feedback_count=_load_feedback_count(),
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
                    background:{'#dc2626' if entry.severity=='severe' else '#d97706' if entry.severity=='moderate' else '#22c55e'};
                    color:white;">
                    {entry.severity.upper()}
                </span>
            </td>
        </tr>"""

    if not heatmap_rows:
        heatmap_rows = '<tr><td colspan="4" style="padding:20px; text-align:center; color:#94a3b8;">No drift data. Run <code>python3 models/drift_monitor.py</code></td></tr>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentinel-AIOps Dashboard</title>
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
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
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

    <div class="footer">
        Sentinel-AIOps &bull; Last drift check: {data.health.last_drift_check or 'Never'}
    </div>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    log.info("Starting Sentinel-AIOps Dashboard on http://0.0.0.0:8200")
    uvicorn.run(app, host="0.0.0.0", port=8200)
