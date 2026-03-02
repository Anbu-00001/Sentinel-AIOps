"""
server.py — Sentinel-AIOps FastMCP Inference Server (v3 Enterprise)
=====================================================================
Production-grade server with:
  - `analyze_log` tool: LightGBM multiclass prediction
  - Prometheus /metrics endpoint (model_drift_score, inference_latency,
    total_anomalies_detected)
  - Error handling: graceful schema validation
  - Structured JSON output with type hints

Workflow Rules (AGENTS.md):
  - Log 'Reasoning' before execution.
  - Serve all inference via local-first FastMCP.
"""

import json
import logging
import os
import sys
import time
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from scipy.sparse import csr_matrix, hstack

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import LogEntry, get_session, init_db  # noqa: E402

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("sentinel.mcp_server")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")

# ── Prometheus metrics ────────────────────────────────────────────────────────
INFERENCE_LATENCY = Histogram(
    "inference_latency_seconds",
    "Time spent on a single analyze_log inference",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)
TOTAL_ANOMALIES = Counter(
    "total_anomalies_detected",
    "Cumulative count of logs classified with confidence > 0.5",
)
TOTAL_INFERENCES = Counter(
    "total_inferences",
    "Cumulative count of all inference requests",
)
MODEL_DRIFT_SCORE = Gauge(
    "model_drift_score",
    "Latest max PSI drift score from drift_report.json",
)
INFERENCE_ERRORS = Counter(
    "inference_errors_total",
    "Cumulative count of failed inference requests",
)


def _refresh_drift_metric() -> None:
    """Read drift_report.json and update the Prometheus gauge."""
    report_path = os.path.join(MODELS_DIR, "drift_report.json")
    if os.path.exists(report_path):
        try:
            with open(report_path) as f:
                report = json.load(f)
            psi_scores = [
                r["score"]
                for r in report.get("feature_results", [])
                if r.get("method") == "PSI"
            ]
            if psi_scores:
                MODEL_DRIFT_SCORE.set(max(psi_scores))
        except Exception:
            pass


# ── Schema — required and optional feature keys ──────────────────────────────
REQUIRED_KEYS: list[str] = [
    "build_duration_sec",
    "test_duration_sec",
    "deploy_duration_sec",
    "cpu_usage_pct",
    "memory_usage_mb",
    "retry_count",
    "error_message",
]

OPTIONAL_KEYS: dict[str, Any] = {
    "repository": "",
    "author": "",
    "failure_stage": "build",
    "severity": "MEDIUM",
    "ci_tool": "Jenkins",
    "language": "Python",
    "os": "Linux",
    "cloud_provider": "AWS",
    "is_flaky_test": False,
    "rollback_triggered": False,
    "incident_created": False,
}

# ── Load artifacts at startup ─────────────────────────────────────────────────
log.info(
    "Reasoning: Loading v3 model artifacts from %s at server startup.", MODELS_DIR
)

try:
    _model = joblib.load(os.path.join(MODELS_DIR, "lgbm_model.joblib"))
    _le = joblib.load(os.path.join(MODELS_DIR, "label_encoder.joblib"))
    _scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    _hasher = joblib.load(os.path.join(MODELS_DIR, "hasher.joblib"))
    _tfidf = joblib.load(os.path.join(MODELS_DIR, "tfidf.joblib"))
    with open(os.path.join(MODELS_DIR, "feature_meta.json")) as f:
        _meta = json.load(f)
    log.info("All v3 artifacts loaded successfully.")
except FileNotFoundError as exc:
    log.error(
        "CRITICAL: Missing model artifact — %s. Run train_v2.py first.", exc
    )
    raise SystemExit(1) from exc

# ── Initialize database ──────────────────────────────────────────────────────
log.info("Reasoning: Initializing SQLite database for inference persistence.")
init_db()

# Build readable feature names
_feature_names: list[str] = []
_feature_names.extend(_meta["numerical_cols"])
_feature_names.extend([f"hash_{i}" for i in range(_meta["hash_n_features"])])
_feature_names.extend(
    [f"tfidf_{i}" for i in range(_meta["tfidf_max_features"])]
)
_feature_names.extend([str(c) for c in _meta["dummy_column_order"]])
_feature_names.extend(_meta["bool_cols"])

# Seed drift metric on startup
_refresh_drift_metric()

# ── FastMCP application ───────────────────────────────────────────────────────
mcp = FastMCP(
    name="Sentinel-AIOps-v3",
    instructions=(
        "You are the Sentinel-AIOps v3 enterprise inference server. "
        "Use 'analyze_log' to classify CI/CD log failures. "
        "Prometheus metrics available at /metrics."
    ),
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _validate_input(features: Dict[str, Any]) -> list[str]:
    """Validate required keys are present."""
    errors: list[str] = []
    for key in REQUIRED_KEYS:
        if key not in features:
            errors.append(f"Missing required field: '{key}'")
    for key in [
        "build_duration_sec", "test_duration_sec",
        "deploy_duration_sec", "cpu_usage_pct",
        "memory_usage_mb", "retry_count",
    ]:
        if key in features:
            try:
                float(features[key])
            except (ValueError, TypeError):
                errors.append(
                    f"Field '{key}' must be numeric, "
                    f"got: {type(features[key]).__name__}"
                )
    return errors


def _transform_input(features: Dict[str, Any]):
    """Apply the 4-group transformation pipeline."""
    for key, default in OPTIONAL_KEYS.items():
        if key not in features:
            features[key] = default

    df = pd.DataFrame([features])

    X_num = csr_matrix(
        _scaler.transform(df[_meta["numerical_cols"]].astype(float))
    )
    hash_input = (
        df[_meta["high_card_cols"]].astype(str).to_dict(orient="records")
    )
    X_hash = _hasher.transform(hash_input)
    X_text = _tfidf.transform(
        df[_meta["text_col"]].fillna("").astype(str)
    )
    df_dummies = pd.get_dummies(df[_meta["low_card_cols"]], drop_first=True)
    df_dummies = df_dummies.reindex(
        columns=_meta["dummy_column_order"], fill_value=0
    )
    bool_data = df[_meta["bool_cols"]].astype(int)
    X_extra = csr_matrix(
        pd.concat([df_dummies, bool_data], axis=1).astype(float).values
    )

    return hstack([X_num, X_hash, X_text, X_extra], format="csr")


def _get_top_features(X_row, n: int = 5) -> list[dict[str, Any]]:
    """Return top-N contributing feature names by absolute value."""
    if hasattr(X_row, "toarray"):
        arr = np.abs(X_row.toarray().flatten())
    else:
        arr = np.abs(X_row.flatten())
    top_idx = np.argsort(arr)[-n:][::-1]
    results: list[dict[str, Any]] = []
    for idx in top_idx:
        name = (
            _feature_names[idx]
            if idx < len(_feature_names)
            else f"f_{idx}"
        )
        results.append(
            {"feature": name, "value": round(float(arr[idx]), 4)}
        )
    return results


def get_prometheus_metrics() -> str:
    """Return Prometheus text exposition format."""
    _refresh_drift_metric()
    return generate_latest().decode("utf-8")


# ── MCP Tool ──────────────────────────────────────────────────────────────────

@mcp.tool()
def analyze_log(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Classify a CI/CD log record into one of 10 failure types.

    Required fields:
      - build_duration_sec, test_duration_sec, deploy_duration_sec
      - cpu_usage_pct, memory_usage_mb, retry_count
      - error_message (string)

    Returns structured JSON with prediction, confidence, and
    top contributing features. Tracks Prometheus metrics.
    """
    log.info("Reasoning: Received analyze_log request.")
    TOTAL_INFERENCES.inc()
    start_time = time.perf_counter()

    # ── Validate ──────────────────────────────────────────────────────────
    errors = _validate_input(features)
    if errors:
        INFERENCE_ERRORS.inc()
        log.warning("Schema validation failed: %s", errors)
        return {"error": "Schema validation failed", "details": errors}

    # ── Transform ─────────────────────────────────────────────────────────
    try:
        X = _transform_input(features)
    except Exception as exc:
        INFERENCE_ERRORS.inc()
        log.error("Transform failed: %s", exc)
        return {"error": "Feature transformation failed", "details": [str(exc)]}

    # ── Predict ───────────────────────────────────────────────────────────
    proba = _model.predict_proba(X)[0]
    pred_idx = int(np.argmax(proba))
    confidence = float(proba[pred_idx])
    prediction: str = _le.inverse_transform([pred_idx])[0]

    top_features = _get_top_features(X, n=5)

    # ── Prometheus tracking ───────────────────────────────────────────────
    latency = time.perf_counter() - start_time
    INFERENCE_LATENCY.observe(latency)
    if confidence > 0.5:
        TOTAL_ANOMALIES.inc()

    log.info(
        "Result → prediction=%s | confidence=%.4f | latency=%.4fs",
        prediction, confidence, latency,
    )

    # ── Persist to database ──────────────────────────────────────────────
    try:
        with get_session() as session:
            entry = LogEntry(
                metrics_payload=features,
                prediction=prediction,
                confidence=confidence,
                top_features=top_features,
            )
            session.add(entry)
        log.info("Inference persisted to database (prediction=%s).", prediction)
    except Exception as db_exc:
        log.warning("DB write failed (non-fatal): %s", db_exc)

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "top_contributing_features": top_features,
    }


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Dual mode: MCP on stdio + optional HTTP metrics endpoint
    import threading
    from http.server import HTTPServer, BaseHTTPRequestHandler

    class MetricsHandler(BaseHTTPRequestHandler):
        """Minimal HTTP handler for /metrics."""
        def do_GET(self) -> None:
            if self.path == "/metrics":
                body = get_prometheus_metrics().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", CONTENT_TYPE_LATEST)
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args) -> None:
            pass  # suppress default logging

    def _run_metrics_server() -> None:
        server = HTTPServer(("0.0.0.0", 9090), MetricsHandler)
        log.info("Prometheus /metrics endpoint on http://0.0.0.0:9090/metrics")
        server.serve_forever()

    metrics_thread = threading.Thread(target=_run_metrics_server, daemon=True)
    metrics_thread.start()

    log.info("Starting Sentinel-AIOps v3 MCP server (stdio transport).")
    mcp.run(transport="stdio")
