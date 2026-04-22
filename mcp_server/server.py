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

import os
import sys

# ── Paths & Sys Path Initialization ─────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mcp_server.logic import validate_input, run_prediction
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from mcp.server.fastmcp import FastMCP
import joblib
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

# Global thread pool for offloading database I/O
_db_pool = ThreadPoolExecutor(max_workers=4)

from database import LogEntry, get_session, init_db  # noqa: E402

# ── Logging ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("sentinel.mcp_server")

# ── Prometheus metrics ──────────────────────────────────────────────────
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


# ── Removed redundant local schema definitions — now in logic.py ─────────────


# ── Load artifacts at startup ───────────────────────────────────────────
log.info(
    "Reasoning: Loading v3 model artifacts from %s at server startup.",
    MODELS_DIR)

from models import crypto_sig


def _secure_load(artifact_name):
    filepath = os.path.join(MODELS_DIR, artifact_name)
    if not crypto_sig.verify_artifact(filepath):
        log.error("CRITICAL: Artifact signature validation failed for %s", filepath)
        raise SystemExit(1)
    return joblib.load(filepath)


try:
    _model = _secure_load("lgbm_model.joblib")
    _le = _secure_load("label_encoder.joblib")
    _scaler = _secure_load("scaler.joblib")
    _hasher = _secure_load("hasher.joblib")
    _tfidf = _secure_load("tfidf.joblib")
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

# ── FastMCP application ─────────────────────────────────────────────────
mcp = FastMCP(
    name="Sentinel-AIOps-v3",
    instructions=(
        "You are the Sentinel-AIOps v3 enterprise inference server. "
        "Use 'analyze_log' to classify CI/CD log failures. "
        "Prometheus metrics available at /metrics."
    ),
)


# ── Removed redundant local transforms — now in logic.py ──────────────────


def get_prometheus_metrics() -> str:
    """Return Prometheus text exposition format."""
    _refresh_drift_metric()
    return generate_latest().decode("utf-8")


# ── MCP Tool ────────────────────────────────────────────────────────────

@mcp.tool()
def analyze_log(features: dict) -> dict:
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
    errors = validate_input(features)
    if errors:
        INFERENCE_ERRORS.inc()
        log.warning("Schema validation failed: %s", errors)
        return {"error": "Schema validation failed", "details": errors}

    # ── Predict ───────────────────────────────────────────────────────────
    try:
        res = run_prediction(
            features,
            _model,
            _le,
            _scaler,
            _hasher,
            _tfidf,
            _meta,
            _feature_names)
        prediction = res["prediction"]
        confidence = res["confidence"]
        top_features = res["top_features"]
    except Exception as exc:
        INFERENCE_ERRORS.inc()
        log.error("Inference failed: %s", exc)
        return {"error": "Inference execution failed", "details": [str(exc)]}

    # ── Prometheus tracking ───────────────────────────────────────────────
    latency = time.perf_counter() - start_time
    INFERENCE_LATENCY.observe(latency)
    if confidence > 0.5:
        TOTAL_ANOMALIES.inc()

    log.info(
        "Result → prediction=%s | confidence=%.4f | latency=%.4fs",
        prediction, confidence, latency,
    )

    # ── Persist to database (non-blocking with retry) ──────────────────
    def _save_inference():
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                with get_session() as session:
                    entry = LogEntry(
                        event_source="mcp",
                        metrics_payload=features,
                        raw_payload=dict(features),   # immutable audit copy
                        prediction=prediction,
                        confidence=confidence,
                        top_features=top_features,
                    )
                    session.add(entry)
                log.info(
                    "Inference persisted to database (prediction=%s).",
                    prediction)
                return  # Success — exit retry loop
            except Exception as db_exc:
                if attempt < max_retries:
                    wait = 0.5 * (2 ** (attempt - 1))  # 0.5s, 1s, 2s
                    log.warning(
                        "DB write attempt %d/%d failed: %s. Retrying in %.1fs...",
                        attempt, max_retries, db_exc, wait)
                    time.sleep(wait)
                else:
                    log.error(
                        "DB write FAILED after %d attempts: %s. Inference record lost.",
                        max_retries, db_exc)

    _db_pool.submit(_save_inference)

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "top_contributing_features": top_features,
    }


# ── Entry point ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Dual mode: MCP on stdio + optional HTTP metrics endpoint
    import atexit
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

    _server_ref = {"instance": None}  # mutable container for cross-function access

    def _run_metrics_server() -> None:
        _server_ref["instance"] = HTTPServer(("0.0.0.0", 9090), MetricsHandler)
        log.info("Prometheus /metrics endpoint on http://0.0.0.0:9090/metrics")
        _server_ref["instance"].serve_forever()

    def _shutdown_metrics_server(thread=None) -> None:
        """Gracefully shut down the metrics HTTP server on process exit."""
        srv = _server_ref.get("instance")
        if srv is not None:
            log.info("Shutting down Prometheus metrics server...")
            srv.shutdown()
            srv.server_close()
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

    metrics_thread = threading.Thread(target=_run_metrics_server, daemon=True)
    metrics_thread.start()
    atexit.register(_shutdown_metrics_server, thread=metrics_thread)

    log.info("Starting Sentinel-AIOps v3 MCP server (stdio transport).")
    mcp.run(transport="stdio")
