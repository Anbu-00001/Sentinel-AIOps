"""
stream_simulator.py — Sentinel-AIOps Event-Driven Ingestion
=============================================================
Produces GitHub-style JSON CI/CD logs via:
  1. A Python generator for internal consumption
  2. A FastAPI endpoint (GET /stream, GET /stream/chaos)
  3. Chaos Mode: injects out-of-distribution data (5x resource spikes)

Workflow Rules (AGENTS.md):
  - Log 'Reasoning' before execution.
  - Type hinting + Pydantic validation throughout.
"""

import asyncio
import logging
import random
import string
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import AsyncGenerator, Generator

from fastapi import FastAPI, Query
from pydantic import BaseModel, Field
import joblib
import os
import sys
import numpy as np

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("sentinel.stream")

# ── Constants (based on training distribution from data_summary.json) ─────────
CI_TOOLS: list[str] = [
    "Jenkins",
    "GitHub Actions",
    "GitLab CI",
    "CircleCI",
    "Travis CI",
]
REPOS: list[str] = [f"repo-{i}" for i in range(50)]  # subset of 500
AUTHORS: list[str] = [f"dev-{i}" for i in range(100)]  # subset of 1000
LANGUAGES: list[str] = ["Python", "Java", "JavaScript", "Go", "Rust", "C++"]
OS_LIST: list[str] = ["Linux", "Windows", "macOS"]
CLOUD_PROVIDERS: list[str] = ["AWS", "GCP", "Azure", "On-Premise"]
BRANCHES: list[str] = ["main", "develop", "feature", "hotfix"]
FAILURE_STAGES: list[str] = ["build", "test", "deploy"]
FAILURE_TYPES: list[str] = [
    "Build Failure",
    "Configuration Error",
    "Dependency Error",
    "Deployment Failure",
    "Network Error",
    "Permission Error",
    "Resource Exhaustion",
    "Security Scan Failure",
    "Test Failure",
    "Timeout",
]
SEVERITIES: list[str] = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

# ── Dynamic Normalization Lookup ──────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")

try:
    log.info("Reasoning: Synchronizing simulator ranges with training artifacts.")
    _scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    # Assuming column order: build, test, deploy, cpu, mem, retry
    _means = _scaler.mean_
    _stds = np.sqrt(_scaler.var_) if hasattr(_scaler, "var_") else [10.0] * 6

    NORMAL_RANGES: dict[str, tuple[float, float]] = {
        "build_duration_sec": (
            max(10, _means[0] - 2 * _stds[0]),
            _means[0] + 2 * _stds[0],
        ),
        "test_duration_sec": (
            max(5, _means[1] - 2 * _stds[1]),
            _means[1] + 2 * _stds[1],
        ),
        "deploy_duration_sec": (
            max(5, _means[2] - 2 * _stds[2]),
            _means[2] + 2 * _stds[2],
        ),
        "cpu_usage_pct": (
            max(0, _means[3] - 2 * _stds[3]),
            min(100, _means[3] + 2 * _stds[3]),
        ),
        "memory_usage_mb": (
            max(256, _means[4] - 2 * _stds[4]),
            _means[4] + 2 * _stds[4],
        ),
        "retry_count": (max(0, _means[5] - 2 * _stds[5]), _means[5] + 2 * _stds[5]),
    }
    log.info("Simulator synchronized with scaler: %s", NORMAL_RANGES)
except Exception as exc:
    log.warning("Failed to load scaler for simulation: %s. Using fallback ranges.", exc)
    NORMAL_RANGES: dict[str, tuple[float, float]] = {
        "build_duration_sec": (10, 3600),
        "test_duration_sec": (5, 600),
        "deploy_duration_sec": (5, 300),
        "cpu_usage_pct": (5.0, 100.0),
        "memory_usage_mb": (256, 16384),
        "retry_count": (0, 5),
    }

# Chaos multipliers (5x above training max)
CHAOS_MULTIPLIER: float = 5.0
import numpy as np


class ChaosLevel(str, Enum):
    """Chaos engineering injection levels."""

    NONE = "none"
    LOW = "low"  # 10% chance of OOD injection
    MEDIUM = "medium"  # 30% chance
    HIGH = "high"  # 60% chance
    EXTREME = "extreme"  # 100% — every log is OOD


class LogRecord(BaseModel):
    """Pydantic model for a CI/CD log event."""

    log_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    pipeline_id: str = Field(default_factory=lambda: f"pipe-{uuid.uuid4().hex[:8]}")
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    ci_tool: str
    repository: str
    branch: str
    commit_hash: str
    author: str
    language: str
    os: str
    cloud_provider: str
    build_duration_sec: int
    test_duration_sec: int
    deploy_duration_sec: int
    failure_stage: str
    failure_type: str
    error_code: str
    error_message: str
    severity: str
    cpu_usage_pct: float
    memory_usage_mb: int
    retry_count: int
    is_flaky_test: bool
    rollback_triggered: bool
    incident_created: bool
    is_chaos: bool = Field(
        default=False, description="True if this record was injected by chaos mode"
    )


def _random_string(length: int = 40) -> str:
    """Generate a random alphanumeric string (commit hash analog)."""
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


def _random_error_message() -> str:
    """Generate a realistic error message."""
    templates: list[str] = [
        "ERROR: Build failed at step {step} — {reason}",
        "FATAL: {service} unreachable on port {port} after {n} retries",
        "TIMEOUT: Job exceeded {limit}s wall-clock limit in {stage} stage",
        "PERMISSION DENIED: Cannot access {resource} — check IAM policy",
        "OOM: Container killed — memory usage {mem}MB exceeded limit",
        "DEPENDENCY: Failed to resolve {pkg}@{ver} from registry",
        "SECURITY: CVE-{cve} detected in {pkg} — blocking deployment",
        "CONFIG: Missing environment variable {var} in {env} namespace",
    ]
    template = random.choice(templates)
    return template.format(
        step=random.randint(1, 15),
        reason=random.choice(["compilation error", "missing artifact", "lint failure"]),
        service=random.choice(["postgres", "redis", "api-gateway", "auth-service"]),
        port=random.choice([443, 5432, 6379, 8080, 9090]),
        n=random.randint(3, 10),
        limit=random.choice([300, 600, 900, 1800]),
        stage=random.choice(FAILURE_STAGES),
        resource=random.choice(["/secrets/prod", "/deploy/k8s", "/artifacts/build"]),
        mem=random.randint(8000, 32000),
        pkg=random.choice(["numpy", "lodash", "spring-boot", "tokio"]),
        ver=f"{random.randint(1, 5)}.{random.randint(0, 20)}.{random.randint(0, 9)}",
        cve=f"2026-{random.randint(10000, 99999)}",
        var=random.choice(["DATABASE_URL", "API_KEY", "SECRET_TOKEN"]),
        env=random.choice(["production", "staging", "ci"]),
    )


def _generate_log(chaos: ChaosLevel = ChaosLevel.NONE) -> LogRecord:
    """Generate a single CI/CD log record, optionally with chaos injection."""
    chaos_probs: dict[ChaosLevel, float] = {
        ChaosLevel.NONE: 0.0,
        ChaosLevel.LOW: 0.10,
        ChaosLevel.MEDIUM: 0.30,
        ChaosLevel.HIGH: 0.60,
        ChaosLevel.EXTREME: 1.0,
    }

    is_chaos: bool = random.random() < chaos_probs.get(chaos, 0.0)

    if is_chaos:
        # OOD injection: 5x above normal ranges
        build_dur = int(
            NORMAL_RANGES["build_duration_sec"][1]
            * CHAOS_MULTIPLIER
            * random.uniform(1.0, 2.0)
        )
        test_dur = int(
            NORMAL_RANGES["test_duration_sec"][1]
            * CHAOS_MULTIPLIER
            * random.uniform(1.0, 2.0)
        )
        deploy_dur = int(
            NORMAL_RANGES["deploy_duration_sec"][1]
            * CHAOS_MULTIPLIER
            * random.uniform(1.0, 2.0)
        )
        cpu = min(
            100.0,
            NORMAL_RANGES["cpu_usage_pct"][1]
            * CHAOS_MULTIPLIER
            * random.uniform(0.8, 1.0),
        )
        mem = int(
            NORMAL_RANGES["memory_usage_mb"][1]
            * CHAOS_MULTIPLIER
            * random.uniform(1.0, 2.0)
        )
        retry = int(
            NORMAL_RANGES["retry_count"][1]
            * CHAOS_MULTIPLIER
            * random.uniform(1.0, 3.0)
        )
    else:
        build_dur = random.randint(
            *[int(x) for x in NORMAL_RANGES["build_duration_sec"]]
        )
        test_dur = random.randint(*[int(x) for x in NORMAL_RANGES["test_duration_sec"]])
        deploy_dur = random.randint(
            *[int(x) for x in NORMAL_RANGES["deploy_duration_sec"]]
        )
        cpu = round(random.uniform(*NORMAL_RANGES["cpu_usage_pct"]), 1)
        mem = random.randint(*[int(x) for x in NORMAL_RANGES["memory_usage_mb"]])
        retry = random.randint(*[int(x) for x in NORMAL_RANGES["retry_count"]])

    return LogRecord(
        ci_tool=random.choice(CI_TOOLS),
        repository=random.choice(REPOS),
        branch=random.choice(BRANCHES),
        commit_hash=_random_string(40),
        author=random.choice(AUTHORS),
        language=random.choice(LANGUAGES),
        os=random.choice(OS_LIST),
        cloud_provider=random.choice(CLOUD_PROVIDERS),
        build_duration_sec=build_dur,
        test_duration_sec=test_dur,
        deploy_duration_sec=deploy_dur,
        failure_stage=random.choice(FAILURE_STAGES),
        failure_type=random.choice(FAILURE_TYPES),
        error_code=f"ERR_{random.randint(100, 999)}",
        error_message=_random_error_message(),
        severity=random.choice(SEVERITIES),
        cpu_usage_pct=cpu,
        memory_usage_mb=mem,
        retry_count=retry,
        is_flaky_test=random.choice([True, False]),
        rollback_triggered=random.choice([True, False]),
        incident_created=random.choice([True, False]),
        is_chaos=is_chaos,
    )


def log_generator(
    count: int = 100,
    chaos: ChaosLevel = ChaosLevel.NONE,
) -> Generator[LogRecord, None, None]:
    """Synchronous generator producing CI/CD log records."""
    log.info(
        "Reasoning: Starting log generator (count=%d, chaos=%s).", count, chaos.value
    )
    for _ in range(count):
        yield _generate_log(chaos)


async def async_log_generator(
    count: int = 100,
    chaos: ChaosLevel = ChaosLevel.NONE,
    delay_ms: int = 100,
) -> AsyncGenerator[LogRecord, None]:
    """Async generator simulating real-time log arrival with configurable delay."""
    log.info(
        "Reasoning: Starting async log generator (count=%d, chaos=%s, delay=%dms).",
        count,
        chaos.value,
        delay_ms,
    )
    for _ in range(count):
        yield _generate_log(chaos)
        await asyncio.sleep(delay_ms / 1000.0)


# ── FastAPI Application ──────────────────────────────────────────────────────
app = FastAPI(
    title="Sentinel-AIOps Stream Simulator",
    description="Event-driven CI/CD log generator with Chaos Engineering support.",
    version="1.0.0",
)


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "stream-simulator"}


@app.get("/stream")
async def stream_logs(
    count: int = Query(
        default=10, ge=1, le=1000, description="Number of logs to generate"
    ),
    chaos: ChaosLevel = Query(default=ChaosLevel.NONE, description="Chaos level"),
) -> list[dict]:
    """Generate a batch of CI/CD log records."""
    log.info("Reasoning: API /stream called (count=%d, chaos=%s).", count, chaos.value)
    return [record.model_dump() for record in log_generator(count, chaos)]


@app.get("/stream/single")
async def stream_single(
    chaos: ChaosLevel = Query(default=ChaosLevel.NONE, description="Chaos level"),
) -> dict:
    """Generate a single CI/CD log record."""
    return _generate_log(chaos).model_dump()


if __name__ == "__main__":
    import uvicorn

    log.info("Starting Stream Simulator on http://0.0.0.0:8100")
    uvicorn.run(app, host="0.0.0.0", port=8100)
