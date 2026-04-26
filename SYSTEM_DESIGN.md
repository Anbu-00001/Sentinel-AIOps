# 🏗️ SYSTEM_DESIGN.md — Sentinel-AIOps Architecture

## Overview

Sentinel-AIOps is an event-driven MLOps platform for CI/CD log anomaly detection. The system ingests real-time log streams, classifies failures using a LightGBM multiclass model, monitors feature drift, collects human feedback, and exposes observability metrics via Prometheus.

## End-to-End Architecture

```mermaid
flowchart TB
    subgraph Ingestion ["Ingestion: Data Ingestion"]
        SS["Stream Simulator\nFastAPI :8100"]
        CM["Chaos Mode\n5x OOD Injection"]
        SS --> CM
    end

    subgraph MLPipeline ["ML Pipeline"]
        PP["Preprocessor\nScaler + Hasher + TF-IDF"]
        LGBM["LightGBM v2\nMulticlass (10 classes)"]
        REG["Model Registry\nregistry.json"]
        PP --> LGBM --> REG
    end

    subgraph Inference ["Inference: FastMCP Server"]
        MCP["analyze_log Tool\nSchema Validation"]
        PROM["Prometheus Metrics\nLatency & Drift"]
        MCP --> PROM
    end

    subgraph Monitoring ["Monitoring: Observability"]
        DM["Drift Monitor\nPSI + Chi-Square"]
        DR["drift_report.json"]
        DM --> DR
    end

    subgraph Feedback ["Feedback: Human-in-the-Loop"]
        FE["Feedback Engine\nsubmit_ground_truth"]
        LBL["human_labels.json\nThread-Safe Store"]
        RT["Retrain Trigger\ncount > 100"]
        FE --> LBL --> RT
    end

    subgraph Dashboard ["Dashboard: Next.js UI"]
        UI["Observability UI\nHealth Badge + Heatmap"]
        API["REST API\n/api/dashboard · /api/drift"]
    end

    SS -->|"JSON Logs"| MCP
    MCP -->|"Predictions"| UI
    DR -->|"Drift Scores"| PROM
    DR -->|"Heatmap Data"| UI
    RT -->|"retrain_required"| REG
    PROM -->|"Scraped by"| UI

    style Ingestion fill:#1e1b4b,stroke:#6366f1,color:#e2e8f0
    style MLPipeline fill:#1a2e1a,stroke:#22c55e,color:#e2e8f0
    style Inference fill:#2e1a1a,stroke:#ef4444,color:#e2e8f0
    style Monitoring fill:#2e2a1a,stroke:#eab308,color:#e2e8f0
    style Feedback fill:#1a2e2e,stroke:#06b6d4,color:#e2e8f0
    style Dashboard fill:#2e1a2e,stroke:#a855f7,color:#e2e8f0
```

## Component Details

### 1. Data Ingestion (`/data/stream_simulator.py`)

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/stream?count=N&chaos=level` | GET | Batch log generation |
| `/stream/single?chaos=level` | GET | Single log record |

**Chaos Levels**: `none` (0%), `low` (10%), `medium` (30%), `high` (60%), `extreme` (100%)

OOD injection multiplies all numerical features by **5x** above training maximum.

### 2. ML Pipeline (`/models/`)

```mermaid
flowchart LR
    RAW["Raw CSV\n45K rows × 25 cols"] --> FE["Feature Engineering"]
    FE --> NUM["StandardScaler\n6 numerical"]
    FE --> HASH["FeatureHasher\n256 buckets"]
    FE --> TFIDF["TF-IDF\n600 features"]
    FE --> DUMMY["One-Hot\n~22 cols"]
    NUM & HASH & TFIDF & DUMMY --> MAT["Sparse Matrix\n884 features"]
    MAT --> LGBM["LightGBM GBDT\n300 rounds"]
    LGBM --> PRED["10-Class Prediction"]
```

### 3. Inference Server (`/mcp-server/server.py`)

**Prometheus Metrics** exposed on `:9090/metrics`:

| Metric | Type | Description |
|---|---|---|
| `inference_latency_seconds` | Histogram | Per-request inference time |
| `total_anomalies_detected` | Counter | Logs classified with confidence > 0.5 |
| `total_inferences` | Counter | All inference requests |
| `model_drift_score` | Gauge | Max PSI score from drift report |
| `inference_errors_total` | Counter | Failed inference requests |

### 4. Drift Monitor (`/models/drift_monitor.py`)

| Feature Type | Method | Threshold |
|---|---|---|
| Numerical | PSI (Population Stability Index) | > 0.2 → retrain |
| Categorical | Chi-Square test | p < 0.05 → drifted |

### 5. Feedback Loop (`/mcp-server/feedback_engine.py`)

```mermaid
sequenceDiagram
    participant User as Human Reviewer
    participant FE as Feedback Engine
    participant FS as human_labels.json
    participant REG as registry.json

    User->>FE: submit_ground_truth(log_id, label)
    FE->>FE: Pydantic validation
    FE->>FS: Thread-safe append
    FE->>REG: Update count
    alt count > 100
        FE->>REG: retrain_required = true
        REG-->>User: ⚠️ Retrain triggered
    end
```

### 6. Dashboard (`/dashboard/app.py`)

| Endpoint | Description |
|---|---|
| `/` | Full HTML dashboard with drift heatmap |
| `/api/dashboard` | JSON payload (health, heatmap, model info) |
| `/api/drift` | Raw drift report |
| `/api/registry` | Model registry |

**Health Badge Logic**:
- 🟢 **Healthy** — No features drifted
- 🟡 **Drift Detected** — Some features drifted, PSI < threshold
- 🔴 **Training Required** — PSI > 0.2 on any numerical feature

## Infrastructure

### Docker Compose Services

| Service | Port | Container |
|---|---|---|
| MCP Server | 9090 | `sentinel-mcp` |
| Stream Simulator | 8100 | `sentinel-stream` |
| Dashboard | 8200 | `sentinel-dashboard` |

### CI/CD Pipeline (`.github/workflows/ci.yml`)

```mermaid
flowchart LR
    PUSH["git push"] --> LINT["flake8\nPEP8 check"]
    LINT --> TEST["pytest\n8 model tests"]
    TEST --> PASS["✅ CI Green"]
```

## Data Flow Summary

```
[CI/CD Logs] → Stream Simulator → MCP Server → LightGBM → Prediction
                                      ↓               ↓
                              Prometheus Metrics   Drift Monitor
                                      ↓               ↓
                                  Dashboard ←── drift_report.json
                                      ↑
                              Feedback Engine  ← Human Corrections
```
