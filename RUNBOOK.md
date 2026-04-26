# Sentinel-AIOps Runbook

Complete operational guide for reproducing all results, deploying, and troubleshooting the Sentinel-AIOps CI/CD anomaly detection system.

## Prerequisites

- **Python 3.12+** — Required for all ML training and API endpoints
- **Docker and Docker Compose** — For containerized deployment
- **Git** — For version control
- **4 GB RAM minimum** — LightGBM training requires memory for 10k synthetic samples
- **Three required environment variables** — See [Secrets Configuration](#secrets-configuration) section

## Quick Start (Local Development)

```bash
# 1. Clone the repository
git clone https://github.com/Anbu-00001/Sentinel-AIOps.git
cd Sentinel-AIOps

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate secrets and configure environment
cp .env.example .env
python -c "import secrets; print('MODEL_SIGNATURE_SECRET=' + secrets.token_hex(32))" >> .env
python -c "import secrets; print('GITHUB_WEBHOOK_SECRET=' + secrets.token_hex(32))" >> .env
python -c "import secrets; print('SENTINEL_API_KEY=' + secrets.token_urlsafe(32))" >> .env

# 4. Load environment
export $(cat .env | xargs)

# 5. Train the model (generates all artifacts)
make train
# Expected output:
#   Macro F1 = 0.XX (between 0.60 and 0.90)
#   Model registered as lgbm_v2.X

# 6. Re-sign artifacts with your secret
make resign

# 7. Run tests
make test
# Expected: all tests pass

# 8. Start the dashboard locally
python -m uvicorn dashboard.app:app --host 0.0.0.0 --port 8200

# 9. Verify it works
curl http://localhost:8200/health
# Expected: {"status":"healthy","service":"sentinel-dashboard"}
```

### Docker Quick Start

```bash
make docker-build
make docker-up
# Services:
#   Dashboard:        http://localhost:8200
#   Stream Simulator: http://localhost:8100
#   MCP Metrics:      http://localhost:9090/metrics
```

## Secrets Configuration

All three secrets must be set before production deployment. Generate each one, add to your `.env` file, and **never commit `.env` to Git**.

### MODEL_SIGNATURE_SECRET

- **Purpose**: HMAC-SHA256 signing of `.joblib` model artifacts. Prevents loading tampered models (RCE mitigation via `joblib.load` arbitrary code execution).
- **Generation**:
  ```bash
  python -c "import secrets; print(secrets.token_hex(32))"
  ```
- **Where to set**: `.env` file, never committed to Git
- **When required**: Before running `make train` or `make resign`
- **What breaks if missing**: `preprocess.py` and `train_v2.py` will raise `RuntimeError` on startup

### GITHUB_WEBHOOK_SECRET

- **Purpose**: HMAC-SHA256 validation of GitHub webhook payloads. Prevents spoofed failure events from skewing drift metrics.
- **Generation**:
  ```bash
  python -c "import secrets; print(secrets.token_hex(32))"
  ```
- **Where to set**: `.env` file AND GitHub repo webhook settings
- **GitHub path**: Repository → Settings → Webhooks → Secret field
- **If unset**: All webhook requests are accepted (dev mode only — logged as a warning)

### SENTINEL_API_KEY

- **Purpose**: Authenticates requests to protected dashboard API endpoints (`/api/dashboard`, `/api/history`, `/api/psi`, `/api/drift`, `/api/registry`)
- **Generation**:
  ```bash
  python -c "import secrets; print(secrets.token_urlsafe(32))"
  ```
- **Where to set**: `.env` file
- **How to use**: Pass as `X-API-Key` header in all API requests:
  ```bash
  curl -H "X-API-Key: <key>" http://localhost:8200/api/dashboard
  ```
- **If unset**: All endpoints are open (development mode only)

## Reproducing the Model F1 Score

This section allows you to reproduce the model's performance metrics from scratch.

```bash
# Full pipeline: preprocess → train → sign
make retrain

# Expected output during training:
#   Macro F1 = 0.XX (assertion enforces range [0.60, 0.90])
#   Macro PR-AUC = 0.XX
```

- **Expected F1 range**: 0.72 – 0.90 (the training script asserts `0.60 <= F1 <= 0.90`)
- **Where results are saved**:
  - `models/registry.json` — Model version, F1, PR-AUC
  - `models/v2_report.json` — Full classification report

### Verifying with Ablation Study

```bash
make ablation
```

Expected output:
```
Full model F1      : 0.8XXX
Without TF-IDF F1  : 0.8XXX
Delta (text contrib): -0.0XXX
Ablation gates     : PASS
```

**How to read the result**: The "Without TF-IDF F1" score should be > 0.55, confirming the model learns from numerical telemetry features (build duration, CPU usage, etc.) rather than memorizing error message text. A delta < 0.30 confirms TF-IDF is only a weak auxiliary signal.

## Running the Test Suite

```bash
# Full suite including ablation (~2-3 minutes)
make test

# Fast mode excluding ablation fixture (~30 seconds)
make test-fast

# Expected: all tests pass, count visible in summary
```

### Interpreting `test_ablation.py` Failures

| Failure | Cause | Fix |
|---------|-------|-----|
| `f1_no_tfidf < 0.55` | Numerical feature distributions lost discriminative signal | Check `generate_synthetic_baseline()` per-class specs |
| `f1_full > 0.90` | Error templates contain class-specific keywords | Remove keywords from `GENERIC_ERROR_TEMPLATES` |
| `delta > 0.30` | TF-IDF features carry too much signal | Verify error messages are class-neutral |

## Docker Deployment

```bash
# Build all images
make docker-build

# Start all services (detached)
make docker-up

# Verify services
curl http://localhost:8200/health        # Dashboard API
curl http://localhost:8200               # Dashboard UI
curl http://localhost:9090/metrics       # Prometheus metrics
curl http://localhost:8100/health        # Stream simulator

# Inspect startup logs
make docker-logs

# Stop all services
make docker-down
```

### Service Architecture

| Service | Port | Purpose |
|---------|------|---------|
| `dashboard` | 8200 | FastAPI dashboard + webhook receiver |
| `mcp-server` | 9090 | FastMCP inference + Prometheus metrics |
| `stream-simulator` | 8100 | Synthetic log stream generator |

All services communicate over the `sentinel-net` Docker network.

## Secret Rotation

When rotating secrets:

1. **Generate new value**:
   ```bash
   python -c "import secrets; print(secrets.token_hex(32))"
   ```
2. **Update `.env` file** with the new value
3. **For `MODEL_SIGNATURE_SECRET` only**: Re-sign all model artifacts:
   ```bash
   export $(cat .env | xargs)
   make resign
   ```
4. **Restart services**:
   ```bash
   make docker-down && make docker-up
   ```
5. **For `GITHUB_WEBHOOK_SECRET`**: Also update the secret in GitHub webhook settings:
   - Repository → Settings → Webhooks → Edit → Secret field

## GitHub Webhook Configuration

Configure GitHub to send CI/CD failure events to Sentinel-AIOps:

1. Navigate to your repository → **Settings** → **Webhooks** → **Add webhook**
2. Configure:
   - **Payload URL**: `http://<your-ip>:8200/webhook/github`
   - **Content type**: `application/json`
   - **Secret**: Same value as `GITHUB_WEBHOOK_SECRET` from your `.env` file
   - **Events**: Select "Workflow runs"
3. **Verify delivery**: After a workflow completes, check the GitHub webhook delivery log — it should show HTTP 200

## Troubleshooting

### "RuntimeError: MODEL_SIGNATURE_SECRET not set"

**Cause**: The `MODEL_SIGNATURE_SECRET` environment variable is not set.

**Fix**:
```bash
export MODEL_SIGNATURE_SECRET=$(python -c "import secrets; print(secrets.token_hex(32))")
# Or load from .env:
export $(cat .env | xargs)
```

### "403 Invalid webhook signature"

**Cause**: The `GITHUB_WEBHOOK_SECRET` in your `.env` does not match the secret configured in GitHub webhook settings.

**Fix**: Ensure the secret value is identical in both locations. Regenerate and update both if needed.

### "403 X-API-Key header is required"

**Cause**: `SENTINEL_API_KEY` is configured on the server but the request does not include the `X-API-Key` header.

**Fix**:
```bash
curl -H "X-API-Key: <your-key>" http://localhost:8200/api/dashboard
```

### "database is locked"

**Cause**: SQLite concurrent write contention. This should not occur with the default `--workers 1` configuration.

**Fix**: Verify `docker-compose.yml` has `--workers 1` for the dashboard service. If you need multiple workers, migrate to PostgreSQL.

### "Artifact signature validation failed"

**Cause**: Model `.joblib` files were signed with a different `MODEL_SIGNATURE_SECRET` than the one currently configured.

**Fix**:
```bash
export $(cat .env | xargs)
make resign
```

### F1 assertion fails during `make train`

**Cause**: The trained model's macro F1 score fell outside the credible range `[0.60, 0.90]`.

**Fix**: Check `models/preprocess.py::generate_synthetic_baseline()` for changes to per-class distributions or noise parameters. Run `make ablation` to diagnose whether the issue is feature signal or text leakage.

### `make ablation` shows delta > 0.30

**Cause**: TF-IDF features are contributing too much discriminative signal, meaning error message templates contain class-specific keywords.

**Fix**: Review `GENERIC_ERROR_TEMPLATES` in `models/preprocess.py` and remove any words that correlate with specific failure classes (e.g., "timeout", "OOM", "CVE", "permission denied").

## Architecture Reference

Sentinel-AIOps follows a local-first architecture where GitHub webhook events are received by the FastAPI dashboard, mapped to numerical CI/CD telemetry features, and classified by a LightGBM model into one of 10 failure categories. Inference results are persisted to SQLite and exposed via a real-time observability dashboard with drift monitoring. The FastMCP server provides an alternative inference path with Prometheus metrics export. For the full system diagram, see `SYSTEM_DESIGN.md`.
