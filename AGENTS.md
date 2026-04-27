# 🛡️ Sentinel-AIOps Mission Briefing

## 🎯 Project Mission
**Sentinel-AIOps** is an autonomous system designed for CI/CD log anomaly detection and remediation within the Antigravity ecosystem.

## 🧠 Technical Core
Our intelligence relies on Python-based Machine Learning models:
- **LightGBM Multiclass Classifier**: Supervised model trained on
  10 CI/CD failure categories using numerical telemetry features
  (CPU usage, build duration, memory consumption, retry count).
  Macro F1 ≈ 0.89 with ablation-verified telemetry-driven signal.
- **Isolation Forest**: Unsupervised anomaly detector for
  out-of-distribution log events that don't match known patterns.
- **PSI Drift Monitor**: Population Stability Index monitoring
  that detects training distribution shift and triggers retraining.

## 🏗️ Infrastructure
The system follows a robust, local-first architecture:
- **Inference Engine**: Powered by a FastMCP server for low-latency, localized model inference.
- **Observability Interface**: A Next.js 15 dashboard for real-time monitoring, visualization, and actionable insights.

## 📜 Workflow Rules
All autonomous agents interacting with this project MUST adhere strictly to the following operational protocols:
1. **Always log 'Reasoning' before execution**: Every action must be preceded by a clear, documented rationale.
2. **Save all ML metrics as 'Artifacts'**: Performance metrics (specifically F1-Score and PR AUC) must be rigorously tracked and saved as permanent project artifacts.

## 📁 Directory Map
The project is structurally divided into the following key domains:
- **`/data`**: For storing raw logs and processed datasets used in model training and inference.
- **`/models`**: For housing the trained weights and configurations of our Isolation Forest and LightGBM models.
- **`/mcp-server`**: For containing the FastMCP-based local inference logic and API endpoints.
- **`/dashboard`**: For the Next.js frontend code providing the observability and monitoring interface.
