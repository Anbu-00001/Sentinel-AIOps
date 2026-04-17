"""
rebuild_pipeline.py — Sentinel-AIOps Unified Artifact Lifecycle
================================================================
Sequentially executes the preprocessing and training scripts to
ensure all model artifacts and feature matrices are in sync.
"""

import subprocess
import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("sentinel.rebuild")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_script(script_path: str):
    """Run a python script and wait for completion."""
    abs_path = os.path.join(ROOT, script_path)
    log.info("Executing: %s", script_path)

    # Use the same python interpreter
    result = subprocess.run([sys.executable, abs_path], capture_output=False, text=True)

    if result.returncode != 0:
        log.error("Script %s failed with exit code %d", script_path, result.returncode)
        sys.exit(result.returncode)
    log.info("Success: %s", script_path)


def main():
    log.info("=== Starting Unified Pipeline Rebuild ===")

    # Ensure CI=true is set for synthetic data generation if datasets are missing
    os.environ["CI"] = "true"

    # 1. Preprocess (generates feature_matrix.npz, labels.npy, scaler.joblib, etc.)
    run_script("models/preprocess.py")

    # 2. Train (generates lgbm_model.joblib, label_encoder.joblib, v2_report.json)
    run_script("models/train_v2.py")

    log.info("=== Pipeline Rebuild Complete. All artifacts are synchronized. ===")


if __name__ == "__main__":
    main()
