#!/usr/bin/env python3
"""
resign_artifacts.py — Re-sign all .joblib model artifacts
==========================================================
After changing or setting MODEL_SIGNATURE_SECRET for the first time,
run this script to regenerate all .sig files.

Usage:
    MODEL_SIGNATURE_SECRET=<your-secret> python models/resign_artifacts.py

Workflow Rules (AGENTS.md):
  - Log 'Reasoning' before execution.
"""

import glob
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crypto_sig

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("sentinel.resign")

MODELS_DIR: str = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    """
    Discover all .joblib files in models/ and regenerate their .sig files.

    Raises
    ------
    RuntimeError
        If MODEL_SIGNATURE_SECRET is not set (propagated from crypto_sig).
    """
    log.info("Reasoning: Scanning %s for .joblib artifacts to re-sign.", MODELS_DIR)
    pattern: str = os.path.join(MODELS_DIR, "*.joblib")
    joblib_files: list[str] = sorted(glob.glob(pattern))

    if not joblib_files:
        log.warning("No .joblib files found in %s — nothing to sign.", MODELS_DIR)
        return

    for filepath in joblib_files:
        log.info("Reasoning: Re-signing %s", os.path.basename(filepath))
        crypto_sig.sign_artifact(filepath)

    log.info("Re-signed %d artifact(s) successfully.", len(joblib_files))


if __name__ == "__main__":
    main()
