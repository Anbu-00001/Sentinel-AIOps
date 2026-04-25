"""
crypto_sig.py — HMAC-SHA256 Model Artifact Signing & Verification
===================================================================
Provides cryptographic signing and verification for ML artifacts (.joblib files)
to prevent tampering or injection of malicious model weights.

The signing secret is loaded exclusively from the ``MODEL_SIGNATURE_SECRET``
environment variable — no hardcoded fallback is shipped in source.

Workflow Rules (AGENTS.md):
  - Log 'Reasoning' before execution.

Security Note:
  After switching to env-only secrets, all existing .sig files become invalid.
  Run ``python models/resign_artifacts.py`` after setting MODEL_SIGNATURE_SECRET
  for the first time.  Also run ``git rm --cached models/*.sig`` once to stop
  tracking .sig files that were previously committed.
"""

import hashlib
import hmac
import logging
import os

log = logging.getLogger("sentinel.crypto_sig")


def _get_secret() -> bytes:
    """
    Retrieve the HMAC signing secret from the environment.

    Returns
    -------
    bytes
        UTF-8 encoded signing secret.

    Raises
    ------
    RuntimeError
        If ``MODEL_SIGNATURE_SECRET`` is not set and the ``TESTING``
        escape hatch is not active.
    """
    secret: str | None = os.getenv("MODEL_SIGNATURE_SECRET")

    if secret is not None:
        return secret.encode("utf-8")

    # Testing escape hatch — only when TESTING=1 AND no real secret is set
    if os.getenv("TESTING") == "1":
        log.warning(
            "WARNING: Using test-only signature secret. "
            "Never use TESTING=1 in production."
        )
        return "test-secret-do-not-use-in-production".encode("utf-8")

    raise RuntimeError(
        "MODEL_SIGNATURE_SECRET environment variable is required but not set. "
        'Generate one with: python -c "import secrets; print(secrets.token_hex(32))"'
    )


def sign_artifact(filepath: str) -> None:
    """
    Generate an HMAC-SHA256 signature for the given file and write it to
    ``{filepath}.sig``.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to the artifact file to sign.
    """
    log.info("Reasoning: Signing artifact %s with HMAC-SHA256.", os.path.basename(filepath))

    if not os.path.exists(filepath):
        log.error("Cannot sign %s: file not found", filepath)
        return

    secret: bytes = _get_secret()

    with open(filepath, "rb") as f:
        data: bytes = f.read()

    signature: str = hmac.new(secret, data, hashlib.sha256).hexdigest()
    sig_path: str = f"{filepath}.sig"

    with open(sig_path, "w", encoding="utf-8") as f:
        f.write(signature)

    log.info("Cryptographic signature generated for %s", os.path.basename(filepath))


def verify_artifact(filepath: str) -> bool:
    """
    Verify the HMAC-SHA256 signature of the given file against
    ``{filepath}.sig``.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to the artifact file to verify.

    Returns
    -------
    bool
        ``True`` if the signature is valid, ``False`` otherwise.
    """
    if not os.path.exists(filepath):
        log.error("Artifact %s not found.", filepath)
        return False

    sig_path: str = f"{filepath}.sig"
    if not os.path.exists(sig_path):
        log.error("Signature file missing for %s. Potential tampering or unsigned model.", filepath)
        return False

    secret: bytes = _get_secret()

    with open(filepath, "rb") as f:
        data: bytes = f.read()

    with open(sig_path, "r", encoding="utf-8") as f:
        expected_sig: str = f.read().strip()

    actual_sig: str = hmac.new(secret, data, hashlib.sha256).hexdigest()

    is_valid: bool = hmac.compare_digest(expected_sig, actual_sig)
    if not is_valid:
        log.error("CRITICAL: Signature mismatch for %s! Possible modification detected.", filepath)

    return is_valid
