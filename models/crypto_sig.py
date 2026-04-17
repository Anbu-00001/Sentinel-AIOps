import hashlib
import hmac
import logging
import os

log = logging.getLogger("sentinel.crypto_sig")

# In production, this should be securely populated. For local-first, we default
# to a local secret or generated key, preventing externally injected models from loading.
_DEFAULT_SECRET = "sentinel-local-crypto-secret-42"
MODEL_SIGNATURE_SECRET = os.getenv("MODEL_SIGNATURE_SECRET", _DEFAULT_SECRET).encode("utf-8")


def sign_artifact(filepath: str) -> None:
    """Generate an HMAC-SHA256 signature for the given file and write it to {filepath}.sig."""
    if not os.path.exists(filepath):
        log.error("Cannot sign %s: file not found", filepath)
        return

    with open(filepath, "rb") as f:
        data = f.read()

    signature = hmac.new(MODEL_SIGNATURE_SECRET, data, hashlib.sha256).hexdigest()
    sig_path = f"{filepath}.sig"

    with open(sig_path, "w", encoding="utf-8") as f:
        f.write(signature)

    log.info("Cryptographic signature generated for %s", os.path.basename(filepath))


def verify_artifact(filepath: str) -> bool:
    """Verify the HMAC-SHA256 signature of the given file against {filepath}.sig."""
    if not os.path.exists(filepath):
        log.error("Artifact %s not found.", filepath)
        return False

    sig_path = f"{filepath}.sig"
    if not os.path.exists(sig_path):
        log.error("Signature file missing for %s. Potential tampering or unsigned model.", filepath)
        return False

    with open(filepath, "rb") as f:
        data = f.read()

    with open(sig_path, "r", encoding="utf-8") as f:
        expected_sig = f.read().strip()

    actual_sig = hmac.new(MODEL_SIGNATURE_SECRET, data, hashlib.sha256).hexdigest()

    is_valid = hmac.compare_digest(expected_sig, actual_sig)
    if not is_valid:
        log.error("CRITICAL: Signature mismatch for %s! Possible modification detected.", filepath)

    return is_valid
