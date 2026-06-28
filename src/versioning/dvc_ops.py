"""
DVC Operations.

Wraps DVC CLI commands in Python functions so the Airflow DAG
doesn't have to deal with raw subprocess calls.

Every snapshot creates a git commit that links:
  - DVC data hash  →  what data was used
  - Git commit SHA →  what code version produced it
  - Timestamp      →  when it happened

This lets you roll back to any previous data version with:
    dvc checkout <git-commit>
"""

import os
import subprocess
import hashlib
from datetime import datetime
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

RAW_DATA_PATH       = os.getenv("RAW_DATA_PATH",       "data/raw")
PROCESSED_DATA_PATH = os.getenv("PROCESSED_DATA_PATH", "data/processed")
FEATURES_DATA_PATH  = os.getenv("FEATURES_DATA_PATH",  "data/features")

GIT_USER_NAME  = os.getenv("GIT_USER_NAME",  "mlops-pipeline")
GIT_USER_EMAIL = os.getenv("GIT_USER_EMAIL", "mlops@pipeline.local")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _run(cmd: list, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and log it."""
    logger.debug(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def _ensure_git_config():
    """
    Make sure git user.name and user.email are set.
    Airflow containers often don't have these, which makes git commit fail.
    """
    result = _run(["git", "config", "user.name"], check=False)
    if not result.stdout.strip():
        _run(["git", "config", "user.name",  GIT_USER_NAME])
        _run(["git", "config", "user.email", GIT_USER_EMAIL])
        logger.info(f"Set git identity: {GIT_USER_NAME} <{GIT_USER_EMAIL}>")


def _git_commit(message: str):
    """Stage DVC metafiles and create a git commit."""
    _ensure_git_config()

    # Stage all .dvc files and dvc.lock (DVC metafiles go into git)
    _run(["git", "add", "*.dvc", "dvc.lock", ".dvc/config"], check=False)

    # Only commit if there's actually something staged
    status = _run(["git", "status", "--porcelain"], check=False)
    if status.stdout.strip():
        _run(["git", "commit", "-m", message])
        logger.info(f"Git commit: {message}")
    else:
        logger.info("Nothing to commit — data unchanged since last snapshot")


# ── DVC operations ────────────────────────────────────────────────────────────

def add_to_dvc(path: str) -> str:
    """
    Track a file or directory with DVC.
    Creates/updates the <path>.dvc metafile.

    Returns:
        DVC content hash for the tracked data.
    """
    result = _run(["dvc", "add", path])
    logger.info(f"DVC tracked: {path}")

    # Extract the md5 hash from the .dvc file
    dvc_file = f"{path}.dvc"
    if os.path.exists(dvc_file):
        import yaml
        with open(dvc_file) as f:
            meta = yaml.safe_load(f)
        return meta.get("outs", [{}])[0].get("md5", "unknown")

    return "unknown"


def push_to_remote():
    """Push cached data to DVC remote storage."""
    result = _run(["dvc", "push"])
    logger.info("DVC push complete")
    return result.returncode == 0


def pull_from_remote(paths: list = None):
    """Pull data from DVC remote (used to restore data on a new machine)."""
    cmd = ["dvc", "pull"]
    if paths:
        cmd += paths
    _run(cmd)
    logger.info("DVC pull complete")


def checkout_version(git_ref: str):
    """
    Restore data to the version it was at a given git commit/tag.

    Usage:
        checkout_version("abc1234")   # restore data from that commit
        checkout_version("v1.0")      # restore data from a git tag
    """
    logger.info(f"Checking out data version at git ref: {git_ref}")
    _run(["git", "checkout", git_ref, "--", "*.dvc"])
    _run(["dvc", "checkout"])
    logger.info(f"Data restored to version {git_ref}")


def get_current_hash(path: str) -> str:
    """
    Return the DVC content hash for a tracked path.
    Useful for logging to MLflow so each training run knows exactly
    which data version it used.
    """
    dvc_file = f"{path}.dvc"
    if not os.path.exists(dvc_file):
        return "not-tracked"

    import yaml
    with open(dvc_file) as f:
        meta = yaml.safe_load(f)
    return meta.get("outs", [{}])[0].get("md5", "unknown")


def list_versions(n: int = 10) -> list:
    """
    Return the last n data versions as a list of dicts with
    git commit SHA, timestamp, and commit message.
    """
    result = _run(
        ["git", "log", f"-{n}", "--pretty=format:%H|%ai|%s", "--", "*.dvc"],
        check=False,
    )
    versions = []
    for line in result.stdout.strip().splitlines():
        parts = line.split("|", 2)
        if len(parts) == 3:
            versions.append({
                "sha":       parts[0],
                "timestamp": parts[1],
                "message":   parts[2],
            })
    return versions


# ── Snapshot functions (called by Airflow) ────────────────────────────────────

def snapshot_raw_data() -> dict:
    """
    Version the raw data directory after scraping.
    Called by Airflow task_version_data.

    Returns:
        {"path": ..., "hash": ..., "commit": ...}
    """
    logger.info("Snapshotting raw data with DVC...")
    data_hash = add_to_dvc(RAW_DATA_PATH)
    date_str  = datetime.now().strftime("%Y-%m-%d")
    _git_commit(f"data: raw snapshot {date_str}")

    push_ok = push_to_remote()
    if not push_ok:
        logger.warning("DVC push failed — data tracked locally only")

    sha_result = _run(["git", "rev-parse", "HEAD"], check=False)
    git_sha    = sha_result.stdout.strip()

    logger.info(f"Raw data versioned — DVC hash: {data_hash} | git: {git_sha[:8]}")
    return {"path": RAW_DATA_PATH, "hash": data_hash, "commit": git_sha}


def snapshot_processed_data() -> dict:
    """Version the processed data directory after cleaning."""
    logger.info("Snapshotting processed data with DVC...")
    data_hash = add_to_dvc(PROCESSED_DATA_PATH)
    date_str  = datetime.now().strftime("%Y-%m-%d")
    _git_commit(f"data: processed snapshot {date_str}")
    push_to_remote()

    sha_result = _run(["git", "rev-parse", "HEAD"], check=False)
    git_sha    = sha_result.stdout.strip()

    return {"path": PROCESSED_DATA_PATH, "hash": data_hash, "commit": git_sha}


def snapshot_features() -> dict:
    """Version the features directory after feature engineering."""
    logger.info("Snapshotting feature data with DVC...")
    data_hash = add_to_dvc(FEATURES_DATA_PATH)
    date_str  = datetime.now().strftime("%Y-%m-%d")
    _git_commit(f"data: features snapshot {date_str}")
    push_to_remote()

    sha_result = _run(["git", "rev-parse", "HEAD"], check=False)
    git_sha    = sha_result.stdout.strip()

    return {"path": FEATURES_DATA_PATH, "hash": data_hash, "commit": git_sha}


if __name__ == "__main__":
    info = snapshot_raw_data()
    print(f"Snapshot complete: {info}")
