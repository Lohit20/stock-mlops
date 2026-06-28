"""
Model Registry — Step 9.

Wraps MLflow Model Registry with a full lifecycle:

  run_id ──register──► None ──stage──► Staging ──promote──► Production
                                                                  │
                         Archived ◄──────────────────────────────┘

Key capabilities:
  • promote_to_staging()       — move a run's registered version to Staging
  • promote_to_production()    — validate + move Staging → Production
  • rollback()                 — restore previous Archived → Production
  • get_registry_summary()     — DataFrame view of all versions + stages
  • compare_versions()         — side-by-side metric comparison
  • set_aliases()              — champion / challenger aliases (MLflow 2.x)
  • cleanup_old_versions()     — archive excess old versions
  • register_best_model()      — full pipeline (register → stage → validate → promote)

Airflow entry point:
    from src.registry.model_registry import register_best_model
    register_best_model(best_info, validation_metrics)
"""

import os
import pandas as pd
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

REGISTERED_MODELS = {
    "LSTM":    "stock_price_forecaster_lstm",
    "TFT":     "stock_price_forecaster_tft",
    "TimesFM": "stock_price_forecaster_timesfm",
    "ARM":     "stock_price_forecaster_arm",
}

# Default minimum quality thresholds before promoting to Production.
# Keys match MLflow metric names logged during training.
DEFAULT_MIN_THRESHOLDS: dict[str, float] = {
    "test_directional_accuracy": 0.45,   # above random guessing
    "test_mape":                 50.0,   # MAPE below 50 % (generous)
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _client():
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return mlflow.tracking.MlflowClient()


def _model_name(model_type: str) -> str:
    name = REGISTERED_MODELS.get(model_type)
    if name is None:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Valid options: {list(REGISTERED_MODELS)}"
        )
    return name


def _run_metrics(client, run_id: str) -> dict:
    try:
        run = client.get_run(run_id)
        return dict(run.data.metrics)
    except Exception:
        return {}


def _version_at_stage(client, model_name: str, stage: str):
    """Return the first ModelVersion at the given stage, or None."""
    versions = client.search_model_versions(f"name='{model_name}'")
    return next(
        (v for v in versions if getattr(v, "current_stage", "") == stage),
        None,
    )


def _transition(client, model_name: str, version: str, stage: str, archive=False):
    client.transition_model_version_stage(
        name=model_name, version=version, stage=stage,
        archive_existing_versions=archive,
    )
    logger.info(f"{model_name} v{version} → {stage}")


# ── Registry summary ──────────────────────────────────────────────────────────

def get_registry_summary() -> pd.DataFrame:
    """
    Return a DataFrame of every registered model version across all model types.

    Columns: model_name, model_type, version, stage, run_id, creation_timestamp,
             val_loss, test_rmse, test_mape, test_directional_accuracy
    """
    client = _client()
    rows   = []

    for model_type, model_name in REGISTERED_MODELS.items():
        try:
            versions = client.search_model_versions(f"name='{model_name}'")
        except Exception as exc:
            logger.warning(f"Could not list versions for {model_name}: {exc}")
            continue

        for v in versions:
            metrics = _run_metrics(client, v.run_id)
            rows.append({
                "model_name":               model_name,
                "model_type":               model_type,
                "version":                  v.version,
                "stage":                    getattr(v, "current_stage", ""),
                "run_id":                   v.run_id,
                "creation_timestamp":       getattr(v, "creation_timestamp", None),
                "val_loss":                 metrics.get("val_loss"),
                "test_rmse":                metrics.get("test_rmse"),
                "test_mape":                metrics.get("test_mape"),
                "test_directional_accuracy": metrics.get("test_directional_accuracy"),
            })

    return pd.DataFrame(rows)


# ── Staging gate ──────────────────────────────────────────────────────────────

def promote_to_staging(model_type: str, run_id: str) -> str:
    """
    Move the registered version linked to `run_id` to Staging.

    If no version is registered for this run_id, logs a warning and returns "".

    Returns the version string (e.g. "2").
    """
    client     = _client()
    model_name = _model_name(model_type)

    versions = client.search_model_versions(f"name='{model_name}'")
    target   = next((v for v in versions if v.run_id == run_id), None)

    if target is None:
        logger.warning(
            f"No registered version for {model_name} run_id={run_id[:8]}. "
            "Model may not have been logged with registered_model_name during training."
        )
        return ""

    _transition(client, model_name, target.version, "Staging")
    return target.version


# ── Validation ────────────────────────────────────────────────────────────────

def validate_version(
    model_type: str,
    version:    str,
    min_thresholds: dict | None = None,
) -> tuple[bool, dict]:
    """
    Check that a Staging version meets minimum quality thresholds.

    Returns:
        (passed: bool, report: dict)
        report maps each threshold to {threshold, actual, passed}.
    """
    client     = _client()
    model_name = _model_name(model_type)
    thresholds = min_thresholds or DEFAULT_MIN_THRESHOLDS

    versions = client.search_model_versions(f"name='{model_name}'")
    target   = next((v for v in versions if v.version == str(version)), None)
    if target is None:
        logger.warning(f"Version {version} not found for {model_name}")
        return False, {}

    metrics = _run_metrics(client, target.run_id)
    report  = {}
    passed  = True

    for metric, threshold in thresholds.items():
        actual = metrics.get(metric)
        if actual is None:
            logger.warning(f"Metric '{metric}' missing for {model_name} v{version} — skipping check")
            report[metric] = {"threshold": threshold, "actual": None, "passed": None}
            continue

        # For directional_accuracy higher is better; for loss/error lower is better
        if "accuracy" in metric:
            ok = float(actual) >= threshold
        else:
            ok = float(actual) <= threshold

        report[metric] = {
            "threshold": threshold,
            "actual":    float(actual),
            "passed":    ok,
        }
        if not ok:
            passed = False
            logger.warning(
                f"FAIL {model_name} v{version}: {metric}={actual:.4f} "
                f"(threshold={'≥' if 'accuracy' in metric else '≤'}{threshold})"
            )
        else:
            logger.info(
                f"PASS {model_name} v{version}: {metric}={actual:.4f}"
            )

    return passed, report


# ── Production promotion ──────────────────────────────────────────────────────

def promote_to_production(
    model_type:     str,
    version:        str,
    min_thresholds: dict | None = None,
    skip_validation: bool = False,
) -> bool:
    """
    Validate a Staging version, then promote it to Production.

    Archives the current Production version first.

    Args:
        model_type:       e.g. "ARM", "LSTM"
        version:          version string to promote (must currently be in Staging)
        min_thresholds:   overrides DEFAULT_MIN_THRESHOLDS; pass {} to disable
        skip_validation:  bypass metric validation (use for manual overrides)

    Returns:
        True if promoted, False if validation failed.
    """
    client     = _client()
    model_name = _model_name(model_type)

    if not skip_validation:
        passed, report = validate_version(model_type, version, min_thresholds)
        if not passed:
            failing = [k for k, v in report.items() if v and v.get("passed") is False]
            logger.error(
                f"Promotion blocked: {model_name} v{version} failed "
                f"thresholds: {failing}"
            )
            return False

    _transition(client, model_name, version, "Production", archive=True)
    return True


# ── Rollback ──────────────────────────────────────────────────────────────────

def rollback(model_type: str) -> str:
    """
    Restore the most-recently Archived version to Production.

    The current Production version is moved to Archived first.

    Returns the restored version string, or "" if no Archived version exists.
    """
    client     = _client()
    model_name = _model_name(model_type)

    versions  = client.search_model_versions(f"name='{model_name}'")
    archived  = [
        v for v in versions
        if getattr(v, "current_stage", "") == "Archived"
    ]

    if not archived:
        logger.warning(f"No Archived versions for {model_name} — cannot rollback")
        return ""

    # Most recent archived version (highest creation_timestamp)
    prev = max(archived, key=lambda v: getattr(v, "creation_timestamp", 0))

    # Move current Production to Archived
    current_prod = _version_at_stage(client, model_name, "Production")
    if current_prod:
        _transition(client, model_name, current_prod.version, "Archived")

    # Restore previous
    _transition(client, model_name, prev.version, "Production")
    logger.info(f"Rollback complete: {model_name} → v{prev.version}")
    return prev.version


# ── Version comparison ────────────────────────────────────────────────────────

def compare_versions(
    model_type: str,
    version_a:  str,
    version_b:  str,
) -> dict:
    """
    Side-by-side metric comparison of two registered versions.

    Returns:
        {
          "model_name": str,
          "version_a": {"version", "run_id", "stage", "metrics"},
          "version_b": {"version", "run_id", "stage", "metrics"},
          "winner":    "a" | "b" | "tie",
          "winner_version": str,
        }
    """
    client     = _client()
    model_name = _model_name(model_type)

    versions = {v.version: v for v in client.search_model_versions(f"name='{model_name}'")}

    def _info(ver):
        v = versions.get(str(ver))
        if v is None:
            raise ValueError(f"Version {ver} not found for {model_name}")
        return {
            "version": v.version,
            "run_id":  v.run_id,
            "stage":   getattr(v, "current_stage", ""),
            "metrics": _run_metrics(client, v.run_id),
        }

    a = _info(version_a)
    b = _info(version_b)

    # Compare on test_rmse (lower is better); fall back to val_loss
    for metric in ("test_rmse", "val_loss"):
        va = a["metrics"].get(metric)
        vb = b["metrics"].get(metric)
        if va is not None and vb is not None:
            if va < vb:
                winner, winner_version = "a", a["version"]
            elif vb < va:
                winner, winner_version = "b", b["version"]
            else:
                winner, winner_version = "tie", a["version"]
            break
    else:
        winner, winner_version = "unknown", a["version"]

    return {
        "model_name":     model_name,
        "version_a":      a,
        "version_b":      b,
        "winner":         winner,
        "winner_version": winner_version,
    }


# ── Champion / Challenger aliases ─────────────────────────────────────────────

def set_champion_challenger(
    model_type:          str,
    champion_version:    str,
    challenger_version:  str,
) -> None:
    """
    Assign 'champion' and 'challenger' aliases to two versions.
    Requires MLflow 2.x — silently skips on older versions.
    """
    client     = _client()
    model_name = _model_name(model_type)

    try:
        client.set_registered_model_alias(model_name, "champion",   champion_version)
        client.set_registered_model_alias(model_name, "challenger", challenger_version)
        logger.info(
            f"Aliases set: {model_name} champion=v{champion_version} "
            f"challenger=v{challenger_version}"
        )
    except AttributeError:
        logger.warning(
            "set_registered_model_alias not available (MLflow < 2.x) — aliases skipped"
        )


def get_model_by_alias(model_type: str, alias: str):
    """
    Load a model version by alias (e.g. 'champion', 'challenger').
    Returns the pyfunc model, or None if alias not set.
    """
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_name = _model_name(model_type)
    try:
        return mlflow.pyfunc.load_model(f"models:/{model_name}@{alias}")
    except Exception as exc:
        logger.warning(f"Could not load {model_name}@{alias}: {exc}")
        return None


# ── Cleanup ───────────────────────────────────────────────────────────────────

def cleanup_old_versions(model_type: str, keep_n: int = 5) -> int:
    """
    Delete registered model versions beyond the `keep_n` most recent.
    Only deletes versions in the 'Archived' stage (never touches Staging/Production).

    Returns the number of versions deleted.
    """
    client     = _client()
    model_name = _model_name(model_type)

    versions = client.search_model_versions(f"name='{model_name}'")
    archived = [
        v for v in versions
        if getattr(v, "current_stage", "") == "Archived"
    ]

    # Sort oldest first
    archived.sort(key=lambda v: getattr(v, "creation_timestamp", 0))
    to_delete = archived[:-keep_n] if len(archived) > keep_n else []

    for v in to_delete:
        client.delete_model_version(name=model_name, version=v.version)
        logger.info(f"Deleted {model_name} v{v.version} (Archived, beyond keep_n={keep_n})")

    return len(to_delete)


# ── Latest production info ────────────────────────────────────────────────────

def get_latest_production(model_type: str) -> dict | None:
    """
    Return metadata for the current Production version of a model, or None.

    Dict keys: version, run_id, stage, metrics, model_uri
    """
    client     = _client()
    model_name = _model_name(model_type)

    v = _version_at_stage(client, model_name, "Production")
    if v is None:
        return None

    return {
        "model_type": model_type,
        "model_name": model_name,
        "version":    v.version,
        "run_id":     v.run_id,
        "stage":      "Production",
        "metrics":    _run_metrics(client, v.run_id),
        "model_uri":  f"models:/{model_name}/Production",
    }


# ── Airflow entry point ───────────────────────────────────────────────────────

def register_best_model(
    best_info:          dict,
    validation_metrics: dict | None = None,
    min_thresholds:     dict | None = None,
    skip_validation:    bool = False,
) -> dict:
    """
    Full Step 9 pipeline for one winning model:
      1. Promote the run's version to Staging
      2. Validate against thresholds
      3. Promote to Production
      4. Set champion alias
      5. Cleanup old archived versions

    Args:
        best_info:           dict from run_comparison() — must have 'model_type' and 'run_id'
        validation_metrics:  override metrics for validation (falls back to stored run metrics)
        min_thresholds:      override DEFAULT_MIN_THRESHOLDS
        skip_validation:     promote without metric checks (emergency override)

    Returns:
        Dict with: model_type, version, promoted (bool), stage
    """
    model_type = best_info.get("model_type", "")
    run_id     = best_info.get("run_id", "")

    if not model_type or not run_id:
        raise ValueError("best_info must contain 'model_type' and 'run_id'")

    # 1. Staging
    version = promote_to_staging(model_type, run_id)
    if not version:
        return {"model_type": model_type, "version": "", "promoted": False,
                "stage": "None"}

    # 2 + 3. Validate → Production
    promoted = promote_to_production(
        model_type=model_type,
        version=version,
        min_thresholds=min_thresholds,
        skip_validation=skip_validation,
    )

    if promoted:
        # 4. Champion alias
        try:
            client     = _client()
            model_name = _model_name(model_type)
            versions   = client.search_model_versions(f"name='{model_name}'")
            archived   = [
                v.version for v in versions
                if getattr(v, "current_stage", "") == "Archived"
            ]
            if archived:
                # The most recent archived version becomes challenger
                challenger = max(archived, key=lambda v: int(v) if v.isdigit() else 0)
                set_champion_challenger(model_type, version, challenger)
        except Exception as exc:
            logger.warning(f"Could not set aliases: {exc}")

        # 5. Cleanup
        try:
            deleted = cleanup_old_versions(model_type, keep_n=5)
            if deleted:
                logger.info(f"Cleaned up {deleted} old archived versions")
        except Exception as exc:
            logger.warning(f"Cleanup failed: {exc}")

    stage = "Production" if promoted else "Staging"
    return {
        "model_type": model_type,
        "version":    version,
        "promoted":   promoted,
        "stage":      stage,
    }
