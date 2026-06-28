"""
Airflow task callables extracted from the DAG file.

All functions here are pure Python — no Airflow imports at module level.
This lets tests import and exercise them without Apache Airflow installed.

The DAG file (airflow/dags/pipeline_dag.py) imports these and wraps them
in PythonOperator / BranchPythonOperator calls.
"""

import os
from loguru import logger


# ── Helpers ───────────────────────────────────────────────────────────────────

def _data_tags(context: dict) -> dict:
    """Pull DVC hash + commit from XCom set by version_data task."""
    ti = context.get("ti")
    if ti is None:
        return {}
    return {
        "data_hash":   ti.xcom_pull(task_ids="version_data", key="raw_data_hash"),
        "data_commit": ti.xcom_pull(task_ids="version_data", key="raw_data_commit"),
    }


# ── Ingestion ─────────────────────────────────────────────────────────────────

def task_scrape() -> None:
    from src.ingestion.scraper import run_price_fetcher, run_scraper
    run_price_fetcher()
    run_scraper()


def task_validate() -> None:
    from src.ingestion.validator import run_validation
    passed = run_validation(os.getenv("RAW_DATA_PATH", "data/raw"))
    if not passed:
        raise ValueError("Data validation failed — check logs for details")


def task_version_data(**context) -> None:
    from src.versioning.dvc_ops import snapshot_raw_data
    info = snapshot_raw_data()
    context["ti"].xcom_push(key="raw_data_hash",   value=info["hash"])
    context["ti"].xcom_push(key="raw_data_commit",  value=info["commit"])


def task_feature_engineering(**context) -> None:
    from src.features.engineer import run_pipeline
    outputs = run_pipeline()
    context["ti"].xcom_push(key="feature_outputs", value=outputs)


# ── Training ──────────────────────────────────────────────────────────────────

def task_train_lstm(**context) -> None:
    from src.training.train_lstm import train
    train(**_data_tags(context))


def task_train_tft(**context) -> None:
    from src.training.train_tft import train
    train(**_data_tags(context))


def task_train_timesfm(**context) -> None:
    from src.training.train_timesfm import train
    train(**_data_tags(context))


def task_train_arm(**context) -> None:
    from src.training.train_arm import train
    train(**_data_tags(context))


# ── Evaluation + registry ─────────────────────────────────────────────────────

def task_compare_models(**context) -> None:
    from src.evaluation.compare_models import run_comparison
    best = run_comparison()
    ti   = context["ti"]
    ti.xcom_push(key="best_model_type", value=best.get("model_type"))
    ti.xcom_push(key="best_run_id",     value=best.get("run_id"))
    ti.xcom_push(key="best_val_loss",   value=best.get("val_loss"))


def task_register_model(**context) -> None:
    from src.registry.model_registry import register_best_model
    ti        = context["ti"]
    best_info = {
        "model_type": ti.xcom_pull(task_ids="compare_models", key="best_model_type"),
        "run_id":     ti.xcom_pull(task_ids="compare_models", key="best_run_id"),
        "val_loss":   ti.xcom_pull(task_ids="compare_models", key="best_val_loss"),
    }
    result = register_best_model(best_info)
    logger.info(
        f"Registry: {result['model_type']} v{result['version']} "
        f"→ {result['stage']} (promoted={result['promoted']})"
    )
    ti.xcom_push(key="registry_result", value=result)


# ── Deployment ────────────────────────────────────────────────────────────────

def task_deploy_to_api(**context) -> None:
    """
    Hot-reload the serving layer by calling POST /admin/reload.
    Non-fatal: if the API is unreachable the pipeline still finishes.
    """
    import requests

    api_url = os.getenv("API_URL", "http://localhost:8000")
    ti      = context["ti"]

    try:
        resp = requests.post(f"{api_url}/admin/reload", timeout=60)
        resp.raise_for_status()
        body   = resp.json()
        loaded = body.get("loaded", [])
        failed = body.get("failed", [])
        logger.info(f"API reloaded: {loaded}")
        if failed:
            logger.warning(f"API failed to reload: {failed}")
        ti.xcom_push(key="reloaded_models", value=loaded)

    except requests.exceptions.ConnectionError:
        logger.warning(
            f"API at {api_url} not reachable — skipping hot-reload. "
            "Models will be picked up on next API restart."
        )
        ti.xcom_push(key="reloaded_models", value=[])

    except Exception as exc:
        logger.warning(f"API reload failed: {exc}")
        ti.xcom_push(key="reloaded_models", value=[])


# ── Monitoring ────────────────────────────────────────────────────────────────

def task_check_drift(**context) -> None:
    from src.monitoring.drift import (
        load_reference_data, load_current_data, run_data_drift_report,
    )
    ti        = context["ti"]
    reference = load_reference_data()
    current   = load_current_data()
    result    = run_data_drift_report(reference, current)

    logger.info(
        f"Drift score: {result['drift_score']:.3f} | "
        f"Detected: {result['drift_detected']} | "
        f"Needs retrain: {result['needs_retraining']}"
    )
    ti.xcom_push(key="needs_retraining", value=result["needs_retraining"])
    ti.xcom_push(key="drift_score",      value=result["drift_score"])
    ti.xcom_push(key="drift_detected",   value=result["drift_detected"])


def task_branch_on_drift(**context) -> str:
    """
    BranchPythonOperator callable.
    Returns the task_id of the branch to follow.
    """
    ti            = context["ti"]
    needs_retrain = ti.xcom_pull(task_ids="check_drift", key="needs_retraining")
    if needs_retrain:
        logger.info("Drift detected — routing to trigger_retrain")
        return "trigger_retrain"
    logger.info("No significant drift — routing to skip_retrain")
    return "skip_retrain"


# ── Reporting ─────────────────────────────────────────────────────────────────

def task_send_report(**context) -> None:
    from src.notifications.report import send_pipeline_report
    send_pipeline_report(context)


# ── Failure callback ──────────────────────────────────────────────────────────

def on_failure(context: dict) -> None:
    from src.notifications.slack import send_task_failure
    send_task_failure(context)
