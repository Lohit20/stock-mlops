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


# ── Monitoring + retraining decision ─────────────────────────────────────────

def task_check_drift(**context) -> None:
    """
    Run Evidently drift check then pass it through the retraining scheduler
    (adaptive threshold + cooldown guard) to produce a final should_retrain
    decision.  All results are pushed to XCom for downstream tasks.
    """
    from src.monitoring.drift import (
        load_reference_data, load_current_data, run_data_drift_report,
    )
    from src.retraining.scheduler import evaluate_retraining_need

    ti       = context["ti"]
    dag_run  = context.get("dag_run")
    run_id   = dag_run.run_id if dag_run else None

    reference = load_reference_data()
    current   = load_current_data()
    drift_res = run_data_drift_report(reference, current)

    decision = evaluate_retraining_need(drift_res, dag_run_id=run_id)

    logger.info(
        f"Drift score: {decision['drift_score']:.3f} | "
        f"Threshold: {decision['threshold_used']:.3f} | "
        f"Cooldown active: {decision['cooldown_active']} | "
        f"Should retrain: {decision['should_retrain']}"
    )

    ti.xcom_push(key="needs_retraining", value=decision["should_retrain"])
    ti.xcom_push(key="drift_score",      value=decision["drift_score"])
    ti.xcom_push(key="drift_detected",   value=decision["drift_detected"])
    ti.xcom_push(key="retrain_reason",   value=decision["reason"])
    ti.xcom_push(key="cooldown_active",  value=decision["cooldown_active"])
    ti.xcom_push(key="threshold_used",   value=decision["threshold_used"])
    ti.xcom_push(key="cooldown_status",  value=decision["cooldown_status"])


def task_branch_on_drift(**context) -> str:
    """
    BranchPythonOperator callable.
    Returns the task_id of the branch to follow.
    """
    ti            = context["ti"]
    needs_retrain = ti.xcom_pull(task_ids="check_drift", key="needs_retraining")
    reason        = ti.xcom_pull(task_ids="check_drift", key="retrain_reason") or ""

    if needs_retrain:
        logger.info(f"Routing to trigger_retrain: {reason}")
        return "trigger_retrain"
    logger.info(f"Routing to skip_retrain: {reason}")
    return "skip_retrain"


def task_record_retrain(**context) -> None:
    """
    Called immediately after trigger_retrain fires.
    Writes the cooldown timestamp and updates the drift history.
    Non-fatal — pipeline continues even if state write fails.
    """
    from src.retraining.scheduler import record_retrain_triggered

    ti      = context["ti"]
    dag_run = context.get("dag_run")
    run_id  = dag_run.run_id if dag_run else None
    score   = ti.xcom_pull(task_ids="check_drift", key="drift_score")

    try:
        record_retrain_triggered(drift_score=score, dag_run_id=run_id)
        logger.info(f"Retrain state recorded: score={score} run_id={run_id}")
    except Exception as exc:
        logger.warning(f"Could not record retrain state: {exc}")


# ── Reporting ─────────────────────────────────────────────────────────────────

def task_send_report(**context) -> None:
    import requests as _req
    from src.notifications.report import send_pipeline_report

    summary = send_pipeline_report(context)

    # Push the summary to the API so Prometheus gauges are updated immediately.
    api_url = os.getenv("API_URL", "http://localhost:8000")
    try:
        resp = _req.post(
            f"{api_url}/monitoring/push",
            json={k: v for k, v in summary.items() if v is not None},
            timeout=15,
        )
        resp.raise_for_status()
        logger.info("Monitoring metrics pushed to API")
    except Exception as exc:
        logger.warning(f"Could not push monitoring metrics: {exc}")


# ── Failure callback ──────────────────────────────────────────────────────────

def on_failure(context: dict) -> None:
    from src.notifications.slack import send_task_failure
    send_task_failure(context)
