"""
Pipeline run report aggregator.

Pulls XCom values from all upstream tasks, builds a unified summary dict,
logs it to MLflow as a run tag set, and dispatches to Slack.
"""

import os
import time
from loguru import logger


def build_summary(context: dict) -> dict:
    """
    Collect all XCom values pushed by upstream tasks and return a summary dict.

    XCom keys consumed:
      version_data          → raw_data_hash, raw_data_commit
      compare_models        → best_model_type, best_run_id, best_val_loss
      register_best_model   → registry_result  {model_type, version, promoted, stage}
      deploy_to_api         → reloaded_models
      check_drift           → needs_retraining
      drift_score           (pulled directly from the drift module result via XCom)
    """
    ti = context["ti"]

    def _pull(task_id: str, key: str, default=None):
        try:
            return ti.xcom_pull(task_ids=task_id, key=key) or default
        except Exception:
            return default

    registry  = _pull("register_best_model", "registry_result") or {}
    reloaded  = _pull("deploy_to_api",       "reloaded_models") or []
    dag_run   = context.get("dag_run")

    # Compute wall-clock duration from dag_run start time
    duration = None
    if dag_run and hasattr(dag_run, "start_date") and dag_run.start_date:
        try:
            import pendulum
            duration = (pendulum.now("UTC") - dag_run.start_date).total_seconds()
        except Exception:
            pass

    summary = {
        "dag_id":           context.get("dag").dag_id if context.get("dag") else "stock_forecast_pipeline",
        "run_id":           dag_run.run_id if dag_run else None,
        "execution_date":   str(context.get("execution_date", "")),
        "duration_seconds": duration,
        "data_hash":        _pull("version_data", "raw_data_hash"),
        "data_commit":      _pull("version_data", "raw_data_commit"),
        "winner_model":     _pull("compare_models", "best_model_type"),
        "winner_run_id":    _pull("compare_models", "best_run_id"),
        "winner_val_loss":  _pull("compare_models", "best_val_loss"),
        "winner_rmse":      None,   # populated below if registry has metrics
        "winner_version":   registry.get("version"),
        "promoted":         registry.get("promoted"),
        "registry_stage":   registry.get("stage"),
        "models_reloaded":  reloaded,
        "needs_retraining": _pull("check_drift", "needs_retraining", False),
        "drift_score":      _pull("check_drift", "drift_score"),
        "tasks_failed":     [],
    }

    # Try to enrich with fresh RMSE from the registry
    try:
        from src.registry.model_registry import get_latest_production
        model_type = summary["winner_model"]
        if model_type:
            info = get_latest_production(model_type)
            if info:
                summary["winner_rmse"] = info.get("metrics", {}).get("test_rmse")
    except Exception:
        pass

    return summary


def log_summary_to_mlflow(summary: dict) -> None:
    """Log pipeline summary as tags on a dedicated MLflow run."""
    try:
        import mlflow
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "stock_price_forecasting"))
        with mlflow.start_run(run_name="Pipeline_Report"):
            tags = {k: str(v) for k, v in summary.items() if v is not None}
            mlflow.set_tags(tags)
            if summary.get("winner_rmse"):
                mlflow.log_metric("winner_rmse", float(summary["winner_rmse"]))
            if summary.get("drift_score") is not None:
                mlflow.log_metric("drift_score", float(summary["drift_score"]))
            if summary.get("duration_seconds"):
                mlflow.log_metric("pipeline_duration_seconds", float(summary["duration_seconds"]))
        logger.info("Pipeline summary logged to MLflow")
    except Exception as exc:
        logger.warning(f"Could not log summary to MLflow: {exc}")


def send_pipeline_report(context: dict) -> dict:
    """
    Build, log, and dispatch the pipeline run report.
    Called by the Airflow send_report task.
    Returns the summary dict.
    """
    from src.notifications.slack import send_pipeline_summary

    summary = build_summary(context)

    logger.info("Pipeline run summary:")
    for k, v in summary.items():
        logger.info(f"  {k}: {v}")

    log_summary_to_mlflow(summary)
    send_pipeline_summary(summary)

    return summary
