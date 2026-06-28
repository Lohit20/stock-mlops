"""
Slack notification helpers for Airflow task callbacks and pipeline summaries.

Reads SLACK_WEBHOOK_URL from the environment. All functions are no-ops when
the variable is not set so the pipeline runs fine without Slack configured.
"""

import json
import os
from datetime import datetime, timezone

import requests
from loguru import logger

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
SLACK_TIMEOUT     = int(os.getenv("SLACK_TIMEOUT_SECONDS", "10"))


def _post(payload: dict) -> bool:
    """Send a JSON payload to the configured Slack webhook. Returns True on success."""
    if not SLACK_WEBHOOK_URL:
        logger.debug("SLACK_WEBHOOK_URL not set — Slack notification skipped")
        return False
    try:
        resp = requests.post(
            SLACK_WEBHOOK_URL,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=SLACK_TIMEOUT,
        )
        resp.raise_for_status()
        return True
    except Exception as exc:
        logger.warning(f"Slack notification failed: {exc}")
        return False


def send_alert(message: str, level: str = "warning") -> bool:
    """Send a plain-text alert to Slack."""
    emoji = {"info": ":information_source:", "warning": ":warning:", "error": ":red_circle:"}.get(level, ":bell:")
    return _post({"text": f"{emoji} *MLOps Alert* — {message}"})


def send_task_failure(context: dict) -> None:
    """Airflow on_failure_callback — posts task failure details to Slack."""
    ti      = context.get("task_instance")
    dag_id  = context.get("dag").dag_id if context.get("dag") else "unknown"
    task_id = ti.task_id if ti else "unknown"
    exec_dt = str(context.get("execution_date", ""))
    exc     = str(context.get("exception", ""))

    _post({
        "text": (
            f":red_circle: *Task Failed* — `{dag_id}.{task_id}`\n"
            f"Execution: `{exec_dt}`\n"
            f"Error: ```{exc[:500]}```"
        )
    })


def send_pipeline_summary(summary: dict) -> bool:
    """
    Send a structured pipeline run summary to Slack.

    Expected keys in summary:
        dag_id, run_id, execution_date, duration_seconds,
        winner_model, winner_version, winner_rmse,
        drift_score, drift_detected, needs_retraining,
        models_reloaded, tasks_failed
    """
    dag_id      = summary.get("dag_id", "stock_forecast_pipeline")
    winner      = summary.get("winner_model", "unknown")
    version     = summary.get("winner_version", "?")
    rmse        = summary.get("winner_rmse")
    drift       = summary.get("drift_score")
    retrain     = summary.get("needs_retraining", False)
    duration    = summary.get("duration_seconds")
    failed      = summary.get("tasks_failed", [])
    reloaded    = summary.get("models_reloaded", [])

    rmse_str    = f"{rmse:.4f}" if rmse else "N/A"
    drift_str   = f"{drift:.3f}" if drift is not None else "N/A"
    dur_str     = f"{int(duration // 60)}m {int(duration % 60)}s" if duration else "N/A"
    status_icon = ":white_check_mark:" if not failed else ":warning:"
    retrain_str = ":arrows_counterclockwise: Retraining scheduled" if retrain else ":heavy_check_mark: No retraining needed"

    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"{status_icon} Pipeline Complete — {dag_id}"},
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Winner Model:*\n{winner} v{version}"},
                {"type": "mrkdwn", "text": f"*Test RMSE:*\n{rmse_str}"},
                {"type": "mrkdwn", "text": f"*Drift Score:*\n{drift_str}"},
                {"type": "mrkdwn", "text": f"*Duration:*\n{dur_str}"},
            ],
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": retrain_str},
        },
    ]

    if failed:
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f":warning: *Failed tasks:* {', '.join(failed)}"},
        })

    if reloaded:
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f":rocket: *API reloaded:* {', '.join(reloaded)}"},
        })

    blocks.append({"type": "divider"})

    return _post({"blocks": blocks})
