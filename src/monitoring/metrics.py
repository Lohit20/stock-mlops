"""
Centralised Prometheus metric definitions for the MLOps pipeline.

API-level metrics (request count, latency, cache hits) live in api.py.
This module owns *pipeline-level* metrics that are pushed by Airflow tasks
via POST /monitoring/push after each pipeline run.

Usage
-----
# From the API endpoint that accepts pipeline summaries:
from src.monitoring.metrics import push_drift_metrics, push_model_metrics

# From Grafana/Prometheus — all metrics are exposed on GET /metrics.
"""

from prometheus_client import Counter, Gauge


# ── Data drift ─────────────────────────────────────────────────────────────────

DRIFT_SCORE = Gauge(
    "pipeline_drift_score",
    "Latest data drift score (share of drifted columns) from Evidently",
    ["dataset"],
)

DRIFT_DETECTED = Gauge(
    "pipeline_drift_detected",
    "1 if dataset drift was detected in the last run, 0 otherwise",
    ["dataset"],
)

NEEDS_RETRAINING = Gauge(
    "pipeline_needs_retraining",
    "1 if drift score exceeded the retraining threshold, 0 otherwise",
    ["dataset"],
)


# ── Model performance ──────────────────────────────────────────────────────────

MODEL_RMSE = Gauge(
    "pipeline_model_rmse",
    "RMSE from the most recent evaluation run",
    ["model_type"],
)

MODEL_MAPE = Gauge(
    "pipeline_model_mape",
    "MAPE (%) from the most recent evaluation run",
    ["model_type"],
)

MODEL_DIR_ACCURACY = Gauge(
    "pipeline_model_directional_accuracy",
    "Directional accuracy (fraction correct) from the most recent evaluation run",
    ["model_type"],
)


# ── Pipeline lifecycle ─────────────────────────────────────────────────────────

PIPELINE_RUNS_TOTAL = Counter(
    "pipeline_runs_total",
    "Total completed pipeline runs (incremented at the send_report task)",
)

PIPELINE_DURATION = Gauge(
    "pipeline_duration_seconds",
    "Wall-clock duration of the last completed pipeline run in seconds",
)

RETRAINING_TRIGGERED_TOTAL = Counter(
    "pipeline_retraining_triggered_total",
    "Number of times drift detection triggered an automatic retraining run",
)

PIPELINE_WINNER_MODEL = Gauge(
    "pipeline_winner_model_info",
    "Info about the winning model — always 1; use labels for the name and version",
    ["model_type", "version"],
)


# ── Helper functions ───────────────────────────────────────────────────────────

def push_drift_metrics(drift_result: dict, dataset: str = "price_features") -> None:
    """
    Update drift Gauges from a run_data_drift_report() result dict.

    Expected keys: drift_score, drift_detected, needs_retraining
    """
    score    = drift_result.get("drift_score", 0.0) or 0.0
    detected = 1 if drift_result.get("drift_detected") else 0
    retrain  = 1 if drift_result.get("needs_retraining") else 0

    DRIFT_SCORE.labels(dataset=dataset).set(score)
    DRIFT_DETECTED.labels(dataset=dataset).set(detected)
    NEEDS_RETRAINING.labels(dataset=dataset).set(retrain)


def push_model_metrics(model_type: str, eval_metrics: dict) -> None:
    """
    Update model-performance Gauges from an evaluation metrics dict.

    Expected keys (all optional): test_rmse, test_mape, test_directional_accuracy
    """
    mt = model_type.upper()
    if "test_rmse" in eval_metrics and eval_metrics["test_rmse"] is not None:
        MODEL_RMSE.labels(model_type=mt).set(float(eval_metrics["test_rmse"]))
    if "test_mape" in eval_metrics and eval_metrics["test_mape"] is not None:
        MODEL_MAPE.labels(model_type=mt).set(float(eval_metrics["test_mape"]))
    if "test_directional_accuracy" in eval_metrics and eval_metrics["test_directional_accuracy"] is not None:
        MODEL_DIR_ACCURACY.labels(model_type=mt).set(
            float(eval_metrics["test_directional_accuracy"])
        )


def push_pipeline_run_metrics(summary: dict) -> None:
    """
    Update all pipeline lifecycle metrics from a send_pipeline_report() summary dict.

    Expected keys: drift_score, drift_detected, needs_retraining, winner_model,
                   winner_version, winner_rmse, duration_seconds
    """
    PIPELINE_RUNS_TOTAL.inc()

    if summary.get("duration_seconds"):
        PIPELINE_DURATION.set(float(summary["duration_seconds"]))

    push_drift_metrics({
        "drift_score":      summary.get("drift_score"),
        "drift_detected":   summary.get("drift_detected"),
        "needs_retraining": summary.get("needs_retraining"),
    })

    if summary.get("needs_retraining"):
        RETRAINING_TRIGGERED_TOTAL.inc()

    winner = summary.get("winner_model")
    version = str(summary.get("winner_version") or "unknown")
    if winner:
        PIPELINE_WINNER_MODEL.labels(model_type=winner.upper(), version=version).set(1)
        if summary.get("winner_rmse") is not None:
            MODEL_RMSE.labels(model_type=winner.upper()).set(float(summary["winner_rmse"]))
