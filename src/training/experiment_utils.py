"""
MLflow experiment tracking utilities shared by all 4 trainers.

Functions:
  get_git_sha()             — current git commit SHA
  tag_current_run()         — add git SHA + data hash + custom tags to active run
  make_model_signature()    — infer MLflow input/output schema from example arrays
  log_prediction_plot()     — save actual-vs-predicted PNG as artifact
  log_metrics_table()       — save per-symbol metrics CSV as artifact
  get_best_run()            — programmatically query best finished run
"""

import io
import os
import subprocess
import tempfile

import numpy as np
import pandas as pd
from loguru import logger


# ── Git helpers ───────────────────────────────────────────────────────────────

def get_git_sha() -> str:
    """Return the current git HEAD SHA (7 chars), or 'unknown'."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return sha
    except Exception:
        return "unknown"


# ── Run tagging ───────────────────────────────────────────────────────────────

def tag_current_run(
    extra_tags: dict | None = None,
    data_hash:  str | None = None,
    data_commit: str | None = None,
):
    """
    Add standardised tags to the currently active MLflow run.

    Tags logged:
      git_sha       — links model to source code version
      data_hash     — DVC content hash of training data (if provided)
      data_commit   — git SHA when data was versioned (if provided)
      + any extra_tags supplied by the caller
    """
    import mlflow

    tags = {"git_sha": get_git_sha()}
    if data_hash:
        tags["data_hash"]   = data_hash
    if data_commit:
        tags["data_commit"] = data_commit
    if extra_tags:
        tags.update(extra_tags)

    mlflow.set_tags(tags)
    logger.debug(f"Run tagged: {tags}")


# ── Model signature ───────────────────────────────────────────────────────────

def make_model_signature(X_sample: np.ndarray, y_sample: np.ndarray):
    """
    Infer an MLflow ModelSignature from example numpy arrays.

    This signature is stored in the MLflow artifact and lets the Model
    Registry validate inputs at serving time.

    Returns:
        mlflow.models.ModelSignature or None if inference fails.
    """
    try:
        from mlflow.models import infer_signature
        return infer_signature(X_sample, y_sample)
    except Exception as exc:
        logger.warning(f"Could not infer model signature: {exc}")
        return None


# ── Artifact: prediction plot ─────────────────────────────────────────────────

def log_prediction_plot(
    y_true:        np.ndarray,
    y_pred:        np.ndarray,
    title:         str = "Actual vs Predicted",
    artifact_path: str = "plots",
    max_samples:   int = 200,
):
    """
    Save a line chart comparing actual vs predicted values as a PNG artifact.

    Args:
        y_true:        Ground-truth values, shape (n_samples,) or (n_samples, horizon)
        y_pred:        Predicted values, same shape
        title:         Chart title
        artifact_path: MLflow artifact sub-directory
        max_samples:   Truncate to first N points for readability
    """
    import mlflow
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend — safe in Airflow/CI
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true).flatten()[:max_samples]
    y_pred = np.asarray(y_pred).flatten()[:max_samples]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), tight_layout=True)

    # Top panel: overlay
    axes[0].plot(y_true, label="Actual",    color="steelblue",  linewidth=1.2)
    axes[0].plot(y_pred, label="Predicted", color="darkorange", linewidth=1.2,
                 linestyle="--")
    axes[0].set_title(title)
    axes[0].set_xlabel("Time step")
    axes[0].set_ylabel("Normalised price")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Bottom panel: residuals
    residuals = y_true - y_pred
    axes[1].bar(range(len(residuals)), residuals, color="grey", alpha=0.6)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Residuals (actual − predicted)")
    axes[1].set_xlabel("Time step")
    axes[1].set_ylabel("Error")
    axes[1].grid(True, alpha=0.3)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        fig.savefig(tmp.name, dpi=120, bbox_inches="tight")
        tmp_path = tmp.name

    plt.close(fig)
    mlflow.log_artifact(tmp_path, artifact_path=artifact_path)
    os.unlink(tmp_path)
    logger.debug(f"Prediction plot logged to {artifact_path}/")


# ── Artifact: per-symbol metrics table ───────────────────────────────────────

def log_metrics_table(
    per_symbol: dict,
    artifact_path: str = "metrics",
    filename: str = "per_symbol_metrics.csv",
):
    """
    Log a CSV table of per-symbol metrics as an MLflow artifact.

    Args:
        per_symbol: {symbol: {"mse": ..., "mae": ..., "rmse": ..., "mape": ...}}
        artifact_path: MLflow artifact sub-directory
        filename:   CSV file name
    """
    import mlflow

    rows = []
    for symbol, m in per_symbol.items():
        row = {"symbol": symbol}
        row.update(m)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("symbol")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as tmp:
        df.to_csv(tmp, index=False)
        tmp_path = tmp.name

    mlflow.log_artifact(tmp_path, artifact_path=artifact_path)
    os.unlink(tmp_path)
    logger.debug(f"Per-symbol metrics table logged to {artifact_path}/{filename}")
    return df


# ── Query best run ────────────────────────────────────────────────────────────

def get_best_run(
    experiment_name: str,
    metric:          str  = "val_loss",
    model_type:      str | None = None,
    max_results:     int  = 1,
) -> dict | None:
    """
    Return the best finished run from an MLflow experiment.

    Args:
        experiment_name: Name of the MLflow experiment
        metric:          Metric to minimise (lower = better)
        model_type:      If set, filter to runs with this model_type param
        max_results:     How many top runs to return (returns a list if > 1)

    Returns:
        dict with run info, or None if no finished runs found.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    client     = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        logger.warning(f"Experiment '{experiment_name}' not found")
        return None

    filter_str = "attributes.status = 'FINISHED'"
    if model_type:
        filter_str += f" AND params.model_type = '{model_type}'"

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_str,
        order_by=[f"metrics.{metric} ASC"],
        max_results=max_results,
    )

    if not runs:
        logger.warning(f"No finished runs found in '{experiment_name}'")
        return None

    best = runs[0]
    return {
        "run_id":     best.info.run_id,
        "model_type": best.data.params.get("model_type"),
        "val_loss":   best.data.metrics.get("val_loss"),
        "params":     dict(best.data.params),
        "metrics":    dict(best.data.metrics),
        "tags":       dict(best.data.tags),
    }


# ── Run summary table ─────────────────────────────────────────────────────────

def get_experiment_summary(experiment_name: str) -> pd.DataFrame:
    """
    Return a DataFrame summarising all finished runs in an experiment.
    Useful for comparing runs programmatically without opening the UI.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    client     = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return pd.DataFrame()

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["metrics.val_loss ASC"],
    )

    rows = []
    for run in runs:
        row = {
            "run_id":     run.info.run_id[:8],
            "model_type": run.data.params.get("model_type", "?"),
            "git_sha":    run.data.tags.get("git_sha", "?"),
        }
        for metric in ("val_loss", "val_mae", "test_rmse", "test_mape",
                       "test_directional_accuracy"):
            row[metric] = run.data.metrics.get(metric)
        rows.append(row)

    return pd.DataFrame(rows)
