"""
Model Comparison.

What this file does:
- Loads all 4 trained models from MLflow
- Runs them on the same test data
- Compares their performance metrics
- Picks the best model and promotes it to Production

Think of it as a sports tournament:
All 4 models compete → best one wins → gets deployed
"""

import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from dotenv import load_dotenv
import os

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME     = os.getenv("MLFLOW_EXPERIMENT_NAME", "stock_price_forecasting")

MODEL_NAMES = [
    "stock_price_forecaster_lstm",
    "stock_price_forecaster_tft",
    "stock_price_forecaster_timesfm",
    "stock_price_forecaster_arm",
]


def get_latest_runs() -> pd.DataFrame:
    """
    Fetch all experiment runs from MLflow.
    Returns a DataFrame with one row per model run.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError(f"Experiment '{EXPERIMENT_NAME}' not found in MLflow")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
    )

    records = []
    for run in runs:
        records.append({
            "run_id":     run.info.run_id,
            "model_type": run.data.params.get("model_type", "unknown"),
            "val_loss":   float(run.data.metrics.get("val_loss", float("inf"))),
            "val_mae":    float(run.data.metrics.get("val_mae", float("inf"))),
            "train_loss": float(run.data.metrics.get("train_loss", float("inf"))),
            "status":     run.info.status,
        })

    return pd.DataFrame(records)


def pick_best_model(runs_df: pd.DataFrame) -> dict:
    """
    Select the best model based on lowest validation loss.

    Args:
        runs_df: DataFrame of all runs from get_latest_runs()

    Returns:
        Dictionary with best model info
    """
    # Filter only finished runs
    finished = runs_df[runs_df["status"] == "FINISHED"]
    if finished.empty:
        raise ValueError("No finished runs found in MLflow")

    # Best model = lowest validation loss
    best_idx   = finished["val_loss"].idxmin()
    best_model = finished.loc[best_idx]

    logger.info("\n=== Model Comparison ===")
    logger.info(runs_df[["model_type", "val_loss", "val_mae"]].to_string())
    logger.info(f"\n🏆 Best Model: {best_model['model_type']} (val_loss={best_model['val_loss']:.4f})")

    return best_model.to_dict()


def promote_best_model(best_model: dict):
    """
    Promote the best model to Production in MLflow Registry.

    MLflow model lifecycle:
    None → Staging → Production → Archived

    This function:
    1. Archives the current Production model
    2. Promotes the new best model to Production
    """
    client      = mlflow.tracking.MlflowClient()
    model_type  = best_model["model_type"].lower()
    model_name  = f"stock_price_forecaster_{model_type}"

    # Archive existing Production version
    try:
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        for v in prod_versions:
            client.transition_model_version_stage(
                name=model_name, version=v.version, stage="Archived"
            )
            logger.info(f"Archived {model_name} v{v.version}")
    except Exception:
        pass  # No existing Production version

    # Promote latest Staging version to Production
    staging_versions = client.get_latest_versions(model_name, stages=["Staging", "None"])
    if staging_versions:
        latest = staging_versions[0]
        client.transition_model_version_stage(
            name=model_name, version=latest.version, stage="Production"
        )
        logger.info(f"✅ Promoted {model_name} v{latest.version} to Production")
    else:
        logger.warning(f"No version found to promote for {model_name}")


def run_comparison():
    """Main comparison function — called by Airflow after all models finish training."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    runs_df    = get_latest_runs()
    best_model = pick_best_model(runs_df)
    promote_best_model(best_model)

    return best_model


if __name__ == "__main__":
    run_comparison()
