"""
Model Comparison — Step 8.

Two-phase comparison:
  Phase 1 (historical) — query best metrics from each model's training run
  Phase 2 (fresh)      — reload registered models and evaluate on current data
                         (ARM + TimesFM/ETS always; LSTM when available)

A dedicated "Model_Comparison" MLflow run is created that contains:
  • Aggregated metric table (val_loss / RMSE / MAPE / directional accuracy)
  • Comparison bar charts
  • Per-symbol heatmap + win-count chart
  • Model card JSON for the winner
  • Promotion of the winner to Production stage

Entry point (called by Airflow task_compare_models):
    run_comparison() → dict (winner info)
"""

import os
import mlflow
import pandas as pd
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME     = os.getenv("MLFLOW_EXPERIMENT_NAME", "stock_price_forecasting")
FEATURES_PATH       = os.getenv("FEATURES_DATA_PATH", "data/features")

_MODEL_TYPES = ["LSTM", "TFT", "TimesFM", "ARM"]
_METRIC_COLS = [
    "val_loss", "val_mae", "test_rmse", "test_mape",
    "test_directional_accuracy",
]


# ── Phase 1: historical metrics from MLflow ───────────────────────────────────

def get_best_runs() -> pd.DataFrame:
    """
    Fetch the best finished run per model type from MLflow.
    Returns a DataFrame with one row per model type.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError(
            f"Experiment '{EXPERIMENT_NAME}' not found. "
            "Run at least one training script first."
        )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["metrics.val_loss ASC"],
    )

    records = []
    for run in runs:
        model_type = run.data.params.get("model_type", "unknown")
        if model_type == "unknown":
            continue
        record = {
            "run_id":     run.info.run_id,
            "model_type": model_type,
            "start_time": run.info.start_time,
            "git_sha":    run.data.tags.get("git_sha", ""),
            "data_hash":  run.data.tags.get("data_hash", ""),
        }
        for col in _METRIC_COLS:
            record[col] = float(run.data.metrics.get(col, float("inf")))
        records.append(record)

    if not records:
        raise ValueError("No finished runs found in MLflow")

    df   = pd.DataFrame(records)
    best = (
        df.sort_values("val_loss")
          .groupby("model_type", sort=False)
          .first()
          .reset_index()
    )
    return best


# ── Phase 2: fresh evaluation ─────────────────────────────────────────────────

def run_fresh_phase(n_steps_in: int = 90, n_steps_out: int = 30) -> tuple:
    """
    Run fresh evaluation using the evaluator module.
    Returns (per_symbol_df, agg_df) or (empty, empty) on failure.
    """
    try:
        from src.evaluation.evaluator import run_fresh_evaluation
        per_symbol_df, agg_df = run_fresh_evaluation(
            n_steps_in=n_steps_in,
            n_steps_out=n_steps_out,
            features_path=FEATURES_PATH,
        )
        return per_symbol_df, agg_df
    except Exception as exc:
        logger.warning(f"Fresh evaluation skipped: {exc}")
        return pd.DataFrame(), pd.DataFrame()


# ── Rank and print ────────────────────────────────────────────────────────────

def rank_models(
    historical_df: pd.DataFrame,
    fresh_agg_df:  pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Merge historical + fresh metrics into a single ranking table.

    Primary ranking: test_rmse from fresh evaluation (when available).
    Fallback:        val_loss from historical MLflow run.

    Returns (ranked_df, best_dict).
    """
    # Start from historical best runs
    ranked = historical_df.copy()

    # Overlay fresh RMSE/MAPE/dir_acc when available
    if not fresh_agg_df.empty:
        fresh_cols = ["model_type", "rmse", "mape", "directional_accuracy"]
        fresh_sub  = fresh_agg_df[
            [c for c in fresh_cols if c in fresh_agg_df.columns]
        ].rename(columns={
            "rmse": "fresh_rmse",
            "mape": "fresh_mape",
            "directional_accuracy": "fresh_dir_acc",
        })
        ranked = ranked.merge(fresh_sub, on="model_type", how="left")

    # Sort: prefer fresh_rmse if available, else val_loss
    if "fresh_rmse" in ranked.columns and ranked["fresh_rmse"].notna().any():
        ranked = ranked.sort_values("fresh_rmse", na_position="last")
        sort_metric = "fresh_rmse"
    else:
        ranked = ranked.sort_values("val_loss")
        sort_metric = "val_loss"

    ranked = ranked.reset_index(drop=True)
    ranked["rank"] = range(1, len(ranked) + 1)

    # Print comparison table
    print("\n" + "=" * 90)
    print("Model Comparison — Step 8")
    print("=" * 90)
    display_cols = [
        c for c in [
            "rank", "model_type", "val_loss", "test_rmse", "test_mape",
            "fresh_rmse", "fresh_mape", "fresh_dir_acc",
            "test_directional_accuracy", "git_sha", "data_hash",
        ] if c in ranked.columns
    ]
    print(ranked[display_cols].to_string(index=False, float_format="{:.4f}".format))
    print("=" * 90)

    best_row = ranked.iloc[0].to_dict()
    logger.info(
        f"Winner: {best_row['model_type']}  "
        f"({sort_metric}={best_row.get(sort_metric, '?'):.4f})  "
        f"run_id={best_row.get('run_id', '')[:8]}"
    )
    return ranked, best_row


# ── MLflow comparison run ─────────────────────────────────────────────────────

def log_comparison_run(
    ranked_df:     pd.DataFrame,
    per_symbol_df: pd.DataFrame,
    fresh_agg_df:  pd.DataFrame,
    best_info:     dict,
):
    """
    Create a dedicated MLflow run for the comparison results.
    Logs aggregated metrics, charts, per-symbol table, and model card.
    """
    from src.evaluation.report import log_all_artifacts
    from src.training.experiment_utils import tag_current_run

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="Model_Comparison"):
        mlflow.log_params({
            "n_models_compared":   len(ranked_df),
            "winner_model_type":   best_info.get("model_type"),
            "winner_run_id":       best_info.get("run_id", "")[:16],
            "fresh_eval_symbols":  len(per_symbol_df["symbol"].unique())
                                   if not per_symbol_df.empty else 0,
        })
        tag_current_run(extra_tags={"step": "model_comparison"})

        # Log scalar summary metrics for the winner
        for key in ("val_loss", "test_rmse", "test_mape",
                    "fresh_rmse", "fresh_mape", "fresh_dir_acc"):
            val = best_info.get(key)
            if val is not None and not (isinstance(val, float) and val != val):
                mlflow.log_metric(f"winner_{key}", float(val))

        # Use fresh_agg_df for charts if available, else build from ranked_df
        agg_for_report = fresh_agg_df if not fresh_agg_df.empty else (
            ranked_df.rename(columns={
                "test_rmse": "rmse", "test_mape": "mape",
                "test_directional_accuracy": "directional_accuracy",
            })
        )

        log_all_artifacts(per_symbol_df, agg_for_report, best_info)
        logger.info("Comparison run logged to MLflow")


# ── Model promotion ───────────────────────────────────────────────────────────

def promote_best_model(best: dict):
    """
    Transition the best model's registered version to Production.
    Archives any existing Production version first.
    """
    client     = mlflow.tracking.MlflowClient()
    model_type = str(best.get("model_type", "")).lower()
    model_name = f"stock_price_forecaster_{model_type}"
    run_id     = best.get("run_id", "")

    try:
        versions = client.search_model_versions(f"name='{model_name}'")
    except Exception as exc:
        logger.error(f"Could not search model versions for {model_name}: {exc}")
        return

    target = next((v for v in versions if v.run_id == run_id), None)
    if target is None:
        logger.warning(
            f"No registered version found for {model_name} run_id={run_id[:8]}. "
            "Model may not have been registered during training."
        )
        return

    # Archive current Production
    for v in versions:
        if getattr(v, "current_stage", None) == "Production":
            client.transition_model_version_stage(
                name=model_name, version=v.version, stage="Archived",
                archive_existing_versions=False,
            )
            logger.info(f"Archived {model_name} v{v.version}")

    client.transition_model_version_stage(
        name=model_name, version=target.version, stage="Production",
        archive_existing_versions=True,
    )
    logger.info(f"Promoted {model_name} v{target.version} to Production")


# ── Main entry point ──────────────────────────────────────────────────────────

def run_comparison(
    n_steps_in:  int = 90,
    n_steps_out: int = 30,
    skip_fresh:  bool = False,
) -> dict:
    """
    Full model comparison pipeline.

    Steps:
      1. Query best historical runs from MLflow
      2. Run fresh evaluation on current data (ARM + TimesFM always)
      3. Rank all models, merging both metric sources
      4. Log a dedicated comparison MLflow run with charts + model card
      5. Promote winner to Production

    Args:
        n_steps_in:  Context window for fresh evaluation.
        n_steps_out: Forecast horizon for fresh evaluation.
        skip_fresh:  If True, skip fresh evaluation (use historical only).

    Returns:
        Dict describing the winning model.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Phase 1: historical
    historical_df = get_best_runs()

    # Phase 2: fresh
    per_symbol_df, fresh_agg_df = (
        (pd.DataFrame(), pd.DataFrame()) if skip_fresh
        else run_fresh_phase(n_steps_in, n_steps_out)
    )

    # Rank
    ranked_df, best_info = rank_models(historical_df, fresh_agg_df)

    # Log comparison run to MLflow
    try:
        log_comparison_run(ranked_df, per_symbol_df, fresh_agg_df, best_info)
    except Exception as exc:
        logger.warning(f"Could not log comparison run: {exc}")

    # Promote winner
    promote_best_model(best_info)

    return best_info


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare and promote best model")
    parser.add_argument("--skip-fresh", action="store_true",
                        help="Use historical MLflow metrics only (no model loading)")
    args = parser.parse_args()

    winner = run_comparison(skip_fresh=args.skip_fresh)
    print(f"\nWinner: {winner['model_type']}")
