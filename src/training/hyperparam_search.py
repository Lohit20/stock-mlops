"""
Hyperparameter search using MLflow nested runs.

Each search creates one parent MLflow run containing N child runs
(one per config). The best child run's config is returned.

UI: In MLflow → Experiments → click the parent run → expand child runs
    to see the full search history in a single view.

Usage:
    from src.training.hyperparam_search import search_lstm, search_arm

    best_lstm_config = search_lstm()
    best_arm_config  = search_arm()
"""

import os
import numpy as np
import pandas as pd
from loguru import logger
from dotenv import load_dotenv

from src.training.metrics import compute_regression_metrics
from src.training.experiment_utils import tag_current_run

load_dotenv()

FEATURES_PATH       = os.getenv("FEATURES_DATA_PATH", "data/features")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME     = os.getenv("MLFLOW_EXPERIMENT_NAME", "stock_price_forecasting")

# ── Search spaces ─────────────────────────────────────────────────────────────

LSTM_SEARCH_SPACE = [
    {
        "n_steps_in": 60,  "n_steps_out": 30, "window": 30,
        "batch_size": 32,  "epochs": 20,       "val_split": 0.1,
        "feature_cols": ["close", "returns", "rsi_14", "volatility_7"],
    },
    {
        "n_steps_in": 90,  "n_steps_out": 30, "window": 30,
        "batch_size": 32,  "epochs": 20,       "val_split": 0.1,
        "feature_cols": ["close", "returns", "rsi_14", "volatility_7"],
    },
    {
        "n_steps_in": 90,  "n_steps_out": 30, "window": 30,
        "batch_size": 64,  "epochs": 20,       "val_split": 0.1,
        "feature_cols": ["close", "returns", "rsi_14", "volatility_7",
                         "sma_7", "sma_30", "ema_7", "ema_30", "bb_width"],
    },
    {
        "n_steps_in": 120, "n_steps_out": 30, "window": 30,
        "batch_size": 64,  "epochs": 20,       "val_split": 0.1,
        "feature_cols": ["close", "returns", "rsi_14", "volatility_7",
                         "sma_7", "sma_30", "ema_7", "ema_30", "bb_width",
                         "momentum_7", "momentum_30"],
    },
]

ARM_SEARCH_SPACE = [
    {"p": 2, "d": 1, "q": 0},
    {"p": 5, "d": 1, "q": 0},
    {"p": 5, "d": 1, "q": 1},
    {"p": 2, "d": 1, "q": 2},
    {"p": 3, "d": 1, "q": 1},
]


# ── LSTM HP search ────────────────────────────────────────────────────────────

def search_lstm(
    search_space: list | None = None,
    data_hash:    str  | None = None,
    data_commit:  str  | None = None,
) -> dict:
    """
    Grid search over LSTM hyperparameters using nested MLflow runs.

    Args:
        search_space: List of config dicts. Defaults to LSTM_SEARCH_SPACE.
        data_hash:    DVC content hash of training data (for tagging).
        data_commit:  Git SHA when data was versioned (for tagging).

    Returns:
        Dict with best config and its val_loss.
    """
    import mlflow

    search_space = search_space or LSTM_SEARCH_SPACE

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    results = []

    with mlflow.start_run(run_name="LSTM_HP_Search") as parent_run:
        tag_current_run(
            extra_tags={"search_type": "grid", "n_trials": str(len(search_space))},
            data_hash=data_hash,
            data_commit=data_commit,
        )
        mlflow.log_param("n_trials", len(search_space))

        for i, config in enumerate(search_space):
            run_name = f"LSTM_trial_{i+1}_bs{config['batch_size']}_in{config['n_steps_in']}"
            logger.info(f"Trial {i+1}/{len(search_space)}: {run_name}")

            try:
                with mlflow.start_run(run_name=run_name, nested=True) as child_run:
                    tag_current_run(
                        extra_tags={
                            "trial_index":   str(i),
                            "parent_run_id": parent_run.info.run_id,
                        },
                        data_hash=data_hash,
                        data_commit=data_commit,
                    )

                    # Use the main train function — it logs params/metrics/model
                    from src.training.train_lstm import train
                    result = train(**config)

                    results.append({
                        "run_id":   child_run.info.run_id,
                        "val_loss": result["val_loss"],
                        "config":   config,
                    })

            except Exception as exc:
                logger.warning(f"Trial {i+1} failed: {exc}")

        if not results:
            raise RuntimeError("All LSTM HP search trials failed")

        # Find and log best config on parent run
        best = min(results, key=lambda r: r["val_loss"])
        mlflow.log_metrics({
            "best_val_loss":   best["val_loss"],
            "n_successful_trials": len(results),
        })
        mlflow.log_params({f"best_{k}": str(v) for k, v in best["config"].items()})
        logger.info(
            f"LSTM HP Search complete. Best val_loss={best['val_loss']:.4f} "
            f"config={best['config']}"
        )

    return best


# ── ARM HP search ─────────────────────────────────────────────────────────────

def search_arm(
    search_space: list | None = None,
    n_steps_out:  int  = 30,
    test_split:   float = 0.15,
    data_hash:    str  | None = None,
    data_commit:  str  | None = None,
) -> dict:
    """
    Grid search over ARIMA (p,d,q) orders using nested MLflow runs.

    Args:
        search_space: List of {"p", "d", "q"} dicts. Defaults to ARM_SEARCH_SPACE.
        n_steps_out:  Forecast horizon.
        test_split:   Fraction of data used for evaluation.
        data_hash:    DVC content hash (for tagging).
        data_commit:  Git SHA when data was versioned (for tagging).

    Returns:
        Dict with best (p,d,q) order and its avg val_loss.
    """
    import mlflow

    search_space = search_space or ARM_SEARCH_SPACE

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    results = []

    with mlflow.start_run(run_name="ARM_HP_Search") as parent_run:
        tag_current_run(
            extra_tags={
                "search_type": "grid",
                "n_trials":    str(len(search_space)),
                "model_type":  "ARM",
            },
            data_hash=data_hash,
            data_commit=data_commit,
        )
        mlflow.log_param("n_trials", len(search_space))

        for i, order_cfg in enumerate(search_space):
            p, d, q = order_cfg["p"], order_cfg["d"], order_cfg["q"]
            run_name = f"ARM_trial_{i+1}_({p},{d},{q})"
            logger.info(f"Trial {i+1}/{len(search_space)}: ARIMA({p},{d},{q})")

            try:
                with mlflow.start_run(run_name=run_name, nested=True) as child_run:
                    tag_current_run(
                        extra_tags={
                            "trial_index":   str(i),
                            "parent_run_id": parent_run.info.run_id,
                        },
                        data_hash=data_hash,
                        data_commit=data_commit,
                    )

                    from src.training.train_arm import train
                    result = train(
                        p=p, d=d, q=q,
                        n_steps_out=n_steps_out,
                        test_split=test_split,
                    )

                    results.append({
                        "run_id":   child_run.info.run_id,
                        "val_loss": result["val_loss"],
                        "config":   order_cfg,
                    })

            except Exception as exc:
                logger.warning(f"Trial {i+1} failed: {exc}")

        if not results:
            raise RuntimeError("All ARM HP search trials failed")

        best = min(results, key=lambda r: r["val_loss"])
        mlflow.log_metrics({
            "best_val_loss":        best["val_loss"],
            "n_successful_trials":  len(results),
        })
        mlflow.log_params({f"best_{k}": str(v) for k, v in best["config"].items()})
        logger.info(
            f"ARM HP Search complete. Best val_loss={best['val_loss']:.4f} "
            f"ARIMA({best['config']['p']},{best['config']['d']},{best['config']['q']})"
        )

    return best


# ── CLI entry points ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run hyperparameter search")
    parser.add_argument("model", choices=["lstm", "arm", "all"],
                        help="Which model to search")
    args = parser.parse_args()

    if args.model in ("lstm", "all"):
        best = search_lstm()
        print(f"\nBest LSTM config: {best['config']}  val_loss={best['val_loss']:.4f}")

    if args.model in ("arm", "all"):
        best = search_arm()
        print(f"\nBest ARM config: {best['config']}  val_loss={best['val_loss']:.4f}")
