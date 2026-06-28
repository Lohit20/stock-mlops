"""
TimesFM (Google) zero-shot inference with MLflow.

Strategy:
  1. Try to import `timesfm` and run Google's foundation model for zero-shot forecasting.
  2. If timesfm is not installed, fall back to ExponentialSmoothing (ETS) from statsmodels
     as a fast, interpretable "statistical foundation model" substitute.

TimesFM install:  pip install timesfm  (requires JAX or PyTorch backend)
"""

import os
import numpy as np
import pandas as pd
from loguru import logger
from dotenv import load_dotenv

from src.training.metrics import compute_regression_metrics
from src.training.experiment_utils import (
    tag_current_run,
    log_prediction_plot,
)

load_dotenv()

FEATURES_PATH       = os.getenv("FEATURES_DATA_PATH", "data/features")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME     = os.getenv("MLFLOW_EXPERIMENT_NAME", "stock_price_forecasting")

_TIMESFM_CHECKPOINT = "google/timesfm-1.0-200m-pytorch"


# ── TimesFM inference ─────────────────────────────────────────────────────────

def _run_timesfm(df: pd.DataFrame, context_len: int, horizon_len: int) -> tuple:
    """
    Run TimesFM zero-shot forecast on each symbol.
    Returns (y_true, y_pred) arrays concatenated across symbols.
    """
    import timesfm

    # Handle both timesfm v1 and v2 API
    try:
        tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="cpu",
                per_core_batch_size=32,
                horizon_len=horizon_len,
                context_len=context_len,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id=_TIMESFM_CHECKPOINT
            ),
        )
    except AttributeError:
        # v1 API
        tfm = timesfm.TimesFm(
            context_len=context_len,
            horizon_len=horizon_len,
            input_patch_len=32,
            output_patch_len=128,
            num_layers=20,
            model_dims=1280,
            backend="cpu",
        )
        tfm.load_from_checkpoint(repo_id=_TIMESFM_CHECKPOINT)

    all_true, all_pred = [], []

    for symbol, grp in df.groupby("symbol"):
        series = grp.sort_values("timestamp")["close"].dropna().values
        if len(series) < context_len + horizon_len:
            continue

        context  = series[-(context_len + horizon_len) : -horizon_len]
        y_true   = series[-horizon_len:]

        point_forecast, _ = tfm.forecast(
            [context],
            freq=[0],   # 0 = daily
        )
        y_pred = np.asarray(point_forecast[0])[:horizon_len]

        all_true.append(y_true)
        all_pred.append(y_pred)

    if not all_true:
        raise ValueError("No symbols had enough data for TimesFM inference")

    return np.vstack(all_true), np.vstack(all_pred)


# ── ETS fallback ──────────────────────────────────────────────────────────────

def _run_ets(df: pd.DataFrame, context_len: int, horizon_len: int) -> tuple:
    """
    ExponentialSmoothing (Holt-Winters) as a statistical fallback.
    Fits per-symbol, returns aggregated (y_true, y_pred).
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    all_true, all_pred = [], []

    for symbol, grp in df.groupby("symbol"):
        series = grp.sort_values("timestamp")["close"].dropna().values
        if len(series) < context_len + horizon_len:
            continue

        train = series[-(context_len + horizon_len) : -horizon_len]
        y_true = series[-horizon_len:]

        try:
            fit = ExponentialSmoothing(
                train,
                trend="add",
                seasonal=None,
                initialization_method="estimated",
            ).fit(optimized=True, remove_bias=True)
            y_pred = fit.forecast(horizon_len)
        except Exception as exc:
            logger.warning(f"ETS failed for {symbol}: {exc} — using naive forecast")
            y_pred = np.full(horizon_len, train[-1])

        all_true.append(y_true[:horizon_len])
        all_pred.append(np.asarray(y_pred)[:horizon_len])

    if not all_true:
        raise ValueError("No symbols had enough data for ETS inference")

    return np.vstack(all_true), np.vstack(all_pred)


# ── Main train function ───────────────────────────────────────────────────────

def train(
    context_len:   int        = 512,
    horizon_len:   int        = 30,
    fine_tune:     bool       = False,
    data_hash:     str | None = None,
    data_commit:   str | None = None,
):
    import mlflow
    import mlflow.pyfunc

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="TimesFM"):
        # ── Load data ─────────────────────────────────────────────────────────
        data_path = os.path.join(FEATURES_PATH, "price_features.csv")
        df = pd.read_csv(data_path, parse_dates=["timestamp"])
        df.sort_values(["symbol", "timestamp"], inplace=True)

        mlflow.log_params({
            "n_symbols":    df["symbol"].nunique(),
            "data_rows":    len(df),
            "context_len":  context_len,
            "horizon_len":  horizon_len,
            "fine_tune":    fine_tune,
        })

        # ── Try TimesFM, fall back to ETS ─────────────────────────────────────
        try:
            import timesfm  # noqa: F401
            logger.info("TimesFM package detected — running zero-shot inference")
            mlflow.log_param("backend", "timesfm")
            mlflow.log_param("checkpoint", _TIMESFM_CHECKPOINT)
            y_true, y_pred = _run_timesfm(df, context_len, horizon_len)
        except ImportError:
            logger.warning(
                "timesfm not installed — using ExponentialSmoothing fallback. "
                "Install with: pip install timesfm"
            )
            mlflow.log_param("backend", "ets_fallback")
            mlflow.log_param("checkpoint", "statsmodels.ExponentialSmoothing")
            y_true, y_pred = _run_ets(df, context_len, horizon_len)

        # ── Metrics ───────────────────────────────────────────────────────────
        test_m = compute_regression_metrics(y_true, y_pred)

        mlflow.log_metrics({
            "train_loss": 0.0,   # zero-shot: no training
            "val_loss":   test_m["mse"],
            "train_mae":  0.0,
            "val_mae":    test_m["mae"],
            "test_mse":   test_m["mse"],
            "test_mae":   test_m["mae"],
            "test_rmse":  test_m["rmse"],
            "test_mape":  test_m["mape"],
            "test_directional_accuracy": test_m["directional_accuracy"],
        })

        logger.info(
            f"TimesFM | test_rmse={test_m['rmse']:.4f} "
            f"test_mape={test_m['mape']:.2f}%"
        )

        # ── Tags + artifacts ─────────────────────────────────────────────────
        tag_current_run(
            extra_tags={"model_type": "TimesFM"},
            data_hash=data_hash,
            data_commit=data_commit,
        )
        try:
            log_prediction_plot(
                y_true.flatten(), y_pred.flatten(),
                title="TimesFM — Actual vs Predicted (zero-shot)",
            )
        except Exception as _exc:
            logger.warning(f"Could not log prediction plot: {_exc}")

        # ── Log predictions as artifact ───────────────────────────────────────
        pred_df = pd.DataFrame({
            "y_true": y_true.flatten(),
            "y_pred": y_pred.flatten(),
        })
        tmp_path = "/tmp/timesfm_predictions.csv"
        pred_df.to_csv(tmp_path, index=False)
        mlflow.log_artifact(tmp_path, artifact_path="predictions")

        run_id = mlflow.active_run().info.run_id
        return {
            "run_id":       run_id,
            "val_loss":     test_m["mse"],
            "test_metrics": test_m,
            "model_name":   "stock_price_forecaster_timesfm",
        }


if __name__ == "__main__":
    train()
