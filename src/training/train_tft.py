"""
TFT (Temporal Fusion Transformer) Training with MLflow.

Uses pytorch-forecasting's TemporalFusionTransformer, which handles
multi-series forecasting natively — one model trained on all symbols.
Falls back with a clear error if pytorch / pytorch-forecasting is missing.
"""

import os
import numpy as np
import pandas as pd
from loguru import logger
from dotenv import load_dotenv

from src.training.metrics import compute_regression_metrics
from src.training.experiment_utils import (
    tag_current_run,
    make_model_signature,
    log_prediction_plot,
)

load_dotenv()

FEATURES_PATH       = os.getenv("FEATURES_DATA_PATH", "data/features")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME     = os.getenv("MLFLOW_EXPERIMENT_NAME", "stock_price_forecasting")

_UNKNOWN_REALS = ["close", "returns", "volatility_7", "rsi_14", "bb_width", "momentum_7"]
_KNOWN_REALS   = ["sma_7", "sma_30", "ema_7", "ema_30"]


def _check_deps():
    missing = []
    for pkg in ("torch", "pytorch_forecasting", "pytorch_lightning", "lightning"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        raise ImportError(
            f"TFT requires: {', '.join(missing)}.\n"
            "Install with: pip install pytorch-forecasting pytorch-lightning"
        )


def prepare_dataset(df: pd.DataFrame, n_steps_in: int, n_steps_out: int):
    """
    Build TimeSeriesDataSet for TFT.
    Adds integer time_idx per symbol, then splits 80/20 train/val.
    """
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer

    df = df.copy()
    df.sort_values(["symbol", "timestamp"], inplace=True)

    # Integer time index required by TimeSeriesDataSet
    df["time_idx"] = df.groupby("symbol").cumcount()

    # Filter to available columns
    unk = [c for c in _UNKNOWN_REALS if c in df.columns]
    knw = [c for c in _KNOWN_REALS   if c in df.columns]

    # Symbols need at least encoder + prediction length rows
    min_rows = n_steps_in + n_steps_out + 5
    valid_symbols = (
        df.groupby("symbol")["time_idx"].count()
        .loc[lambda s: s >= min_rows]
        .index.tolist()
    )
    df = df[df["symbol"].isin(valid_symbols)].copy()
    logger.info(f"TFT: {len(valid_symbols)} valid symbols, {len(df):,} rows")

    max_time_idx = df.groupby("symbol")["time_idx"].max()
    cutoff       = int(max_time_idx.quantile(0.8))

    df_train = df[df["time_idx"] <= cutoff].copy()
    df_val   = df.copy()

    training = TimeSeriesDataSet(
        df_train,
        time_idx="time_idx",
        target="close",
        group_ids=["symbol"],
        max_encoder_length=n_steps_in,
        max_prediction_length=n_steps_out,
        static_categoricals=["symbol"],
        time_varying_known_reals=["time_idx"] + knw,
        time_varying_unknown_reals=unk,
        target_normalizer=GroupNormalizer(groups=["symbol"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    validation = TimeSeriesDataSet.from_dataset(training, df_val, predict=True)
    return training, validation


def train(
    max_epochs:          int        = 30,
    batch_size:          int        = 64,
    learning_rate:       float      = 0.001,
    hidden_size:         int        = 64,
    attention_head_size: int        = 4,
    dropout:             float      = 0.1,
    hidden_continuous_size: int     = 32,
    n_steps_in:          int        = 90,
    n_steps_out:         int        = 30,
    gradient_clip_val:   float      = 0.1,
    num_workers:         int        = 0,
    data_hash:           str | None = None,
    data_commit:         str | None = None,
):
    _check_deps()

    import torch
    import pytorch_lightning as pl
    from pytorch_forecasting import TemporalFusionTransformer
    from pytorch_forecasting.metrics import QuantileLoss
    import mlflow
    import mlflow.pytorch

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="TFT"):
        mlflow.log_params({
            "model_type":            "TFT",
            "max_epochs":            max_epochs,
            "batch_size":            batch_size,
            "learning_rate":         learning_rate,
            "hidden_size":           hidden_size,
            "attention_heads":       attention_head_size,
            "dropout":               dropout,
            "hidden_continuous_size": hidden_continuous_size,
            "n_steps_in":            n_steps_in,
            "n_steps_out":           n_steps_out,
            "gradient_clip_val":     gradient_clip_val,
        })

        # ── Load data ─────────────────────────────────────────────────────────
        data_path = os.path.join(FEATURES_PATH, "price_features.csv")
        df = pd.read_csv(data_path, parse_dates=["timestamp"])

        mlflow.log_params({"n_symbols": df["symbol"].nunique(), "data_rows": len(df)})

        # ── Build datasets ────────────────────────────────────────────────────
        training, validation = prepare_dataset(df, n_steps_in, n_steps_out)

        train_loader = training.to_dataloader(
            train=True, batch_size=batch_size, num_workers=num_workers
        )
        val_loader = validation.to_dataloader(
            train=False, batch_size=batch_size * 2, num_workers=num_workers
        )

        mlflow.log_param("n_sequences_train", len(training))

        # ── Build model ───────────────────────────────────────────────────────
        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            attention_head_size=attention_head_size,
            dropout=dropout,
            hidden_continuous_size=hidden_continuous_size,
            output_size=7,          # 7 quantiles
            loss=QuantileLoss(),
            log_interval=10,
            reduce_on_plateau_patience=4,
        )
        logger.info(f"TFT parameters: {tft.size():,}")

        # ── Train ─────────────────────────────────────────────────────────────
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="auto",
            gradient_clip_val=gradient_clip_val,
            enable_progress_bar=True,
            enable_model_summary=True,
            log_every_n_steps=1,
        )
        trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # ── Evaluate ──────────────────────────────────────────────────────────
        best_model_path = trainer.checkpoint_callback.best_model_path
        best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

        predictions = best_tft.predict(val_loader, return_y=True)
        y_pred_median = predictions.output[:, :, 3].cpu().numpy()  # median quantile
        y_true        = predictions.y[0].cpu().numpy()

        test_m = compute_regression_metrics(y_true, y_pred_median)

        val_loss = float(trainer.callback_metrics.get("val_loss", 0.0))
        mlflow.log_metrics({
            "train_loss": float(trainer.callback_metrics.get("train_loss_epoch", 0.0)),
            "val_loss":   val_loss,
            "train_mae":  0.0,
            "val_mae":    0.0,
            "test_mse":   test_m["mse"],
            "test_mae":   test_m["mae"],
            "test_rmse":  test_m["rmse"],
            "test_mape":  test_m["mape"],
            "test_directional_accuracy": test_m["directional_accuracy"],
        })

        logger.info(
            f"TFT | val_loss={val_loss:.4f} "
            f"test_rmse={test_m['rmse']:.4f} "
            f"test_mape={test_m['mape']:.2f}%"
        )

        # ── Tags + artifacts ─────────────────────────────────────────────────
        tag_current_run(
            extra_tags={"model_type": "TFT"},
            data_hash=data_hash,
            data_commit=data_commit,
        )
        try:
            log_prediction_plot(
                y_true.flatten(), y_pred_median.flatten(),
                title="TFT — Actual vs Predicted (validation set)",
            )
        except Exception as _exc:
            logger.warning(f"Could not log prediction plot: {_exc}")

        sig = make_model_signature(
            np.zeros((1, n_steps_in, 1), dtype="float32"),
            np.zeros((1, n_steps_out),   dtype="float32"),
        )

        # ── Register model ────────────────────────────────────────────────────
        mlflow.pytorch.log_model(
            best_tft,
            artifact_path="tft_model",
            registered_model_name="stock_price_forecaster_tft",
            signature=sig,
        )

        run_id = mlflow.active_run().info.run_id
        return {
            "run_id":       run_id,
            "val_loss":     val_loss,
            "test_metrics": test_m,
            "model_name":   "stock_price_forecaster_tft",
        }


if __name__ == "__main__":
    train()
