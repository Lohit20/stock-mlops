"""
ARM — AutoRegressive Model (ARIMA per symbol) Training with MLflow.

Fits one ARIMA(p,d,q) per symbol, computes aggregated metrics, then wraps
all fitted results in an MLflow pyfunc model so it can be served like LSTM/TFT.
"""

import os
import pickle
import tempfile
import numpy as np
import pandas as pd
import mlflow
import mlflow.pyfunc
from loguru import logger
from dotenv import load_dotenv

from src.training.metrics import compute_regression_metrics
from src.training.experiment_utils import (
    tag_current_run,
    make_model_signature,
    log_prediction_plot,
    log_metrics_table,
)

load_dotenv()

FEATURES_PATH       = os.getenv("FEATURES_DATA_PATH", "data/features")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME     = os.getenv("MLFLOW_EXPERIMENT_NAME", "stock_price_forecasting")


# ── MLflow pyfunc wrapper ─────────────────────────────────────────────────────

class ArimaWrapper(mlflow.pyfunc.PythonModel):
    """
    Wraps a dict of per-symbol fitted ARIMA result objects.

    predict() input DataFrame columns: symbol, close, timestamp
    predict() output: list of dicts with {symbol, forecast_day, predicted_close}
    """

    def __init__(self, fitted_models: dict, n_steps_out: int, order: tuple):
        self._models     = fitted_models
        self._n_out      = n_steps_out
        self._order      = order

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        if isinstance(model_input, dict):
            model_input = pd.DataFrame(model_input)

        requested = (
            model_input["symbol"].unique().tolist()
            if "symbol" in model_input.columns
            else list(self._models.keys())
        )

        rows = []
        for symbol in requested:
            result = self._models.get(symbol)
            if result is None:
                logger.warning(f"No ARIMA model for {symbol} — skipping")
                continue
            forecast = result.forecast(steps=self._n_out)
            for day, val in enumerate(forecast.values, start=1):
                rows.append({
                    "symbol":          symbol,
                    "forecast_day":    day,
                    "predicted_close": float(val),
                })

        return pd.DataFrame(rows)


# ── ARIMA fitting helpers ─────────────────────────────────────────────────────

def fit_arima(series: pd.Series, order: tuple):
    from statsmodels.tsa.arima.model import ARIMA
    return ARIMA(series, order=order).fit()


def evaluate_arima(result, test_series: pd.Series, n_steps_out: int) -> dict:
    forecast = result.forecast(steps=n_steps_out)
    y_true   = test_series.values
    y_pred   = forecast.values

    n = min(len(y_true), len(y_pred))
    return compute_regression_metrics(y_true[:n], y_pred[:n])


# ── Main train function ───────────────────────────────────────────────────────

def train(
    p:           int   = 5,
    d:           int   = 1,
    q:           int   = 0,
    n_steps_out: int   = 30,
    test_split:  float = 0.15,
    data_hash:   str | None = None,
    data_commit: str | None = None,
):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    order = (p, d, q)

    with mlflow.start_run(run_name="ARM"):
        mlflow.log_params({
            "model_type":  "ARM",
            "arima_p":     p,
            "arima_d":     d,
            "arima_q":     q,
            "n_steps_out": n_steps_out,
            "test_split":  test_split,
            "model_class": "ARIMA",
        })

        # ── Load data ─────────────────────────────────────────────────────────
        data_path = os.path.join(FEATURES_PATH, "price_features.csv")
        df = pd.read_csv(data_path, parse_dates=["timestamp"])
        df.sort_values(["symbol", "timestamp"], inplace=True)

        n_symbols = df["symbol"].nunique()
        mlflow.log_params({"n_symbols": n_symbols, "data_rows": len(df)})
        logger.info(f"Fitting ARIMA{order} on {n_symbols} symbols...")

        # ── Per-symbol fitting ────────────────────────────────────────────────
        fitted_models:  dict  = {}
        per_symbol_mse: list  = []
        per_symbol_mae: list  = []
        per_symbol_rmse: list = []
        per_symbol_mape: list = []
        min_rows = n_steps_out + 20

        for symbol, grp in df.groupby("symbol"):
            series = grp.sort_values("timestamp")["close"].dropna()

            if len(series) < min_rows:
                logger.warning(f"Skipping {symbol}: {len(series)} rows < {min_rows}")
                continue

            split      = int(len(series) * (1 - test_split))
            train_data = series.iloc[:split]
            test_data  = series.iloc[split : split + n_steps_out]

            if len(test_data) < 2:
                logger.warning(f"Skipping {symbol}: test set too small")
                continue

            try:
                result  = fit_arima(train_data, order)
                metrics = evaluate_arima(result, test_data, n_steps_out)

                fitted_models[symbol] = result
                per_symbol_mse.append(metrics["mse"])
                per_symbol_mae.append(metrics["mae"])
                per_symbol_rmse.append(metrics["rmse"])
                per_symbol_mape.append(metrics["mape"])

                # Log per-epoch style: one step per symbol
                step = len(per_symbol_mse) - 1
                mlflow.log_metrics({
                    "symbol_mse":  metrics["mse"],
                    "symbol_mae":  metrics["mae"],
                    "symbol_rmse": metrics["rmse"],
                }, step=step)

                logger.info(
                    f"  {symbol:<8} RMSE={metrics['rmse']:.2f} "
                    f"MAPE={metrics['mape']:.1f}%"
                )

            except Exception as exc:
                logger.warning(f"ARIMA failed for {symbol}: {exc}")

        if not fitted_models:
            raise RuntimeError("ARIMA fitting failed for all symbols")

        # ── Aggregate metrics ─────────────────────────────────────────────────
        avg_mse  = float(np.mean(per_symbol_mse))
        avg_mae  = float(np.mean(per_symbol_mae))
        avg_rmse = float(np.mean(per_symbol_rmse))
        avg_mape = float(np.mean(per_symbol_mape))

        mlflow.log_metrics({
            "train_loss":          avg_mse,   # convention: same metric name for compare_models
            "val_loss":            avg_mse,
            "train_mae":           avg_mae,
            "val_mae":             avg_mae,
            "test_mse":            avg_mse,
            "test_mae":            avg_mae,
            "test_rmse":           avg_rmse,
            "test_mape":           avg_mape,
            "symbols_fitted":      len(fitted_models),
            "symbols_total":       n_symbols,
        })

        logger.info(
            f"ARM | {len(fitted_models)}/{n_symbols} symbols fitted | "
            f"avg_rmse={avg_rmse:.4f}  avg_mape={avg_mape:.2f}%"
        )

        # ── Tags ──────────────────────────────────────────────────────────────
        tag_current_run(
            extra_tags={"model_type": "ARM"},
            data_hash=data_hash,
            data_commit=data_commit,
        )

        # ── Per-symbol metrics artifact ────────────────────────────────────────
        per_symbol_dict = {}
        for sym_idx, symbol in enumerate(list(fitted_models.keys())):
            if sym_idx < len(per_symbol_mse):
                per_symbol_dict[symbol] = {
                    "mse":  per_symbol_mse[sym_idx],
                    "mae":  per_symbol_mae[sym_idx],
                    "rmse": per_symbol_rmse[sym_idx],
                    "mape": per_symbol_mape[sym_idx],
                }
        if per_symbol_dict:
            log_metrics_table(per_symbol_dict)

        # ── Register pyfunc model in MLflow ───────────────────────────────────
        wrapper = ArimaWrapper(fitted_models, n_steps_out, order)

        with tempfile.TemporaryDirectory() as tmp:
            pickle_path = os.path.join(tmp, "arima_models.pkl")
            with open(pickle_path, "wb") as f:
                pickle.dump({"models": fitted_models, "order": order,
                             "n_steps_out": n_steps_out}, f)
            mlflow.log_artifact(pickle_path, artifact_path="arima_models")

        mlflow.pyfunc.log_model(
            artifact_path="arm_model",
            python_model=wrapper,
            registered_model_name="stock_price_forecaster_arm",
        )

        run_id = mlflow.active_run().info.run_id
        return {
            "run_id":       run_id,
            "val_loss":     avg_mse,
            "test_metrics": {
                "mse":  avg_mse,
                "mae":  avg_mae,
                "rmse": avg_rmse,
                "mape": avg_mape,
                "directional_accuracy": float("nan"),
            },
            "model_name":    "stock_price_forecaster_arm",
            "symbols_fitted": len(fitted_models),
        }


if __name__ == "__main__":
    train()
