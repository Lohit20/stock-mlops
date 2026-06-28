"""
Fresh model evaluation on held-out test data.

For each model type, loads the best registered version from MLflow and
generates 30-day forecasts for every symbol in price_features.csv.
The last `n_steps_out` rows per symbol are the ground truth.

Model-specific prediction adapters:
  LSTM      — pyfunc (numpy input)
  TFT       — historical metrics only (pytorch dataset too complex to rebuild)
  TimesFM   — re-runs ETS fallback directly (no GPU required)
  ARM       — pyfunc ArimaWrapper (DataFrame input)

Falls back to stored MLflow metrics when a model cannot be loaded.
"""

import os
import numpy as np
import pandas as pd
from loguru import logger
from dotenv import load_dotenv

from src.training.metrics import compute_regression_metrics

load_dotenv()

FEATURES_PATH       = os.getenv("FEATURES_DATA_PATH", "data/features")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME     = os.getenv("MLFLOW_EXPERIMENT_NAME", "stock_price_forecasting")

_MODELS = {
    "LSTM":    "stock_price_forecaster_lstm",
    "TFT":     "stock_price_forecaster_tft",
    "TimesFM": "stock_price_forecaster_timesfm",
    "ARM":     "stock_price_forecaster_arm",
}
_DEFAULT_FEATURES = [
    "close", "returns", "volatility_7", "rsi_14",
    "sma_7", "sma_30", "ema_7", "ema_30", "bb_width",
]


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_test_data(
    features_path: str,
    n_steps_in:    int = 90,
    n_steps_out:   int = 30,
) -> dict:
    """
    Split price_features.csv into per-symbol (X_context, y_true) pairs.

    For each symbol, the last `n_steps_out` rows are ground truth and
    the `n_steps_in` rows before that are the input context.

    Returns:
        {symbol: {"context": np.ndarray (n_steps_in,), "y_true": np.ndarray (n_steps_out,)}}
    """
    path = os.path.join(features_path, "price_features.csv")
    df   = pd.read_csv(path, parse_dates=["timestamp"])
    df.sort_values(["symbol", "timestamp"], inplace=True)

    min_rows = n_steps_in + n_steps_out
    result   = {}

    for symbol, grp in df.groupby("symbol"):
        prices = grp["close"].dropna().values
        if len(prices) < min_rows:
            logger.warning(f"Skipping {symbol}: only {len(prices)} rows")
            continue

        context = prices[-(n_steps_in + n_steps_out):-n_steps_out]
        y_true  = prices[-n_steps_out:]
        result[symbol] = {"context": context, "y_true": y_true, "df": grp}

    return result


# ── Per-model predictors ──────────────────────────────────────────────────────

def _predict_lstm(model, test_data: dict, n_steps_in: int, n_steps_out: int) -> dict:
    """Predict with LSTM pyfunc model (numpy array input)."""
    from sklearn.preprocessing import RobustScaler

    per_symbol = {}
    for symbol, data in test_data.items():
        try:
            grp = data["df"]
            cols = [c for c in _DEFAULT_FEATURES if c in grp.columns]
            series = grp.sort_values("timestamp")[cols].dropna().values
            if len(series) < n_steps_in + n_steps_out:
                continue

            # Normalise the same way training did (per-symbol RobustScaler)
            scaler = RobustScaler()
            scaler.fit(series[:-n_steps_out])
            X_scaled = scaler.transform(series[-(n_steps_in + n_steps_out):-n_steps_out])
            X = X_scaled.reshape(1, n_steps_in, len(cols)).astype("float32")

            # pyfunc predict accepts numpy; output is (1, n_steps_out) scaled
            y_pred_scaled = model.predict(X)
            # Inverse-transform: only the close column (index 0) matters
            pad = np.zeros((y_pred_scaled.shape[1], len(cols)), dtype="float32")
            pad[:, 0] = y_pred_scaled.flatten()
            y_pred = scaler.inverse_transform(pad)[:, 0]

            per_symbol[symbol] = {
                "y_true": data["y_true"],
                "y_pred": y_pred,
            }
        except Exception as exc:
            logger.warning(f"LSTM prediction failed for {symbol}: {exc}")

    return per_symbol


def _predict_arm(model, test_data: dict, n_steps_out: int) -> dict:
    """Predict with ARM pyfunc (ArimaWrapper) — DataFrame input."""
    per_symbol = {}
    for symbol, data in test_data.items():
        try:
            inp = pd.DataFrame({"symbol": [symbol]})
            out = model.predict(inp)
            y_pred = out["predicted_close"].values[:n_steps_out]
            per_symbol[symbol] = {
                "y_true": data["y_true"],
                "y_pred": y_pred,
            }
        except Exception as exc:
            logger.warning(f"ARM prediction failed for {symbol}: {exc}")
    return per_symbol


def _predict_timesfm(test_data: dict, n_steps_out: int) -> dict:
    """Predict using ETS fallback (no MLflow model required)."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    per_symbol = {}
    for symbol, data in test_data.items():
        try:
            context = data["context"]
            fit = ExponentialSmoothing(
                context, trend="add", seasonal=None,
                initialization_method="estimated",
            ).fit(optimized=True, remove_bias=True)
            y_pred = fit.forecast(n_steps_out)
            per_symbol[symbol] = {
                "y_true": data["y_true"],
                "y_pred": np.asarray(y_pred),
            }
        except Exception as exc:
            logger.warning(f"TimesFM/ETS prediction failed for {symbol}: {exc}")
    return per_symbol


# ── Metrics aggregation ───────────────────────────────────────────────────────

def _aggregate(model_type: str, per_symbol: dict) -> list:
    """Convert per-symbol {y_true, y_pred} dict into a list of metric rows."""
    rows = []
    for symbol, data in per_symbol.items():
        m = compute_regression_metrics(
            data["y_true"].reshape(1, -1),
            data["y_pred"].reshape(1, -1),
        )
        rows.append({
            "model_type":            model_type,
            "symbol":                symbol,
            "rmse":                  m["rmse"],
            "mae":                   m["mae"],
            "mse":                   m["mse"],
            "mape":                  m["mape"],
            "directional_accuracy":  m["directional_accuracy"],
        })
    return rows


# ── Main evaluator ────────────────────────────────────────────────────────────

def run_fresh_evaluation(
    n_steps_in:  int = 90,
    n_steps_out: int = 30,
    features_path: str = FEATURES_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate all 4 models on the same held-out test set.

    Returns:
        (per_symbol_df, aggregated_df)
        per_symbol_df: one row per (model_type, symbol) with rmse/mape/etc.
        aggregated_df: one row per model_type with avg metrics and rank.
    """
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    test_data = load_test_data(features_path, n_steps_in, n_steps_out)
    if not test_data:
        raise ValueError("No symbols with enough data for evaluation")

    logger.info(f"Evaluating on {len(test_data)} symbols, "
                f"context={n_steps_in}d, horizon={n_steps_out}d")

    all_rows = []

    # ── ARM (fresh pyfunc) ────────────────────────────────────────────────────
    try:
        arm_model = mlflow.pyfunc.load_model(
            f"models:/{_MODELS['ARM']}/Production"
        )
        rows = _aggregate("ARM", _predict_arm(arm_model, test_data, n_steps_out))
        all_rows.extend(rows)
        logger.info(f"ARM evaluated on {len(rows)} symbols")
    except Exception as exc:
        logger.warning(f"ARM fresh evaluation skipped: {exc}")

    # ── TimesFM / ETS (no model loading required) ─────────────────────────────
    try:
        rows = _aggregate("TimesFM", _predict_timesfm(test_data, n_steps_out))
        all_rows.extend(rows)
        logger.info(f"TimesFM/ETS evaluated on {len(rows)} symbols")
    except Exception as exc:
        logger.warning(f"TimesFM evaluation skipped: {exc}")

    # ── LSTM (pyfunc numpy) ───────────────────────────────────────────────────
    try:
        lstm_model = mlflow.pyfunc.load_model(
            f"models:/{_MODELS['LSTM']}/Production"
        )
        rows = _aggregate("LSTM", _predict_lstm(lstm_model, test_data, n_steps_in, n_steps_out))
        all_rows.extend(rows)
        logger.info(f"LSTM evaluated on {len(rows)} symbols")
    except Exception as exc:
        logger.warning(f"LSTM fresh evaluation skipped: {exc}")

    # ── TFT (skip fresh eval — use stored metrics) ────────────────────────────
    logger.info("TFT: using stored MLflow metrics (TimeSeriesDataSet rebuild not supported)")

    if not all_rows:
        raise RuntimeError(
            "No models could be freshly evaluated. "
            "Check that at least ARM is registered in MLflow."
        )

    per_symbol_df = pd.DataFrame(all_rows)

    # Aggregated: mean metrics per model_type, ranked by rmse
    agg = (
        per_symbol_df.groupby("model_type")[["rmse", "mae", "mse", "mape", "directional_accuracy"]]
        .mean()
        .reset_index()
        .sort_values("rmse")
    )
    agg["rank"] = range(1, len(agg) + 1)

    return per_symbol_df, agg
