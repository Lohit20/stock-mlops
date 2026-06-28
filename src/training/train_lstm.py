"""
LSTM Model Training with MLflow tracking.

Architecture: LSTM(60) -> Dropout(0.2) -> LSTM(30) -> Dropout(0.2) -> Dense(30)
Input:  90-day window of price + technical features, per-symbol RobustScaler normalization
Output: 30-day price forecast
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
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

_DEFAULT_FEATURES = [
    "close", "returns", "volatility_7", "rsi_14",
    "sma_7", "sma_30", "ema_7", "ema_30", "bb_width",
]


def split_sequence(seq, n_steps_in, n_steps_out, window):
    """Sliding-window conversion from time series to (X, y) pairs."""
    X, y = [], []
    for i in range(0, len(seq) - n_steps_in - n_steps_out + 1, window):
        X.append(seq[i : i + n_steps_in, :])
        y.append(seq[i + n_steps_in : i + n_steps_in + n_steps_out, 0])
    return np.array(X), np.array(y)


def build_sequences(df, n_steps_in, n_steps_out, window, feature_cols):
    """Per-symbol normalization then sequence creation."""
    all_X, all_y, scalers = [], [], {}
    min_rows = n_steps_in + n_steps_out + 10

    for symbol, grp in df.groupby("symbol"):
        cols_available = [c for c in feature_cols if c in grp.columns]
        sub = grp.sort_values("timestamp")[cols_available].dropna()

        if len(sub) < min_rows:
            logger.warning(f"Skipping {symbol}: {len(sub)} rows < {min_rows} required")
            continue

        scaler = RobustScaler()
        scaled = scaler.fit_transform(sub.values.astype(float))
        scalers[symbol] = scaler

        X, y = split_sequence(scaled, n_steps_in, n_steps_out, window)
        all_X.append(X)
        all_y.append(y)

    if not all_X:
        raise ValueError("No symbols with enough data — run bootstrap_data.py first")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    # Shuffle so batches mix symbols
    idx = np.random.default_rng(42).permutation(len(X))
    return X[idx], y[idx], scalers


def build_model(n_in, n_features, n_out):
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout

    model = Sequential([
        LSTM(60, activation="softsign", return_sequences=True,
             input_shape=(n_in, n_features)),
        Dropout(0.2),
        LSTM(30, activation="softsign"),
        Dropout(0.2),
        Dense(n_out),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train(
    n_steps_in=90,
    n_steps_out=30,
    window=30,
    epochs=50,
    batch_size=32,
    val_split=0.1,
    feature_cols=None,
):
    import mlflow
    import mlflow.keras

    feature_cols = feature_cols or _DEFAULT_FEATURES

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="LSTM"):
        mlflow.log_params({
            "model_type":   "LSTM",
            "n_steps_in":   n_steps_in,
            "n_steps_out":  n_steps_out,
            "window":       window,
            "epochs":       epochs,
            "batch_size":   batch_size,
            "val_split":    val_split,
            "optimizer":    "adam",
            "activation":   "softsign",
            "dropout":      0.2,
            "features":     str(feature_cols),
            "units_layer1": 60,
            "units_layer2": 30,
        })

        data_path = os.path.join(FEATURES_PATH, "price_features.csv")
        df = pd.read_csv(data_path, parse_dates=["timestamp"])
        df.sort_values(["symbol", "timestamp"], inplace=True)

        n_symbols = df["symbol"].nunique()
        mlflow.log_params({"n_symbols": n_symbols, "data_rows": len(df)})
        logger.info(f"Loaded {len(df):,} rows for {n_symbols} symbols")

        X, y, scalers = build_sequences(df, n_steps_in, n_steps_out, window, feature_cols)
        n_features = X.shape[2]
        logger.info(f"Sequences: X={X.shape}  y={y.shape}  features={n_features}")
        mlflow.log_params({"n_sequences": len(X), "n_features": n_features})

        test_size = max(1, int(len(X) * 0.10))
        X_train, y_train = X[:-test_size], y[:-test_size]
        X_test,  y_test  = X[-test_size:], y[-test_size:]

        model = build_model(n_steps_in, n_features, n_steps_out)

        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=val_split,
            verbose=1,
        )

        for step, (loss, val) in enumerate(
            zip(history.history["loss"], history.history["val_loss"])
        ):
            mlflow.log_metrics({"epoch_loss": loss, "epoch_val_loss": val}, step=step)

        y_pred = model.predict(X_test)
        test_m = compute_regression_metrics(y_test, y_pred)

        mlflow.log_metrics({
            "train_loss": history.history["loss"][-1],
            "val_loss":   history.history["val_loss"][-1],
            "train_mae":  history.history["mae"][-1],
            "val_mae":    history.history["val_mae"][-1],
            "test_mse":   test_m["mse"],
            "test_mae":   test_m["mae"],
            "test_rmse":  test_m["rmse"],
            "test_mape":  test_m["mape"],
            "test_directional_accuracy": test_m["directional_accuracy"],
        })

        logger.info(
            f"LSTM | val_loss={history.history['val_loss'][-1]:.4f} "
            f"test_rmse={test_m['rmse']:.4f} "
            f"test_mape={test_m['mape']:.2f}%"
        )

        mlflow.keras.log_model(
            model,
            artifact_path="lstm_model",
            registered_model_name="stock_price_forecaster_lstm",
            signature=sig,
        )

        run_id = mlflow.active_run().info.run_id
        return {
            "run_id":       run_id,
            "val_loss":     history.history["val_loss"][-1],
            "test_metrics": test_m,
            "model_name":   "stock_price_forecaster_lstm",
        }


if __name__ == "__main__":
    train()
