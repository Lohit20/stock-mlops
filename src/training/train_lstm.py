"""
LSTM Model Training with MLflow tracking.

What this file does:
- Loads feature data
- Trains your existing LSTM model
- Logs EVERYTHING to MLflow (params, metrics, model)
- Registers model in MLflow Model Registry

MLflow is like a lab notebook:
Every time you train, it automatically records:
- What settings you used (lookback=90, epochs=10...)
- How well it performed (MSE=0.068...)
- The actual trained model weights
- Plots and artifacts
"""

import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.keras
from sklearn.preprocessing import RobustScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

FEATURES_PATH       = os.getenv("FEATURES_DATA_PATH", "data/features")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME     = os.getenv("MLFLOW_EXPERIMENT_NAME", "stock_price_forecasting")


def split_sequence(seq: np.ndarray, n_steps_in: int, n_steps_out: int, window: int):
    """
    Convert time series into supervised learning format.

    Example:
    seq = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    n_steps_in=3, n_steps_out=2, window=2

    X (input)    y (output)
    [1, 2, 3] -> [4, 5]
    [3, 4, 5] -> [6, 7]
    [5, 6, 7] -> [8, 9]
    """
    X, y = [], []
    for i in range(0, len(seq), window):
        end     = i + n_steps_in
        out_end = end + n_steps_out
        if out_end > len(seq):
            break
        X.append(seq[i:end, :])
        y.append(seq[end:out_end, 0])
    return np.array(X), np.array(y)


def build_model(n_in: int, n_features: int, n_out: int) -> Sequential:
    """Build LSTM model architecture."""
    model = Sequential([
        LSTM(60, activation='softsign', return_sequences=True, input_shape=(n_in, n_features)),
        Dropout(0.2),       # Prevents overfitting (randomly drops 20% of neurons)
        LSTM(30, activation='softsign'),
        Dropout(0.2),
        Dense(n_out)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def train(
    n_steps_in:  int   = 90,
    n_steps_out: int   = 30,
    window:      int   = 30,
    epochs:      int   = 50,
    batch_size:  int   = 32,
    val_split:   float = 0.1,
):
    """
    Full LSTM training run with MLflow tracking.

    Args:
        n_steps_in:  How many past days to look at (lookback window)
        n_steps_out: How many future days to predict
        window:      Step size when creating training sequences
        epochs:      How many times to train on full dataset
        batch_size:  How many samples per gradient update
        val_split:   Fraction of data held out for validation
    """
    # ── 1. Connect to MLflow ──────────────────────────────────────────────────
    # MLflow needs to know WHERE to save experiment data
    # http://localhost:5000 is the MLflow server we'll run via Docker
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="LSTM"):
        logger.info("Starting LSTM training run...")

        # ── 2. Log hyperparameters to MLflow ─────────────────────────────────
        # These are the settings we used — MLflow stores them so we can
        # compare different runs later
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
        })

        # ── 3. Load and preprocess data ───────────────────────────────────────
        data_path = os.path.join(FEATURES_PATH, "price_features.csv")
        df = pd.read_csv(data_path, index_col='timestamp', parse_dates=True)
        df.sort_index(inplace=True)

        # Use only price column for now (can add more features later)
        price_df = df[['price']].copy()

        # Scale data — RobustScaler is best for financial data (handles outliers)
        scaler = RobustScaler()
        scaled = scaler.fit_transform(price_df)
        close_scaler = RobustScaler()
        close_scaler.fit(price_df[['price']])

        mlflow.log_param("data_rows", len(df))
        mlflow.log_param("features", list(price_df.columns))

        # ── 4. Create sequences ───────────────────────────────────────────────
        n_features = scaled.shape[1]
        X, y = split_sequence(scaled, n_steps_in, n_steps_out, window)
        logger.info(f"Training sequences: X={X.shape}, y={y.shape}")
        mlflow.log_param("n_sequences", len(X))

        # ── 5. Build and train model ──────────────────────────────────────────
        model = build_model(n_steps_in, n_features, n_steps_out)
        model.summary()

        history = model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=val_split,
            verbose=1,
        )

        # ── 6. Log metrics to MLflow ──────────────────────────────────────────
        # Log final epoch metrics
        final_loss     = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        final_mae      = history.history['mae'][-1]
        final_val_mae  = history.history['val_mae'][-1]

        mlflow.log_metrics({
            "train_loss":  final_loss,
            "val_loss":    final_val_loss,
            "train_mae":   final_mae,
            "val_mae":     final_val_mae,
        })

        # Log loss curve (every epoch) so we can see training progress in MLflow UI
        for epoch, (loss, val_loss) in enumerate(
            zip(history.history['loss'], history.history['val_loss'])
        ):
            mlflow.log_metrics({"epoch_loss": loss, "epoch_val_loss": val_loss}, step=epoch)

        logger.info(f"Final train_loss={final_loss:.4f} | val_loss={final_val_loss:.4f}")

        # ── 7. Save model to MLflow ───────────────────────────────────────────
        # This saves the model so it can be loaded and served later
        mlflow.keras.log_model(
            model,
            artifact_path="lstm_model",
            registered_model_name="stock_price_forecaster_lstm",
        )

        run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow run ID: {run_id}")
        logger.info("LSTM training complete.")

        return {
            "run_id":      run_id,
            "train_loss":  final_loss,
            "val_loss":    final_val_loss,
            "model_name":  "stock_price_forecaster_lstm",
        }


if __name__ == "__main__":
    train()
