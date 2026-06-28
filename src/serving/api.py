"""
FastAPI Prediction Server.

What this file does:
- Runs a web server that anyone can call to get stock predictions
- Loads the best Production model from MLflow
- Returns 30-day price forecasts as JSON

Think of it as a restaurant:
- Customer sends order (stock symbol)
- Kitchen (ML model) prepares the forecast
- Waiter (FastAPI) delivers the prediction
"""

import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.keras
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from loguru import logger
from dotenv import load_dotenv
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import PlainTextResponse

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
FEATURES_PATH       = os.getenv("FEATURES_DATA_PATH", "data/features")

# ── Prometheus metrics ────────────────────────────────────────────────────────
# These counters are automatically scraped by Prometheus every 15 seconds
REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total API requests",
    ["model", "endpoint"]
)
REQUEST_LATENCY = Histogram(
    "api_request_duration_seconds",
    "API request duration",
    ["model"]
)
PREDICTION_ERROR = Counter(
    "api_prediction_errors_total",
    "Total prediction errors",
    ["model"]
)

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Stock Price Forecasting API",
    description="Serves predictions from LSTM, TFT, TimesFM and ARM models",
    version="1.0.0",
)


# ── Request/Response schemas ──────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    """What the caller must send."""
    symbol:     str = "AAPL"   # Stock ticker
    days_ahead: int = 30       # How many days to forecast


class PredictionResponse(BaseModel):
    """What we send back."""
    symbol:      str
    model:       str
    predictions: list[float]
    dates:       list[str]


# ── Model loader ──────────────────────────────────────────────────────────────
def load_model(model_name: str):
    """
    Load a model from MLflow Model Registry.

    'models:/stock_price_forecaster_lstm/Production'
     means: load the Production version of lstm model
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{model_name}/Production"
    logger.info(f"Loading model from: {model_uri}")
    return mlflow.pyfunc.load_model(model_uri)


# Load all models at startup (so first request is fast)
models = {}

@app.on_event("startup")
async def load_all_models():
    """Load all Production models when the API server starts."""
    model_names = [
        "stock_price_forecaster_lstm",
        "stock_price_forecaster_tft",
        "stock_price_forecaster_timesfm",
        "stock_price_forecaster_arm",
    ]
    for name in model_names:
        try:
            models[name] = load_model(name)
            logger.info(f"✅ Loaded {name}")
        except Exception as e:
            logger.warning(f"⚠️ Could not load {name}: {e}")


# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Quick check that the API is running."""
    return {
        "status":         "healthy",
        "models_loaded":  list(models.keys()),
    }


@app.get("/models")
def list_models():
    """List all available models."""
    return {"available_models": list(models.keys())}


@app.post("/predict/{model_type}", response_model=PredictionResponse)
def predict(model_type: str, request: PredictionRequest):
    """
    Get prediction from a specific model.

    Example call:
    POST /predict/lstm
    {"symbol": "AAPL", "days_ahead": 30}
    """
    model_name = f"stock_price_forecaster_{model_type}"

    if model_name not in models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_type}' not available. Use: {list(models.keys())}"
        )

    REQUEST_COUNT.labels(model=model_type, endpoint="predict").inc()

    with REQUEST_LATENCY.labels(model=model_type).time():
        try:
            # Load recent price data for this symbol
            data_path = os.path.join(FEATURES_PATH, "price_features.csv")
            df = pd.read_csv(data_path, index_col='timestamp', parse_dates=True)

            # Use last 90 days as input
            recent = df['price'].tail(90).values.reshape(1, 90, 1).astype(np.float32)

            # Run model
            preds = models[model_name].predict(recent)
            preds = preds.flatten().tolist()

            # Generate future dates
            last_date = df.index[-1]
            future_dates = pd.date_range(
                start=last_date, periods=len(preds) + 1, freq='B'  # 'B' = business days only
            )[1:]

            return PredictionResponse(
                symbol=request.symbol,
                model=model_type,
                predictions=preds,
                dates=[d.strftime("%Y-%m-%d") for d in future_dates],
            )

        except Exception as e:
            PREDICTION_ERROR.labels(model=model_type).inc()
            logger.error(f"Prediction error for {model_type}: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/best", response_model=PredictionResponse)
def predict_best(request: PredictionRequest):
    """Use whichever model is currently best (lowest val_loss)."""
    if not models:
        raise HTTPException(status_code=503, detail="No models loaded")

    # Use first available model (in production this picks the best from MLflow)
    best_model_name = list(models.keys())[0]
    model_type = best_model_name.replace("stock_price_forecaster_", "")
    return predict(model_type, request)


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    """Prometheus scrapes this endpoint for metrics."""
    return generate_latest()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
