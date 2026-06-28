"""
FastAPI Prediction Server — Step 10 / Step 13.

Endpoints
---------
GET  /health                           liveness + loaded model list
GET  /health/detailed                  per-model registry info + cache stats
GET  /models                           list loaded model names
POST /predict/best                     predict with whichever model is in Production
POST /predict/compare                  run ALL loaded models, return side-by-side
POST /predict/{model_type}             predict with a specific model (cached)
GET  /registry/summary                 all registered versions + stages (DataFrame → JSON)
GET  /registry/{model_type}/production production model info for one model type
POST /admin/reload                     hot-reload all models from the registry
DELETE /cache                          flush prediction cache (optional symbol/model filter)
GET  /cache/stats                      cache hit/miss counters
POST /monitoring/push                  ingest a pipeline-run summary → update Prometheus gauges
GET  /monitoring/drift                 run live drift check and return result (also updates gauges)
GET  /metrics                          Prometheus scrape endpoint
"""

import os
import numpy as np
import pandas as pd
from datetime import date
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from loguru import logger
from dotenv import load_dotenv
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import PlainTextResponse

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
FEATURES_PATH       = os.getenv("FEATURES_DATA_PATH", "data/features")
N_STEPS_IN          = int(os.getenv("N_STEPS_IN",  "90"))
N_STEPS_OUT         = int(os.getenv("N_STEPS_OUT", "30"))

_DEFAULT_FEATURES = [
    "close", "returns", "volatility_7", "rsi_14",
    "sma_7", "sma_30", "ema_7", "ema_30", "bb_width",
]

# ── Prometheus metrics ────────────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "api_requests_total", "Total API requests", ["model", "endpoint"]
)
REQUEST_LATENCY = Histogram(
    "api_request_duration_seconds", "API request duration", ["model"]
)
PREDICTION_ERROR = Counter(
    "api_prediction_errors_total", "Total prediction errors", ["model"]
)
CACHE_HIT = Counter("api_cache_hits_total",   "Cache hits",   ["model"])
CACHE_MISS = Counter("api_cache_misses_total", "Cache misses", ["model"])

# Pipeline-level metrics (imported so they register in the same Prometheus registry)
from src.monitoring.metrics import (  # noqa: E402
    push_drift_metrics,
    push_model_metrics,
    push_pipeline_run_metrics,
)

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Stock Price Forecasting API",
    description="Serves predictions from LSTM, TFT, TimesFM and ARM models",
    version="2.0.0",
)

# ── In-memory state ───────────────────────────────────────────────────────────
models          = {}   # model_name → pyfunc model
model_metadata  = {}   # model_name → {version, stage, run_id, loaded_at}


# ── Schemas ───────────────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    symbol:     str = "AAPL"
    days_ahead: int = 30


class PredictionResponse(BaseModel):
    symbol:        str
    model:         str
    model_version: str | None = None
    predictions:   list[float]
    dates:         list[str]
    cached:        bool = False


class CompareResponse(BaseModel):
    symbol:      str
    dates:       list[str]
    predictions: dict[str, list[float]]  # model_type → forecast list
    metadata:    dict[str, dict]         # model_type → {version, cached}


# ── Registry helpers ──────────────────────────────────────────────────────────

def _get_registry_info(model_type: str) -> dict | None:
    try:
        from src.registry.model_registry import get_latest_production
        return get_latest_production(model_type)
    except Exception:
        return None


# ── Model loader ──────────────────────────────────────────────────────────────

def load_model(model_name: str) -> tuple:
    """
    Load Production model from registry.
    Returns (pyfunc_model, metadata_dict).
    """
    import mlflow.pyfunc
    from src.registry.model_registry import get_latest_production, REGISTERED_MODELS

    model_type = next(
        (mt for mt, mn in REGISTERED_MODELS.items() if mn == model_name), None
    )
    version = None

    if model_type:
        info = get_latest_production(model_type)
        if info:
            logger.info(f"Registry: loading {model_name} v{info['version']}")
            try:
                mdl     = mlflow.pyfunc.load_model(info["model_uri"])
                version = info["version"]
                return mdl, {"version": version, "stage": "Production",
                              "run_id": info.get("run_id"), "loaded_at": str(date.today())}
            except Exception as exc:
                logger.warning(f"Registry load failed for {model_name}: {exc}")

    # Direct-URI fallback
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{model_name}/Production"
    logger.info(f"Loading (direct): {model_uri}")
    mdl = mlflow.pyfunc.load_model(model_uri)
    return mdl, {"version": None, "stage": "Production",
                  "run_id": None, "loaded_at": str(date.today())}


@app.on_event("startup")
async def load_all_models():
    model_names = [
        "stock_price_forecaster_lstm",
        "stock_price_forecaster_tft",
        "stock_price_forecaster_timesfm",
        "stock_price_forecaster_arm",
    ]
    for name in model_names:
        try:
            mdl, meta = load_model(name)
            models[name]         = mdl
            model_metadata[name] = meta
            logger.info(f"Loaded {name} v{meta.get('version')}")
        except Exception as exc:
            logger.warning(f"Could not load {name}: {exc}")


# ── Prediction adapters ───────────────────────────────────────────────────────

def _load_features_df() -> pd.DataFrame:
    path = os.path.join(FEATURES_PATH, "price_features.csv")
    return pd.read_csv(path, parse_dates=["timestamp"])


def _predict_lstm(model, symbol: str, df: pd.DataFrame, n_steps_in: int, n_steps_out: int) -> np.ndarray:
    """Build multi-feature input for LSTM and invert the close-column scale."""
    from sklearn.preprocessing import RobustScaler

    sym_df = df[df["symbol"] == symbol].sort_values("timestamp")
    cols   = [c for c in _DEFAULT_FEATURES if c in sym_df.columns]

    if len(sym_df) < n_steps_in:
        raise ValueError(f"Not enough rows for {symbol}: {len(sym_df)} < {n_steps_in}")

    series = sym_df[cols].dropna().values
    scaler = RobustScaler().fit(series[:-n_steps_out] if len(series) > n_steps_out else series)
    X_scaled = scaler.transform(series[-n_steps_in:])
    X = X_scaled.reshape(1, n_steps_in, len(cols)).astype("float32")

    y_scaled = np.asarray(model.predict(X)).flatten()
    # Inverse-transform close column only
    pad = np.zeros((len(y_scaled), len(cols)), dtype="float32")
    pad[:, 0] = y_scaled
    return scaler.inverse_transform(pad)[:, 0]


def _predict_arm(model, symbol: str, n_steps_out: int) -> np.ndarray:
    """ARM pyfunc expects a DataFrame with a 'symbol' column."""
    inp = pd.DataFrame({"symbol": [symbol]})
    out = model.predict(inp)
    return np.asarray(out["predicted_close"].values[:n_steps_out], dtype="float64")


def _predict_generic(model, symbol: str, df: pd.DataFrame, n_steps_in: int) -> np.ndarray:
    """Fallback: pass last n_steps_in close prices as numpy array."""
    sym_df = df[df["symbol"] == symbol].sort_values("timestamp")
    close  = sym_df["close"].dropna().values
    X = close[-n_steps_in:].reshape(1, n_steps_in, 1).astype("float32")
    return np.asarray(model.predict(X)).flatten()


def _run_model(model_name: str, symbol: str, df: pd.DataFrame,
               n_steps_in: int, n_steps_out: int) -> list[float]:
    """Dispatch to the right adapter based on model_name suffix."""
    model = models[model_name]
    suffix = model_name.replace("stock_price_forecaster_", "").lower()

    if suffix == "lstm":
        preds = _predict_lstm(model, symbol, df, n_steps_in, n_steps_out)
    elif suffix == "arm":
        preds = _predict_arm(model, symbol, n_steps_out)
    else:
        preds = _predict_generic(model, symbol, df, n_steps_in)

    return preds.flatten()[:n_steps_out].tolist()


# ── Date helpers ──────────────────────────────────────────────────────────────

def _future_dates(df: pd.DataFrame, symbol: str, n: int) -> list[str]:
    sym_df = df[df["symbol"] == symbol] if "symbol" in df.columns else df
    if sym_df.empty:
        last = pd.Timestamp(date.today())
    else:
        last = pd.to_datetime(sym_df["timestamp"].iloc[-1])
    return [d.strftime("%Y-%m-%d")
            for d in pd.date_range(last, periods=n + 1, freq="B")[1:]]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {
        "status":        "healthy",
        "models_loaded": list(models.keys()),
    }


@app.get("/health/detailed")
def health_detailed():
    from src.serving.cache import cache_stats as cs
    registry_info = {}
    from src.registry.model_registry import REGISTERED_MODELS
    for model_type in REGISTERED_MODELS:
        info = _get_registry_info(model_type)
        registry_info[model_type] = info or {}

    return {
        "status":        "healthy",
        "models_loaded": list(models.keys()),
        "model_metadata": model_metadata,
        "registry":      registry_info,
        "cache":         cs(),
    }


@app.get("/models")
def list_models():
    return {"available_models": list(models.keys())}


# ── /predict/compare  (must be defined BEFORE /predict/{model_type}) ──────────

@app.post("/predict/compare", response_model=CompareResponse)
def predict_compare(request: PredictionRequest):
    """
    Run all loaded models on the same symbol and return side-by-side forecasts.
    Each model's predictions are cached independently.
    """
    if not models:
        raise HTTPException(status_code=503, detail="No models loaded")

    from src.serving.cache import cache_key, get_cached, set_cached

    df           = _load_features_df()
    dates        = _future_dates(df, request.symbol, request.days_ahead)
    all_preds    = {}
    all_metadata = {}

    for model_name in list(models.keys()):
        model_type = model_name.replace("stock_price_forecaster_", "")
        REQUEST_COUNT.labels(model=model_type, endpoint="compare").inc()

        key    = cache_key(request.symbol, model_type)
        cached = get_cached(key)

        if cached:
            CACHE_HIT.labels(model=model_type).inc()
            all_preds[model_type]    = cached["predictions"]
            all_metadata[model_type] = {
                "version": cached.get("model_version"),
                "cached":  True,
            }
            continue

        CACHE_MISS.labels(model=model_type).inc()
        with REQUEST_LATENCY.labels(model=model_type).time():
            try:
                preds = _run_model(model_name, request.symbol, df,
                                   N_STEPS_IN, request.days_ahead)
                version = (model_metadata.get(model_name) or {}).get("version")
                entry   = {"predictions": preds, "model_version": version}
                set_cached(key, entry)
                all_preds[model_type]    = preds
                all_metadata[model_type] = {"version": version, "cached": False}
            except Exception as exc:
                logger.warning(f"Compare: {model_name} failed for {request.symbol}: {exc}")
                PREDICTION_ERROR.labels(model=model_type).inc()

    if not all_preds:
        raise HTTPException(
            status_code=500,
            detail=f"All models failed for symbol '{request.symbol}'"
        )

    return CompareResponse(
        symbol=request.symbol,
        dates=dates[:request.days_ahead],
        predictions=all_preds,
        metadata=all_metadata,
    )


# ── /predict/best  (defined AFTER /predict/compare to avoid route conflict) ───

@app.post("/predict/best", response_model=PredictionResponse)
def predict_best(request: PredictionRequest):
    """Predict with the first loaded Production model."""
    if not models:
        raise HTTPException(status_code=503, detail="No models loaded")
    best_name  = list(models.keys())[0]
    model_type = best_name.replace("stock_price_forecaster_", "")
    return predict(model_type, request)


# ── /predict/{model_type} ─────────────────────────────────────────────────────

@app.post("/predict/{model_type}", response_model=PredictionResponse)
def predict(model_type: str, request: PredictionRequest):
    """Predict with a specific model (result is cached per symbol per day)."""
    model_name = f"stock_price_forecaster_{model_type}"
    if model_name not in models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_type}' not loaded. Available: {list(models.keys())}"
        )

    REQUEST_COUNT.labels(model=model_type, endpoint="predict").inc()

    # Cache lookup
    from src.serving.cache import cache_key, get_cached, set_cached
    key    = cache_key(request.symbol, model_type)
    cached = get_cached(key)
    if cached:
        CACHE_HIT.labels(model=model_type).inc()
        df    = _load_features_df()
        dates = _future_dates(df, request.symbol, request.days_ahead)
        return PredictionResponse(
            symbol=request.symbol,
            model=model_type,
            model_version=cached.get("model_version"),
            predictions=cached["predictions"][:request.days_ahead],
            dates=dates[:request.days_ahead],
            cached=True,
        )

    CACHE_MISS.labels(model=model_type).inc()
    with REQUEST_LATENCY.labels(model=model_type).time():
        try:
            df    = _load_features_df()
            preds = _run_model(model_name, request.symbol, df,
                               N_STEPS_IN, request.days_ahead)
            dates   = _future_dates(df, request.symbol, request.days_ahead)
            version = (model_metadata.get(model_name) or {}).get("version")

            set_cached(key, {"predictions": preds, "model_version": version})

            return PredictionResponse(
                symbol=request.symbol,
                model=model_type,
                model_version=version,
                predictions=preds[:request.days_ahead],
                dates=dates[:request.days_ahead],
                cached=False,
            )
        except Exception as exc:
            PREDICTION_ERROR.labels(model=model_type).inc()
            logger.error(f"Prediction error for {model_type}/{request.symbol}: {exc}")
            raise HTTPException(status_code=500, detail=str(exc))


# ── /registry/* ───────────────────────────────────────────────────────────────

@app.get("/registry/summary")
def registry_summary():
    """All registered model versions and stages across all model types."""
    try:
        from src.registry.model_registry import get_registry_summary
        df = get_registry_summary()
        return df.fillna("").to_dict(orient="records")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Registry unavailable: {exc}")


@app.get("/registry/{model_type}/production")
def registry_production(model_type: str):
    """Production version info for a specific model type."""
    info = _get_registry_info(model_type.upper())
    if info is None:
        raise HTTPException(
            status_code=404,
            detail=f"No Production version found for model_type='{model_type}'"
        )
    return info


# ── /admin/reload ─────────────────────────────────────────────────────────────

@app.post("/admin/reload")
def reload_models():
    """
    Hot-reload all models from the MLflow registry.
    Useful after a Step 9 promotion without restarting the server.
    """
    loaded  = []
    failed  = []
    for name in [
        "stock_price_forecaster_lstm",
        "stock_price_forecaster_tft",
        "stock_price_forecaster_timesfm",
        "stock_price_forecaster_arm",
    ]:
        try:
            mdl, meta        = load_model(name)
            models[name]     = mdl
            model_metadata[name] = meta
            loaded.append(name)
            logger.info(f"Reloaded {name} v{meta.get('version')}")
        except Exception as exc:
            failed.append({"name": name, "error": str(exc)})
            logger.warning(f"Reload failed for {name}: {exc}")

    return {"loaded": loaded, "failed": failed}


# ── /cache/* ──────────────────────────────────────────────────────────────────

@app.delete("/cache")
def flush_cache(
    symbol:     str | None = Query(default=None),
    model_type: str | None = Query(default=None),
):
    """
    Invalidate cached predictions.
    Pass ?symbol=AAPL and/or ?model_type=lstm to narrow the scope.
    Omit both to flush everything.
    """
    from src.serving.cache import invalidate
    deleted = invalidate(symbol=symbol, model_type=model_type)
    return {"deleted": deleted, "symbol": symbol, "model_type": model_type}


@app.get("/cache/stats")
def get_cache_stats():
    from src.serving.cache import cache_stats
    return cache_stats()


# ── /monitoring/* ────────────────────────────────────────────────────────────

class MonitoringPushRequest(BaseModel):
    """
    Payload sent by the Airflow send_report task after each pipeline run.
    All fields are optional — only present keys update Prometheus gauges.
    """
    drift_score:        float | None = None
    drift_detected:     bool  | None = None
    needs_retraining:   bool  | None = None
    winner_model:       str   | None = None
    winner_version:     str   | None = None
    winner_rmse:        float | None = None
    duration_seconds:   float | None = None
    dataset:            str         = "price_features"
    model_metrics:      dict        = {}  # model_type → {test_rmse, test_mape, ...}


@app.post("/monitoring/push")
def monitoring_push(payload: MonitoringPushRequest):
    """
    Accept a pipeline-run summary from Airflow and update all Prometheus gauges.
    The Prometheus scrape endpoint (/metrics) will reflect updated values immediately.
    """
    summary = payload.model_dump()
    push_pipeline_run_metrics(summary)

    for model_type, eval_metrics in (payload.model_metrics or {}).items():
        push_model_metrics(model_type, eval_metrics)

    logger.info(
        f"Monitoring push: drift={payload.drift_score} "
        f"winner={payload.winner_model} retrain={payload.needs_retraining}"
    )
    return {"status": "ok", "updated": list(summary.keys())}


@app.get("/monitoring/drift")
def monitoring_drift():
    """
    Run a live data drift check, update Prometheus gauges, and return the result.
    Wraps src.monitoring.drift.run_data_drift_report.
    """
    try:
        from src.monitoring.drift import (
            load_reference_data, load_current_data, run_data_drift_report,
        )
        reference = load_reference_data()
        current   = load_current_data()
        result    = run_data_drift_report(reference, current)
        push_drift_metrics(result)
        return result
    except Exception as exc:
        logger.error(f"Drift check failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


# ── /metrics (Prometheus) ─────────────────────────────────────────────────────

@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return generate_latest()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
