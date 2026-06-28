"""
Tests for Step 10: FastAPI serving enhancements
  src/serving/api.py
  src/serving/cache.py
"""

import json
import time
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


# ── Shared fixtures ───────────────────────────────────────────────────────────

def _price_df(n=150, symbol="AAPL"):
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    price = 100 + np.cumsum(np.random.randn(n))
    df    = pd.DataFrame({
        "timestamp":    dates,
        "symbol":       symbol,
        "close":        price,
        "returns":      np.random.randn(n) * 0.01,
        "volatility_7": np.abs(np.random.randn(n)) * 0.02,
        "rsi_14":       np.random.uniform(30, 70, n),
        "sma_7":        price, "sma_30": price,
        "ema_7":        price, "ema_30": price,
        "bb_width":     np.random.uniform(0.01, 0.05, n),
    })
    return df


def _mock_lstm_model(n_out=30):
    m = MagicMock()
    m.predict.return_value = np.zeros((1, n_out), dtype="float32")
    return m


def _mock_arm_model(symbol="AAPL", n_out=30):
    m   = MagicMock()
    out = pd.DataFrame({
        "symbol":          [symbol] * n_out,
        "forecast_day":    list(range(1, n_out + 1)),
        "predicted_close": [150.0] * n_out,
    })
    m.predict.return_value = out
    return m


@pytest.fixture
def client_with_lstm():
    """TestClient with only LSTM mock loaded; startup bypassed; cache reset."""
    from src.serving import cache as cache_mod
    from src.serving.api import app, models, model_metadata
    cache_mod._reset_client()

    models.clear()
    model_metadata.clear()
    models["stock_price_forecaster_lstm"]          = _mock_lstm_model()
    model_metadata["stock_price_forecaster_lstm"]  = {"version": "3", "stage": "Production",
                                                       "run_id": "abc", "loaded_at": "2024-01-01"}
    app.router.on_startup.clear()

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c

    models.clear()
    model_metadata.clear()


@pytest.fixture
def client_multi():
    """TestClient with LSTM + ARM mocks loaded."""
    from src.serving import cache as cache_mod
    from src.serving.api import app, models, model_metadata
    cache_mod._reset_client()

    models.clear()
    model_metadata.clear()
    models["stock_price_forecaster_lstm"] = _mock_lstm_model()
    models["stock_price_forecaster_arm"]  = _mock_arm_model()
    model_metadata["stock_price_forecaster_lstm"] = {"version": "2", "loaded_at": "2024-01-01"}
    model_metadata["stock_price_forecaster_arm"]  = {"version": "1", "loaded_at": "2024-01-01"}
    app.router.on_startup.clear()

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c

    models.clear()
    model_metadata.clear()


@pytest.fixture
def empty_client():
    from src.serving import cache as cache_mod
    from src.serving.api import app, models, model_metadata
    cache_mod._reset_client()
    models.clear()
    model_metadata.clear()
    app.router.on_startup.clear()

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c

    models.clear()
    model_metadata.clear()


# ── /health ───────────────────────────────────────────────────────────────────

class TestHealth:
    def test_returns_200(self, client_with_lstm):
        assert client_with_lstm.get("/health").status_code == 200

    def test_lists_loaded_models(self, client_with_lstm):
        body = client_with_lstm.get("/health").json()
        assert "models_loaded" in body
        assert "stock_price_forecaster_lstm" in body["models_loaded"]

    def test_detailed_includes_cache_and_registry(self, client_with_lstm):
        with patch("src.serving.api._get_registry_info", return_value=None), \
             patch("src.serving.cache.cache_stats", return_value={"hits": 0}):
            body = client_with_lstm.get("/health/detailed").json()
        assert "cache"          in body
        assert "registry"       in body
        assert "model_metadata" in body


# ── /models ───────────────────────────────────────────────────────────────────

class TestModels:
    def test_returns_available_models(self, client_with_lstm):
        body = client_with_lstm.get("/models").json()
        assert "available_models" in body
        assert "stock_price_forecaster_lstm" in body["available_models"]


# ── /predict/{model_type} ─────────────────────────────────────────────────────

class TestPredictSingle:
    def test_returns_200_with_predictions(self, client_with_lstm):
        with patch("src.serving.api._load_features_df", return_value=_price_df()):
            resp = client_with_lstm.post(
                "/predict/lstm", json={"symbol": "AAPL", "days_ahead": 30}
            )
        assert resp.status_code == 200
        body = resp.json()
        assert "predictions" in body
        assert "dates"       in body
        assert body["symbol"] == "AAPL"
        assert body["model"]  == "lstm"

    def test_includes_model_version(self, client_with_lstm):
        with patch("src.serving.api._load_features_df", return_value=_price_df()):
            body = client_with_lstm.post(
                "/predict/lstm", json={"symbol": "AAPL", "days_ahead": 30}
            ).json()
        assert body["model_version"] == "3"

    def test_prediction_length_matches_days_ahead(self, client_with_lstm):
        with patch("src.serving.api._load_features_df", return_value=_price_df()):
            body = client_with_lstm.post(
                "/predict/lstm", json={"symbol": "AAPL", "days_ahead": 10}
            ).json()
        assert len(body["predictions"]) == 10
        assert len(body["dates"])       == 10

    def test_unknown_model_returns_404(self, client_with_lstm):
        resp = client_with_lstm.post(
            "/predict/xgboost", json={"symbol": "AAPL", "days_ahead": 30}
        )
        assert resp.status_code == 404

    def test_dates_are_business_days(self, client_with_lstm):
        with patch("src.serving.api._load_features_df", return_value=_price_df()):
            body = client_with_lstm.post(
                "/predict/lstm", json={"symbol": "AAPL", "days_ahead": 10}
            ).json()
        for d in body["dates"]:
            assert pd.Timestamp(d).dayofweek < 5, f"{d} is a weekend"

    def test_second_call_returns_cached_true(self, client_with_lstm):
        with patch("src.serving.api._load_features_df", return_value=_price_df()):
            client_with_lstm.post("/predict/lstm", json={"symbol": "AAPL", "days_ahead": 30})
            body2 = client_with_lstm.post(
                "/predict/lstm", json={"symbol": "AAPL", "days_ahead": 30}
            ).json()
        assert body2["cached"] is True

    def test_first_call_returns_cached_false(self, client_with_lstm):
        from src.serving import cache as cache_mod
        cache_mod._reset_client()  # ensure clean cache
        # Use MSFT so there's no prior cache entry for this symbol
        with patch("src.serving.api._load_features_df",
                   return_value=_price_df(symbol="MSFT")):
            body = client_with_lstm.post(
                "/predict/lstm", json={"symbol": "MSFT", "days_ahead": 5}
            ).json()
        assert body["cached"] is False

    def test_arm_model_uses_dataframe_adapter(self, client_multi):
        with patch("src.serving.api._load_features_df", return_value=_price_df()):
            resp = client_multi.post(
                "/predict/arm", json={"symbol": "AAPL", "days_ahead": 10}
            )
        assert resp.status_code == 200
        # ARM mock returns DataFrame → should be adapted to list
        body = resp.json()
        assert isinstance(body["predictions"], list)


# ── /predict/best ─────────────────────────────────────────────────────────────

class TestPredictBest:
    def test_returns_200(self, client_with_lstm):
        with patch("src.serving.api._load_features_df", return_value=_price_df()):
            resp = client_with_lstm.post(
                "/predict/best", json={"symbol": "AAPL", "days_ahead": 30}
            )
        assert resp.status_code == 200

    def test_503_when_no_models_loaded(self, empty_client):
        resp = empty_client.post(
            "/predict/best", json={"symbol": "AAPL", "days_ahead": 30}
        )
        assert resp.status_code == 503


# ── /predict/compare ──────────────────────────────────────────────────────────

class TestPredictCompare:
    def test_returns_all_loaded_models(self, client_multi):
        with patch("src.serving.api._load_features_df", return_value=_price_df()):
            resp = client_multi.post(
                "/predict/compare", json={"symbol": "AAPL", "days_ahead": 10}
            )
        assert resp.status_code == 200
        body = resp.json()
        assert "lstm" in body["predictions"]
        assert "arm"  in body["predictions"]

    def test_predictions_dict_keys_are_model_types(self, client_multi):
        with patch("src.serving.api._load_features_df", return_value=_price_df()):
            body = client_multi.post(
                "/predict/compare", json={"symbol": "AAPL", "days_ahead": 5}
            ).json()
        for key in body["predictions"]:
            assert key in ("lstm", "tft", "timesfm", "arm")

    def test_metadata_present_per_model(self, client_multi):
        with patch("src.serving.api._load_features_df", return_value=_price_df()):
            body = client_multi.post(
                "/predict/compare", json={"symbol": "AAPL", "days_ahead": 5}
            ).json()
        assert "metadata" in body
        for model_type in body["predictions"]:
            assert model_type in body["metadata"]
            assert "cached" in body["metadata"][model_type]

    def test_dates_list_in_response(self, client_multi):
        with patch("src.serving.api._load_features_df", return_value=_price_df()):
            body = client_multi.post(
                "/predict/compare", json={"symbol": "AAPL", "days_ahead": 7}
            ).json()
        assert len(body["dates"]) == 7

    def test_503_when_no_models(self, empty_client):
        resp = empty_client.post(
            "/predict/compare", json={"symbol": "AAPL", "days_ahead": 5}
        )
        assert resp.status_code == 503

    def test_compare_uses_cache_on_repeat(self, client_multi):
        with patch("src.serving.api._load_features_df", return_value=_price_df()):
            client_multi.post("/predict/compare", json={"symbol": "AAPL", "days_ahead": 5})
            body2 = client_multi.post(
                "/predict/compare", json={"symbol": "AAPL", "days_ahead": 5}
            ).json()
        for meta in body2["metadata"].values():
            assert meta["cached"] is True


# ── /registry/* ───────────────────────────────────────────────────────────────

class TestRegistryEndpoints:
    def test_summary_returns_list(self, client_with_lstm):
        mock_df = pd.DataFrame([{
            "model_name": "stock_price_forecaster_arm",
            "model_type": "ARM", "version": "1",
            "stage": "Production", "run_id": "abc",
            "creation_timestamp": None, "val_loss": 0.05,
            "test_rmse": 8.0, "test_mape": 2.5,
            "test_directional_accuracy": 0.60,
        }])
        with patch("src.registry.model_registry.get_registry_summary", return_value=mock_df):
            resp = client_with_lstm.get("/registry/summary")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_summary_502_on_registry_error(self, client_with_lstm):
        with patch("src.registry.model_registry.get_registry_summary",
                   side_effect=Exception("MLflow down")):
            resp = client_with_lstm.get("/registry/summary")
        assert resp.status_code == 502

    def test_production_info_returns_200(self, client_with_lstm):
        with patch("src.serving.api._get_registry_info", return_value={
            "model_type": "LSTM", "version": "3",
            "stage": "Production", "run_id": "xyz",
            "model_uri": "models:/stock_price_forecaster_lstm/Production",
            "metrics": {},
        }):
            resp = client_with_lstm.get("/registry/lstm/production")
        assert resp.status_code == 200
        assert resp.json()["version"] == "3"

    def test_production_info_404_when_none(self, client_with_lstm):
        with patch("src.serving.api._get_registry_info", return_value=None):
            resp = client_with_lstm.get("/registry/lstm/production")
        assert resp.status_code == 404


# ── /admin/reload ─────────────────────────────────────────────────────────────

class TestAdminReload:
    def test_reload_returns_loaded_and_failed(self, client_with_lstm):
        def fake_load(name):
            if "lstm" in name:
                return _mock_lstm_model(), {"version": "4", "loaded_at": "2024-01-01"}
            raise RuntimeError("not registered")

        with patch("src.serving.api.load_model", side_effect=fake_load):
            resp = client_with_lstm.post("/admin/reload")
        assert resp.status_code == 200
        body = resp.json()
        assert "loaded" in body
        assert "failed" in body
        assert "stock_price_forecaster_lstm" in body["loaded"]

    def test_reload_updates_model_in_memory(self, client_with_lstm):
        from src.serving.api import models, model_metadata

        new_model = _mock_lstm_model()
        new_meta  = {"version": "99", "loaded_at": "2024-01-01"}

        with patch("src.serving.api.load_model", return_value=(new_model, new_meta)):
            client_with_lstm.post("/admin/reload")

        assert model_metadata.get("stock_price_forecaster_lstm", {}).get("version") == "99"


# ── /cache endpoints ──────────────────────────────────────────────────────────

class TestCacheEndpoints:
    def test_stats_returns_dict(self, client_with_lstm):
        resp = client_with_lstm.get("/cache/stats")
        assert resp.status_code == 200
        body = resp.json()
        assert "hits" in body or "backend" in body

    def test_flush_cache_returns_200(self, client_with_lstm):
        resp = client_with_lstm.delete("/cache")
        assert resp.status_code == 200

    def test_flush_with_symbol_filter(self, client_with_lstm):
        resp = client_with_lstm.delete("/cache?symbol=AAPL")
        assert resp.status_code == 200
        assert resp.json()["symbol"] == "AAPL"

    def test_flush_with_model_filter(self, client_with_lstm):
        resp = client_with_lstm.delete("/cache?model_type=lstm")
        assert resp.status_code == 200
        assert resp.json()["model_type"] == "lstm"

    def test_cache_cleared_after_flush(self, client_with_lstm):
        from src.serving import cache as cache_mod
        cache_mod._reset_client()

        with patch("src.serving.api._load_features_df", return_value=_price_df()):
            # Populate cache
            client_with_lstm.post("/predict/lstm", json={"symbol": "AAPL", "days_ahead": 5})
            # Flush
            client_with_lstm.delete("/cache")
            # Next request should be a cache miss
            body = client_with_lstm.post(
                "/predict/lstm", json={"symbol": "AAPL", "days_ahead": 5}
            ).json()
        assert body["cached"] is False


# ── /metrics ──────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_prometheus_format(self, client_with_lstm):
        resp = client_with_lstm.get("/metrics")
        assert resp.status_code == 200
        assert "api_requests_total" in resp.text or "TYPE" in resp.text


# ── cache.py unit tests ───────────────────────────────────────────────────────

class TestDictCache:
    def setup_method(self):
        from src.serving.cache import _DictCache
        self.cache = _DictCache()

    def test_set_and_get(self):
        self.cache.setex("k", 60, "hello")
        assert self.cache.get("k") == b"hello"

    def test_get_returns_none_for_missing(self):
        assert self.cache.get("nonexistent") is None

    def test_expired_entry_returns_none(self):
        self.cache.setex("k", 1, "value")
        self.cache._store["k"] = (self.cache._store["k"][0], time.time() - 1)
        assert self.cache.get("k") is None

    def test_delete_removes_entry(self):
        self.cache.setex("k", 60, "v")
        self.cache.delete("k")
        assert self.cache.get("k") is None

    def test_flushdb_clears_all(self):
        self.cache.setex("a", 60, "1")
        self.cache.setex("b", 60, "2")
        self.cache.flushdb()
        assert self.cache.get("a") is None
        assert self.cache.get("b") is None

    def test_stats_tracks_hits_and_misses(self):
        self.cache.setex("k", 60, "v")
        self.cache.get("k")    # hit
        self.cache.get("miss") # miss
        stats = self.cache.stats()
        assert stats["hits"]   == 1
        assert stats["misses"] == 1

    def test_hit_rate_computed_correctly(self):
        self.cache.setex("k", 60, "v")
        self.cache.get("k")   # hit
        self.cache.get("k")   # hit
        self.cache.get("x")   # miss
        stats = self.cache.stats()
        assert abs(stats["hit_rate"] - 2 / 3) < 0.01


class TestCachePublicAPI:
    def setup_method(self):
        from src.serving import cache as cache_mod
        cache_mod._reset_client()
        # Force dict backend (no Redis in test env)
        cache_mod._client_singleton = cache_mod._DictCache()

    def test_cache_key_format(self):
        from src.serving.cache import cache_key
        key = cache_key("AAPL", "lstm")
        assert key.startswith("stock:pred:AAPL:lstm:")

    def test_cache_key_uppercases_symbol(self):
        from src.serving.cache import cache_key
        assert cache_key("aapl", "lstm") == cache_key("AAPL", "lstm")

    def test_get_and_set_cached(self):
        from src.serving.cache import get_cached, set_cached, cache_key
        key  = cache_key("AAPL", "lstm")
        data = {"predictions": [1.0, 2.0], "model_version": "3"}
        set_cached(key, data)
        result = get_cached(key)
        assert result["predictions"] == [1.0, 2.0]
        assert result["model_version"] == "3"

    def test_get_cached_returns_none_on_miss(self):
        from src.serving.cache import get_cached
        assert get_cached("stock:pred:NONEXISTENT:lstm:2099-01-01") is None

    def test_invalidate_flushes_when_both_none(self):
        from src.serving.cache import get_cached, set_cached, invalidate, cache_key
        set_cached(cache_key("AAPL", "lstm"), {"predictions": []})
        invalidate()
        # In-process dict: flushdb called → all gone
        assert get_cached(cache_key("AAPL", "lstm")) is None

    def test_cache_stats_returns_dict(self):
        from src.serving.cache import cache_stats
        stats = cache_stats()
        assert isinstance(stats, dict)
        assert "hits" in stats


# ── _predict_lstm adapter (unit) ──────────────────────────────────────────────

class TestPredictLSTMAdapter:
    def test_returns_float_array(self):
        from src.serving.api import _predict_lstm
        df    = _price_df(n=150)
        model = _mock_lstm_model(n_out=30)
        result = _predict_lstm(model, "AAPL", df, n_steps_in=90, n_steps_out=30)
        assert hasattr(result, "__len__")
        assert len(result) == 30

    def test_raises_when_too_few_rows(self):
        from src.serving.api import _predict_lstm
        df    = _price_df(n=50)
        model = _mock_lstm_model()
        with pytest.raises(ValueError, match="Not enough rows"):
            _predict_lstm(model, "AAPL", df, n_steps_in=90, n_steps_out=30)


class TestPredictARMAdapter:
    def test_returns_list_of_floats(self):
        from src.serving.api import _predict_arm
        model = _mock_arm_model(n_out=30)
        result = _predict_arm(model, "AAPL", n_steps_out=30)
        assert len(result) == 30
        assert all(isinstance(v, float) for v in result)

    def test_truncates_to_n_steps_out(self):
        from src.serving.api import _predict_arm
        model = _mock_arm_model(n_out=30)
        result = _predict_arm(model, "AAPL", n_steps_out=10)
        assert len(result) == 10
