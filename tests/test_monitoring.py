"""
Tests for Step 13: Monitoring stack.

Covers:
  - src.monitoring.metrics — Prometheus metric helpers
  - src.monitoring.drift   — drift report parsing (Evidently stubbed)
  - src.serving.api        — POST /monitoring/push and GET /monitoring/drift
  - monitoring/alerts      — alert rule YAML structure
"""

import sys
import pytest
import yaml
from types import ModuleType
from unittest.mock import MagicMock, patch


# ── Evidently stub (must run before any import of src.monitoring.drift) ──────

def _stub_evidently():
    ev = ModuleType("evidently")
    sys.modules.setdefault("evidently", ev)

    report_mod         = sys.modules.setdefault("evidently.report", ModuleType("evidently.report"))
    preset_mod         = sys.modules.setdefault("evidently.metric_preset", ModuleType("evidently.metric_preset"))
    metrics_mod        = sys.modules.setdefault("evidently.metrics", ModuleType("evidently.metrics"))

    # Provide placeholder classes so `from evidently.xxx import Yyy` succeeds
    report_mod.Report            = MagicMock()
    preset_mod.DataDriftPreset   = MagicMock()
    preset_mod.RegressionPreset  = MagicMock()
    metrics_mod.DatasetDriftMetric = MagicMock()
    metrics_mod.ColumnDriftMetric  = MagicMock()

_stub_evidently()

# Pre-import drift module so patch() can resolve it by attribute
import src.monitoring.drift as _drift_mod  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
#  Prometheus metrics module
# ══════════════════════════════════════════════════════════════════════════════

class TestMetricsModule:
    def test_push_drift_metrics_sets_gauges(self):
        from src.monitoring.metrics import (
            DRIFT_SCORE, DRIFT_DETECTED, NEEDS_RETRAINING, push_drift_metrics,
        )
        drift_result = {
            "drift_score":      0.42,
            "drift_detected":   True,
            "needs_retraining": True,
        }
        push_drift_metrics(drift_result, dataset="test_ds")

        assert DRIFT_SCORE.labels(dataset="test_ds")._value.get()    == pytest.approx(0.42)
        assert DRIFT_DETECTED.labels(dataset="test_ds")._value.get() == 1.0
        assert NEEDS_RETRAINING.labels(dataset="test_ds")._value.get() == 1.0

    def test_push_drift_metrics_no_drift(self):
        from src.monitoring.metrics import (
            DRIFT_SCORE, DRIFT_DETECTED, NEEDS_RETRAINING, push_drift_metrics,
        )
        push_drift_metrics(
            {"drift_score": 0.1, "drift_detected": False, "needs_retraining": False},
            dataset="test_ds2",
        )
        assert DRIFT_DETECTED.labels(dataset="test_ds2")._value.get()    == 0.0
        assert NEEDS_RETRAINING.labels(dataset="test_ds2")._value.get()  == 0.0

    def test_push_model_metrics_sets_rmse_mape_dir_acc(self):
        from src.monitoring.metrics import (
            MODEL_RMSE, MODEL_MAPE, MODEL_DIR_ACCURACY, push_model_metrics,
        )
        push_model_metrics("lstm", {
            "test_rmse": 7.5,
            "test_mape": 3.2,
            "test_directional_accuracy": 0.63,
        })
        assert MODEL_RMSE.labels(model_type="LSTM")._value.get()         == pytest.approx(7.5)
        assert MODEL_MAPE.labels(model_type="LSTM")._value.get()         == pytest.approx(3.2)
        assert MODEL_DIR_ACCURACY.labels(model_type="LSTM")._value.get() == pytest.approx(0.63)

    def test_push_model_metrics_partial_keys(self):
        from src.monitoring.metrics import MODEL_RMSE, push_model_metrics
        # Should not raise even when only one key is present
        push_model_metrics("arm", {"test_rmse": 9.1})
        assert MODEL_RMSE.labels(model_type="ARM")._value.get() == pytest.approx(9.1)

    def test_push_pipeline_run_metrics_increments_counter(self):
        from src.monitoring.metrics import (
            PIPELINE_RUNS_TOTAL, PIPELINE_DURATION, push_pipeline_run_metrics,
        )
        before = PIPELINE_RUNS_TOTAL._value.get()
        push_pipeline_run_metrics({
            "drift_score":      0.2,
            "drift_detected":   False,
            "needs_retraining": False,
            "winner_model":     "ARM",
            "winner_version":   "5",
            "winner_rmse":      8.8,
            "duration_seconds": 3600.0,
        })
        assert PIPELINE_RUNS_TOTAL._value.get() == before + 1
        assert PIPELINE_DURATION._value.get()   == pytest.approx(3600.0)

    def test_push_pipeline_run_metrics_retraining_counter(self):
        from src.monitoring.metrics import (
            RETRAINING_TRIGGERED_TOTAL, push_pipeline_run_metrics,
        )
        before = RETRAINING_TRIGGERED_TOTAL._value.get()
        push_pipeline_run_metrics({
            "drift_score":      0.6,
            "drift_detected":   True,
            "needs_retraining": True,
            "winner_model":     "LSTM",
            "winner_version":   "2",
            "winner_rmse":      None,
            "duration_seconds": None,
        })
        assert RETRAINING_TRIGGERED_TOTAL._value.get() == before + 1

    def test_push_pipeline_run_metrics_no_retraining_does_not_increment(self):
        from src.monitoring.metrics import (
            RETRAINING_TRIGGERED_TOTAL, push_pipeline_run_metrics,
        )
        before = RETRAINING_TRIGGERED_TOTAL._value.get()
        push_pipeline_run_metrics({
            "drift_score": 0.1, "drift_detected": False,
            "needs_retraining": False, "winner_model": None,
            "winner_version": None, "winner_rmse": None, "duration_seconds": None,
        })
        assert RETRAINING_TRIGGERED_TOTAL._value.get() == before


# ══════════════════════════════════════════════════════════════════════════════
#  Drift module
# ══════════════════════════════════════════════════════════════════════════════

def _make_fake_report(drift_score=0.35, drift_detected=True):
    """
    Build a mock evidently Report instance with the given drift score and patch
    the Report *name* inside src.monitoring.drift (which was bound at import time
    via `from evidently.report import Report`).
    """
    import src.monitoring.drift as drift_mod

    fake_instance = MagicMock()
    fake_instance.run       = MagicMock()
    fake_instance.save_html = MagicMock()
    fake_instance.as_dict.return_value = {
        "metrics": [{"result": {
            "dataset_drift":            drift_detected,
            "share_of_drifted_columns": float(drift_score),
        }}]
    }
    # Patch the module-level name, not sys.modules — `from x import y` binds y at import time
    drift_mod.Report = MagicMock(return_value=fake_instance)
    return fake_instance


class TestDriftFunctions:
    def test_run_data_drift_report_returns_required_keys(self, tmp_path):
        import pandas as pd
        import src.monitoring.drift as drift_mod

        _make_fake_report()
        df = pd.DataFrame({"price": [100, 101], "sma_7": [100, 101],
                           "sma_30": [100, 101], "returns": [0, 0.01],
                           "volatility": [0.1, 0.1]})

        with patch.object(drift_mod, "REPORTS_PATH", str(tmp_path)):
            result = drift_mod.run_data_drift_report(df, df.copy())

        for key in ("drift_detected", "drift_score", "needs_retraining", "report_path"):
            assert key in result

    def test_run_data_drift_report_drift_detected_is_bool(self, tmp_path):
        import pandas as pd
        import src.monitoring.drift as drift_mod

        _make_fake_report()
        df = pd.DataFrame({"price": [1], "sma_7": [1], "sma_30": [1],
                           "returns": [0.0], "volatility": [0.1]})
        with patch.object(drift_mod, "REPORTS_PATH", str(tmp_path)):
            result = drift_mod.run_data_drift_report(df, df)

        assert isinstance(result["drift_detected"], bool)
        assert isinstance(result["drift_score"],    float)

    def test_drift_threshold_triggers_retraining(self, tmp_path):
        import pandas as pd
        import src.monitoring.drift as drift_mod

        _make_fake_report(drift_score=0.35, drift_detected=True)
        df = pd.DataFrame({"price": [1], "sma_7": [1], "sma_30": [1],
                           "returns": [0.0], "volatility": [0.1]})
        original = drift_mod.DRIFT_THRESHOLD
        try:
            drift_mod.DRIFT_THRESHOLD = 0.3
            with patch.object(drift_mod, "REPORTS_PATH", str(tmp_path)):
                result = drift_mod.run_data_drift_report(df, df)
            assert result["needs_retraining"] is True
        finally:
            drift_mod.DRIFT_THRESHOLD = original

    def test_drift_below_threshold_no_retraining(self, tmp_path):
        import pandas as pd
        import src.monitoring.drift as drift_mod

        _make_fake_report(drift_score=0.1, drift_detected=False)
        df = pd.DataFrame({"price": [1], "sma_7": [1], "sma_30": [1],
                           "returns": [0.0], "volatility": [0.1]})
        original = drift_mod.DRIFT_THRESHOLD
        try:
            drift_mod.DRIFT_THRESHOLD = 0.3
            with patch.object(drift_mod, "REPORTS_PATH", str(tmp_path)):
                result = drift_mod.run_data_drift_report(df, df)
            assert result["needs_retraining"] is False
        finally:
            drift_mod.DRIFT_THRESHOLD = original


# ══════════════════════════════════════════════════════════════════════════════
#  API monitoring endpoints
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def api_client():
    from starlette.testclient import TestClient
    from src.serving.api import app
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


class TestMonitoringPushEndpoint:
    def test_push_returns_ok_status(self, api_client):
        resp = api_client.post("/monitoring/push", json={
            "drift_score":      0.25,
            "drift_detected":   False,
            "needs_retraining": False,
            "winner_model":     "ARM",
            "winner_version":   "3",
            "winner_rmse":      9.1,
            "duration_seconds": 3612.0,
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_push_empty_payload_accepted(self, api_client):
        resp = api_client.post("/monitoring/push", json={})
        assert resp.status_code == 200

    def test_push_updates_prometheus_gauge(self, api_client):
        from src.monitoring.metrics import DRIFT_SCORE
        api_client.post("/monitoring/push", json={
            "drift_score": 0.77,
            "drift_detected": True,
            "needs_retraining": True,
        })
        assert DRIFT_SCORE.labels(dataset="price_features")._value.get() == pytest.approx(0.77)

    def test_push_model_metrics_per_model(self, api_client):
        from src.monitoring.metrics import MODEL_RMSE
        api_client.post("/monitoring/push", json={
            "model_metrics": {
                "TFT": {"test_rmse": 6.3, "test_mape": 2.1}
            }
        })
        assert MODEL_RMSE.labels(model_type="TFT")._value.get() == pytest.approx(6.3)

    def test_push_increments_pipeline_counter(self, api_client):
        from src.monitoring.metrics import PIPELINE_RUNS_TOTAL
        before = PIPELINE_RUNS_TOTAL._value.get()
        api_client.post("/monitoring/push", json={"duration_seconds": 100.0})
        assert PIPELINE_RUNS_TOTAL._value.get() == before + 1

    def test_metrics_endpoint_exposes_pipeline_gauges(self, api_client):
        api_client.post("/monitoring/push", json={
            "drift_score": 0.31, "drift_detected": True, "needs_retraining": True,
        })
        resp = api_client.get("/metrics")
        assert resp.status_code == 200
        assert "pipeline_drift_score" in resp.text
        assert "pipeline_needs_retraining" in resp.text


class TestMonitoringDriftEndpoint:
    def test_drift_endpoint_returns_required_keys(self, api_client, tmp_path):
        import pandas as pd
        import src.monitoring.drift as drift_mod

        fake_df = pd.DataFrame({
            "price": range(10), "sma_7": range(10), "sma_30": range(10),
            "returns": [0.0] * 10, "volatility": [0.1] * 10,
        })
        mock_result = {
            "drift_detected": False, "drift_score": 0.1,
            "needs_retraining": False, "report_path": str(tmp_path / "r.html"),
        }
        with patch.object(drift_mod, "load_reference_data",   return_value=fake_df), \
             patch.object(drift_mod, "load_current_data",     return_value=fake_df), \
             patch.object(drift_mod, "run_data_drift_report", return_value=mock_result):
            resp = api_client.get("/monitoring/drift")

        assert resp.status_code == 200
        for key in ("drift_detected", "drift_score", "needs_retraining"):
            assert key in resp.json()

    def test_drift_endpoint_returns_500_on_error(self, api_client):
        import src.monitoring.drift as drift_mod
        with patch.object(drift_mod, "load_reference_data", side_effect=FileNotFoundError("no data")):
            resp = api_client.get("/monitoring/drift")
        assert resp.status_code == 500


# ══════════════════════════════════════════════════════════════════════════════
#  Alert rules YAML
# ══════════════════════════════════════════════════════════════════════════════

class TestAlertRules:
    @pytest.fixture(scope="class")
    def rules(self):
        path = "monitoring/alerts/rules.yml"
        with open(path) as f:
            return yaml.safe_load(f)

    def test_yaml_is_valid(self, rules):
        assert rules is not None
        assert "groups" in rules

    def test_has_required_groups(self, rules):
        group_names = {g["name"] for g in rules["groups"]}
        for expected in ("api_health", "drift_monitoring", "model_performance", "pipeline_lifecycle"):
            assert expected in group_names

    def test_all_alerts_have_summary_annotation(self, rules):
        for group in rules["groups"]:
            for rule in group.get("rules", []):
                if "alert" in rule:
                    assert "summary" in rule.get("annotations", {}), \
                        f"{rule['alert']} missing summary annotation"

    def test_drift_threshold_is_reasonable(self, rules):
        drift_group = next(g for g in rules["groups"] if g["name"] == "drift_monitoring")
        high_drift  = next(r for r in drift_group["rules"] if r.get("alert") == "HighDataDrift")
        # Threshold should be between 0.3 and 0.9
        expr = high_drift["expr"]
        assert "0." in expr

    def test_api_error_rate_severity_is_warning(self, rules):
        api_group = next(g for g in rules["groups"] if g["name"] == "api_health")
        error_rule = next(r for r in api_group["rules"] if r.get("alert") == "APIHighErrorRate")
        assert error_rule["labels"]["severity"] == "warning"

    def test_critical_alerts_exist(self, rules):
        all_rules = [r for g in rules["groups"] for r in g.get("rules", [])]
        critical  = [r for r in all_rules if r.get("labels", {}).get("severity") == "critical"]
        assert len(critical) >= 1, "At least one critical alert should be defined"

    def test_all_alerts_have_for_duration(self, rules):
        for group in rules["groups"]:
            for rule in group.get("rules", []):
                if "alert" in rule:
                    assert "for" in rule, f"{rule['alert']} missing 'for' field"
