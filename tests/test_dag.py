"""
Tests for Step 12: Airflow DAG structure and task callable logic.

Split into two groups:
  1. DAG-structure tests — require Apache Airflow; skipped if not installed.
  2. Callable / notification tests — pure Python, always run.
"""

import pytest
from unittest.mock import MagicMock, patch

# ── Airflow availability guard ────────────────────────────────────────────────

def _airflow_available() -> bool:
    try:
        from airflow.models import DAG  # noqa: F401
        return True
    except Exception:
        return False

AIRFLOW_AVAILABLE = _airflow_available()
skip_no_airflow   = pytest.mark.skipif(
    not AIRFLOW_AVAILABLE,
    reason="Apache Airflow not installed in this environment",
)


# ── DAG fixture (only used when Airflow is available) ─────────────────────────

@pytest.fixture(scope="module")
def loaded_dag():
    if not AIRFLOW_AVAILABLE:
        pytest.skip("Airflow not available")
    # Add project root to sys.path so dags can import src.*
    import sys, os
    root = os.path.join(os.path.dirname(__file__), "..")
    if root not in sys.path:
        sys.path.insert(0, root)
    from airflow.dags.pipeline_dag import dag
    return dag


@pytest.fixture(scope="module")
def tasks(loaded_dag):
    return {t.task_id: t for t in loaded_dag.tasks}


# ══════════════════════════════════════════════════════════════════════════════
#  GROUP 1 — DAG structure (Airflow required)
# ══════════════════════════════════════════════════════════════════════════════

@skip_no_airflow
class TestDAGProperties:
    def test_dag_id(self, loaded_dag):
        assert loaded_dag.dag_id == "stock_forecast_pipeline"

    def test_schedule_is_weekday_evenings(self, loaded_dag):
        assert loaded_dag.schedule_interval == "0 18 * * 1-5"

    def test_catchup_disabled(self, loaded_dag):
        assert loaded_dag.catchup is False

    def test_max_active_runs_is_one(self, loaded_dag):
        assert loaded_dag.max_active_runs == 1

    def test_has_expected_tags(self, loaded_dag):
        assert "mlops" in loaded_dag.tags
        assert "stock" in loaded_dag.tags

    def test_default_owner(self, loaded_dag):
        assert loaded_dag.default_args["owner"] == "mlops"

    def test_default_retries(self, loaded_dag):
        assert loaded_dag.default_args["retries"] == 2

    def test_has_failure_callback(self, loaded_dag):
        assert loaded_dag.default_args.get("on_failure_callback") is not None


@skip_no_airflow
class TestTaskPresence:
    REQUIRED = [
        "scrape_live_data", "validate_data", "version_data",
        "clean_and_engineer_features",
        "train_lstm", "train_tft", "train_timesfm", "train_arm",
        "compare_models", "register_best_model", "deploy_to_api",
        "check_drift", "branch_on_drift",
        "trigger_retrain", "skip_retrain", "send_report",
    ]

    def test_all_required_tasks_present(self, tasks):
        missing = [t for t in self.REQUIRED if t not in tasks]
        assert not missing, f"Missing tasks: {missing}"


@skip_no_airflow
class TestTaskTypes:
    def test_branch_on_drift_is_branch_operator(self, tasks):
        from airflow.operators.python import BranchPythonOperator
        assert isinstance(tasks["branch_on_drift"], BranchPythonOperator)

    def test_trigger_retrain_is_trigger_operator(self, tasks):
        from airflow.operators.trigger_dagrun import TriggerDagRunOperator
        assert isinstance(tasks["trigger_retrain"], TriggerDagRunOperator)

    def test_skip_retrain_is_empty_operator(self, tasks):
        from airflow.operators.empty import EmptyOperator
        assert isinstance(tasks["skip_retrain"], EmptyOperator)


@skip_no_airflow
class TestDependencyGraph:
    def _up(self, task) -> set:
        return {t.task_id for t in task.upstream_list}

    def test_linear_ingestion_chain(self, tasks):
        assert "scrape_live_data" in self._up(tasks["validate_data"])
        assert "validate_data"    in self._up(tasks["version_data"])
        assert "version_data"     in self._up(tasks["clean_and_engineer_features"])

    def test_all_trainers_depend_on_features(self, tasks):
        for tid in ("train_lstm", "train_tft", "train_timesfm", "train_arm"):
            assert "clean_and_engineer_features" in self._up(tasks[tid])

    def test_compare_depends_on_all_trainers(self, tasks):
        up = self._up(tasks["compare_models"])
        for tid in ("train_lstm", "train_tft", "train_timesfm", "train_arm"):
            assert tid in up

    def test_post_compare_chain(self, tasks):
        assert "compare_models"    in self._up(tasks["register_best_model"])
        assert "register_best_model" in self._up(tasks["deploy_to_api"])
        assert "deploy_to_api"     in self._up(tasks["check_drift"])
        assert "check_drift"       in self._up(tasks["branch_on_drift"])

    def test_branch_feeds_both_leaves(self, tasks):
        assert "branch_on_drift" in self._up(tasks["trigger_retrain"])
        assert "branch_on_drift" in self._up(tasks["skip_retrain"])

    def test_report_joins_both_branches(self, tasks):
        up = self._up(tasks["send_report"])
        assert "trigger_retrain" in up
        assert "skip_retrain"    in up

    def test_training_tasks_are_parallel(self, tasks):
        training = {"train_lstm", "train_tft", "train_timesfm", "train_arm"}
        for tid in training:
            assert not training.intersection(self._up(tasks[tid]) - {tid})

    def test_trigger_retrain_points_to_same_dag(self, tasks):
        assert tasks["trigger_retrain"].trigger_dag_id == "stock_forecast_pipeline"


@skip_no_airflow
class TestTriggerRules:
    def test_compare_requires_all_success(self, tasks):
        from airflow.utils.trigger_rule import TriggerRule
        assert tasks["compare_models"].trigger_rule == TriggerRule.ALL_SUCCESS

    def test_send_report_joins_branches(self, tasks):
        from airflow.utils.trigger_rule import TriggerRule
        assert tasks["send_report"].trigger_rule == TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS


@skip_no_airflow
class TestExecutionTimeouts:
    def test_training_tasks_have_timeouts(self, tasks):
        for tid in ("train_lstm", "train_tft", "train_timesfm", "train_arm"):
            assert tasks[tid].execution_timeout is not None

    def test_tft_timeout_is_longest(self, tasks):
        assert tasks["train_tft"].execution_timeout >= tasks["train_lstm"].execution_timeout

    def test_deploy_and_compare_have_timeouts(self, tasks):
        assert tasks["deploy_to_api"].execution_timeout is not None
        assert tasks["compare_models"].execution_timeout is not None


# ══════════════════════════════════════════════════════════════════════════════
#  GROUP 2 — Task callables (no Airflow required)
# ══════════════════════════════════════════════════════════════════════════════

class TestDeployToAPI:
    def test_pushes_reloaded_models_on_success(self):
        from src.pipelines.tasks import task_deploy_to_api

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"loaded": ["stock_price_forecaster_arm"], "failed": []}
        mock_resp.raise_for_status.return_value = None
        mock_ti = MagicMock()

        with patch("requests.post", return_value=mock_resp):
            task_deploy_to_api(ti=mock_ti)

        mock_ti.xcom_push.assert_called_with(
            key="reloaded_models",
            value=["stock_price_forecaster_arm"],
        )

    def test_non_fatal_on_connection_error(self):
        import requests
        from src.pipelines.tasks import task_deploy_to_api

        mock_ti = MagicMock()
        with patch("requests.post", side_effect=requests.exceptions.ConnectionError):
            task_deploy_to_api(ti=mock_ti)   # must not raise

        mock_ti.xcom_push.assert_called_with(key="reloaded_models", value=[])

    def test_non_fatal_on_generic_error(self):
        from src.pipelines.tasks import task_deploy_to_api

        mock_ti = MagicMock()
        with patch("requests.post", side_effect=Exception("timeout")):
            task_deploy_to_api(ti=mock_ti)   # must not raise

        mock_ti.xcom_push.assert_called_with(key="reloaded_models", value=[])

    def test_uses_api_url_env_var(self):
        from src.pipelines.tasks import task_deploy_to_api

        captured = {}
        def fake_post(url, **kw):
            captured["url"] = url
            m = MagicMock()
            m.json.return_value = {"loaded": [], "failed": []}
            m.raise_for_status.return_value = None
            return m

        with patch("requests.post", side_effect=fake_post), \
             patch.dict("os.environ", {"API_URL": "http://api-prod:8000"}):
            task_deploy_to_api(ti=MagicMock())

        assert "api-prod:8000" in captured["url"]


class TestBranchOnDrift:
    def test_returns_trigger_retrain_when_drift(self):
        from src.pipelines.tasks import task_branch_on_drift

        ti = MagicMock()
        ti.xcom_pull.return_value = True
        assert task_branch_on_drift(ti=ti) == "trigger_retrain"

    def test_returns_skip_retrain_when_no_drift(self):
        from src.pipelines.tasks import task_branch_on_drift

        ti = MagicMock()
        ti.xcom_pull.return_value = False
        assert task_branch_on_drift(ti=ti) == "skip_retrain"

    def test_returns_skip_when_xcom_is_none(self):
        from src.pipelines.tasks import task_branch_on_drift

        ti = MagicMock()
        ti.xcom_pull.return_value = None
        assert task_branch_on_drift(ti=ti) == "skip_retrain"

    def test_pulls_from_check_drift_task(self):
        from src.pipelines.tasks import task_branch_on_drift

        ti = MagicMock()
        ti.xcom_pull.return_value = False
        task_branch_on_drift(ti=ti)
        ti.xcom_pull.assert_called_with(task_ids="check_drift", key="needs_retraining")


class TestTaskCheckDrift:
    def test_pushes_needs_retraining_to_xcom(self):
        import sys
        from types import ModuleType
        from src.pipelines.tasks import task_check_drift

        mock_result = {
            "needs_retraining": True,
            "drift_score":      0.45,
            "drift_detected":   True,
        }

        # Stub the entire drift module so task_check_drift can import from it
        # even when evidently is not installed in the test environment.
        # task_check_drift does a local import, so sys.modules is consulted at call time.
        fake_drift = ModuleType("src.monitoring.drift")
        fake_drift.load_reference_data   = MagicMock(return_value=MagicMock())
        fake_drift.load_current_data     = MagicMock(return_value=MagicMock())
        fake_drift.run_data_drift_report = MagicMock(return_value=mock_result)

        mock_ti = MagicMock()
        with patch.dict(sys.modules, {
            "src.monitoring.drift": fake_drift,
            "evidently": ModuleType("evidently"),
            "evidently.report": ModuleType("evidently.report"),
            "evidently.metric_preset": ModuleType("evidently.metric_preset"),
        }):
            task_check_drift(ti=mock_ti)

        calls = {c[1]["key"]: c[1]["value"] for c in mock_ti.xcom_push.call_args_list}
        assert calls["needs_retraining"] is True
        assert abs(calls["drift_score"] - 0.45) < 1e-6


# ══════════════════════════════════════════════════════════════════════════════
#  GROUP 3 — Notifications
# ══════════════════════════════════════════════════════════════════════════════

class TestSlackNotifications:
    def test_send_alert_noop_without_webhook(self):
        import src.notifications.slack as slack_mod
        with patch.object(slack_mod, "SLACK_WEBHOOK_URL", ""):
            result = slack_mod.send_alert("test")
        assert result is False

    def test_send_alert_posts_when_webhook_set(self):
        import src.notifications.slack as slack_mod

        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        with patch.object(slack_mod, "SLACK_WEBHOOK_URL", "https://hooks.slack.com/x"), \
             patch("requests.post", return_value=mock_resp) as mock_post:
            result = slack_mod.send_alert("hello")

        assert result is True
        mock_post.assert_called_once()

    def test_pipeline_summary_includes_winner_model(self):
        import src.notifications.slack as slack_mod

        captured = {}
        def fake_post(payload):
            captured["p"] = payload
            return True

        summary = {
            "dag_id": "stock_forecast_pipeline",
            "winner_model": "ARM", "winner_version": "3",
            "winner_rmse": 8.5, "drift_score": 0.15,
            "needs_retraining": False, "duration_seconds": 3600,
            "models_reloaded": [], "tasks_failed": [],
        }
        with patch.object(slack_mod, "_post", side_effect=fake_post):
            slack_mod.send_pipeline_summary(summary)

        assert "ARM" in str(captured.get("p", ""))

    def test_task_failure_includes_task_id(self):
        import src.notifications.slack as slack_mod

        captured = {}
        def fake_post(payload):
            captured["p"] = payload
            return True

        mock_ti  = MagicMock(); mock_ti.task_id = "train_lstm"
        mock_dag = MagicMock(); mock_dag.dag_id  = "stock_forecast_pipeline"
        context  = {
            "task_instance": mock_ti, "dag": mock_dag,
            "execution_date": "2024-01-15", "exception": "OOM",
        }
        with patch.object(slack_mod, "_post", side_effect=fake_post):
            slack_mod.send_task_failure(context)

        assert "train_lstm" in str(captured.get("p", ""))

    def test_send_alert_does_not_raise_on_network_error(self):
        import src.notifications.slack as slack_mod
        with patch.object(slack_mod, "SLACK_WEBHOOK_URL", "https://hooks.slack.com/x"), \
             patch("requests.post", side_effect=Exception("network error")):
            result = slack_mod.send_alert("test")  # must not raise
        assert result is False


class TestReportModule:
    def test_build_summary_returns_dict_with_required_keys(self):
        from src.notifications.report import build_summary

        mock_ti  = MagicMock(); mock_ti.xcom_pull.return_value = None
        mock_dag = MagicMock(); mock_dag.dag_id = "stock_forecast_pipeline"
        context  = {"ti": mock_ti, "dag": mock_dag, "dag_run": None, "execution_date": "2024-01-15"}

        with patch("src.registry.model_registry.get_latest_production", return_value=None):
            summary = build_summary(context)

        for key in ("dag_id", "winner_model", "needs_retraining", "models_reloaded"):
            assert key in summary

    def test_build_summary_dag_id_from_context(self):
        from src.notifications.report import build_summary

        mock_ti  = MagicMock(); mock_ti.xcom_pull.return_value = None
        mock_dag = MagicMock(); mock_dag.dag_id = "custom_pipeline"
        context  = {"ti": mock_ti, "dag": mock_dag, "dag_run": None, "execution_date": ""}

        with patch("src.registry.model_registry.get_latest_production", return_value=None):
            summary = build_summary(context)

        assert summary["dag_id"] == "custom_pipeline"
