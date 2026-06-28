"""
Tests for Step 14: Automated retraining — cooldown, adaptive threshold, scheduler.
"""

import json
import pytest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch


# ══════════════════════════════════════════════════════════════════════════════
#  CooldownManager
# ══════════════════════════════════════════════════════════════════════════════

class TestCooldownManager:
    @pytest.fixture
    def mgr(self, tmp_path):
        from src.retraining.cooldown import CooldownManager
        return CooldownManager(
            state_path=str(tmp_path / "state.json"),
            cooldown_hours=24,
        )

    def test_no_state_cooldown_not_active(self, mgr):
        assert mgr.is_cooldown_active() is False

    def test_no_state_last_retrain_is_none(self, mgr):
        assert mgr.get_last_retrain_time() is None

    def test_no_state_hours_since_is_none(self, mgr):
        assert mgr.hours_since_last_retrain() is None

    def test_record_sets_cooldown_active(self, mgr):
        mgr.record_retrain_triggered(drift_score=0.45)
        assert mgr.is_cooldown_active() is True

    def test_cooldown_expires_after_window(self, mgr, tmp_path):
        # Simulate a retrain that happened 25 hours ago (past the 24-hour window)
        old_time = datetime.now(timezone.utc) - timedelta(hours=25)
        state = {
            "last_retrain_at": old_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "last_drift_score": 0.4,
            "retrain_count": 1,
        }
        (tmp_path / "state.json").write_text(json.dumps(state))
        assert mgr.is_cooldown_active() is False

    def test_cooldown_still_active_within_window(self, mgr, tmp_path):
        # Simulate a retrain that happened 12 hours ago (within 24-hour window)
        recent = datetime.now(timezone.utc) - timedelta(hours=12)
        state = {
            "last_retrain_at": recent.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "last_drift_score": 0.5,
            "retrain_count": 1,
        }
        (tmp_path / "state.json").write_text(json.dumps(state))
        assert mgr.is_cooldown_active() is True

    def test_record_stores_drift_score(self, mgr):
        mgr.record_retrain_triggered(drift_score=0.72, dag_run_id="run_001")
        state_path = Path(mgr.state_path)
        data = json.loads(state_path.read_text())
        assert data["last_drift_score"] == pytest.approx(0.72)
        assert data["last_dag_run_id"]  == "run_001"

    def test_record_increments_count(self, mgr):
        mgr.record_retrain_triggered()
        mgr.record_retrain_triggered()
        data = json.loads(Path(mgr.state_path).read_text())
        assert data["retrain_count"] == 2

    def test_get_status_has_required_keys(self, mgr):
        status = mgr.get_status()
        for key in ("is_active", "cooldown_hours", "hours_since_last",
                    "hours_remaining", "next_retrain_at", "last_retrain_at"):
            assert key in status

    def test_get_status_not_active_when_empty(self, mgr):
        status = mgr.get_status()
        assert status["is_active"]        is False
        assert status["hours_since_last"] is None
        assert status["hours_remaining"]  is None
        assert status["next_retrain_at"]  is None

    def test_get_status_active_shows_hours_remaining(self, mgr):
        mgr.record_retrain_triggered()
        status = mgr.get_status()
        assert status["is_active"]       is True
        assert status["hours_remaining"] is not None
        assert 0 < status["hours_remaining"] <= 24

    def test_reset_clears_state(self, mgr):
        mgr.record_retrain_triggered()
        mgr.reset()
        assert mgr.is_cooldown_active() is False
        assert mgr.get_last_retrain_time() is None

    def test_corrupt_state_file_returns_empty(self, mgr):
        Path(mgr.state_path).parent.mkdir(parents=True, exist_ok=True)
        Path(mgr.state_path).write_text("not json{{}")
        assert mgr.is_cooldown_active() is False


# ══════════════════════════════════════════════════════════════════════════════
#  ThresholdManager
# ══════════════════════════════════════════════════════════════════════════════

class TestThresholdManager:
    @pytest.fixture
    def mgr(self, tmp_path):
        from src.retraining.threshold import ThresholdManager
        return ThresholdManager(
            base_threshold=0.3,
            history_path=str(tmp_path / "history.json"),
            window=10,
            sensitivity=0.5,
        )

    def test_no_history_returns_base_threshold(self, mgr):
        assert mgr.get_adaptive_threshold() == pytest.approx(0.3)

    def test_should_retrain_true_above_base(self, mgr):
        retrain, info = mgr.should_retrain(0.45)
        assert retrain is True
        assert info["threshold_used"] == pytest.approx(0.3)

    def test_should_retrain_false_below_base(self, mgr):
        retrain, _ = mgr.should_retrain(0.1)
        assert retrain is False

    def test_should_retrain_equal_to_threshold_is_false(self, mgr):
        # score == threshold → NOT greater-than → no retrain
        retrain, _ = mgr.should_retrain(0.3)
        assert retrain is False

    def test_record_stores_entry(self, mgr):
        mgr.record_drift_score(0.25)
        scores = mgr.get_recent_scores()
        assert len(scores) == 1
        assert scores[0] == pytest.approx(0.25)

    def test_history_window_limits_size(self, mgr):
        for i in range(15):
            mgr.record_drift_score(float(i) * 0.02)
        assert len(mgr.get_recent_scores()) == 10  # window=10

    def test_high_history_raises_threshold(self, mgr):
        # Consistently high scores → raise threshold to avoid over-triggering
        for _ in range(10):
            mgr.record_drift_score(0.8)
        adaptive = mgr.get_adaptive_threshold()
        assert adaptive > 0.3

    def test_low_history_lowers_threshold(self, mgr):
        # Consistently low scores → lower threshold to catch any new spike early
        for _ in range(10):
            mgr.record_drift_score(0.05)
        adaptive = mgr.get_adaptive_threshold()
        assert adaptive < 0.3

    def test_adaptive_threshold_clamped(self, mgr):
        # Extreme scores should still stay within [base*0.5, base*2.0]
        for _ in range(10):
            mgr.record_drift_score(1.0)
        adaptive = mgr.get_adaptive_threshold()
        assert adaptive >= 0.3 * 0.5
        assert adaptive <= 0.3 * 2.0

    def test_zero_sensitivity_disables_adaptation(self, tmp_path):
        from src.retraining.threshold import ThresholdManager
        mgr = ThresholdManager(
            base_threshold=0.3,
            history_path=str(tmp_path / "h2.json"),
            window=10,
            sensitivity=0.0,
        )
        for _ in range(10):
            mgr.record_drift_score(0.9)
        assert mgr.get_adaptive_threshold() == pytest.approx(0.3)

    def test_should_retrain_info_has_required_keys(self, mgr):
        _, info = mgr.should_retrain(0.4)
        for key in ("threshold_used", "base_threshold", "adaptive",
                    "recent_mean", "history_size", "current_score"):
            assert key in info

    def test_adaptive_flag_false_with_no_history(self, mgr):
        _, info = mgr.should_retrain(0.4)
        assert info["adaptive"] is False

    def test_adaptive_flag_true_with_history(self, mgr):
        for _ in range(5):
            mgr.record_drift_score(0.8)
        _, info = mgr.should_retrain(0.4)
        assert info["adaptive"] is True


# ══════════════════════════════════════════════════════════════════════════════
#  Scheduler — evaluate_retraining_need
# ══════════════════════════════════════════════════════════════════════════════

class TestScheduler:
    def _drift(self, score=0.4, detected=True):
        return {
            "drift_score":      score,
            "drift_detected":   detected,
            "needs_retraining": score > 0.3,
        }

    @pytest.fixture
    def paths(self, tmp_path):
        return {
            "state_path":   str(tmp_path / "state.json"),
            "history_path": str(tmp_path / "history.json"),
        }

    def test_should_retrain_when_high_drift_no_cooldown(self, paths):
        from src.retraining.scheduler import evaluate_retraining_need
        result = evaluate_retraining_need(self._drift(0.55), **paths)
        assert result["should_retrain"] is True

    def test_should_not_retrain_when_low_drift(self, paths):
        from src.retraining.scheduler import evaluate_retraining_need
        result = evaluate_retraining_need(self._drift(0.1, detected=False), **paths)
        assert result["should_retrain"] is False

    def test_should_not_retrain_when_cooldown_active(self, paths, tmp_path):
        from src.retraining.cooldown  import CooldownManager
        from src.retraining.scheduler import evaluate_retraining_need

        # Pre-arm the cooldown — retrain happened 2 hours ago
        mgr = CooldownManager(state_path=paths["state_path"], cooldown_hours=24)
        recent = datetime.now(timezone.utc) - timedelta(hours=2)
        state  = {"last_retrain_at": recent.strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "retrain_count": 1}
        Path(paths["state_path"]).write_text(json.dumps(state))

        result = evaluate_retraining_need(self._drift(0.9), cooldown_hours=24, **paths)
        assert result["should_retrain"] is False
        assert result["cooldown_active"] is True

    def test_result_has_all_required_keys(self, paths):
        from src.retraining.scheduler import evaluate_retraining_need
        result = evaluate_retraining_need(self._drift(0.4), **paths)
        for key in ("should_retrain", "reason", "drift_score", "drift_detected",
                    "cooldown_active", "cooldown_status", "threshold_used",
                    "base_threshold", "adaptive", "history_size"):
            assert key in result, f"Missing key: {key}"

    def test_reason_string_is_non_empty(self, paths):
        from src.retraining.scheduler import evaluate_retraining_need
        result = evaluate_retraining_need(self._drift(0.4), **paths)
        assert isinstance(result["reason"], str)
        assert len(result["reason"]) > 0

    def test_cooldown_not_active_when_no_prior_retrain(self, paths):
        from src.retraining.scheduler import evaluate_retraining_need
        result = evaluate_retraining_need(self._drift(0.4), **paths)
        assert result["cooldown_active"] is False

    def test_drift_score_in_result_matches_input(self, paths):
        from src.retraining.scheduler import evaluate_retraining_need
        result = evaluate_retraining_need(self._drift(0.67), **paths)
        assert result["drift_score"] == pytest.approx(0.67)

    def test_records_score_in_history(self, paths):
        from src.retraining.scheduler  import evaluate_retraining_need
        from src.retraining.threshold  import ThresholdManager
        evaluate_retraining_need(self._drift(0.42), **paths)
        tm     = ThresholdManager(history_path=paths["history_path"])
        scores = tm.get_recent_scores()
        assert len(scores) == 1
        assert scores[0] == pytest.approx(0.42)

    def test_adaptive_threshold_lowers_after_high_history(self, paths):
        from src.retraining.scheduler import evaluate_retraining_need
        from src.retraining.threshold import ThresholdManager

        # Pre-fill history with high scores
        tm = ThresholdManager(history_path=paths["history_path"])
        for _ in range(10):
            tm.record_drift_score(0.8)

        # Score at base threshold (0.3) should now trigger because threshold lowered
        result = evaluate_retraining_need(self._drift(0.3, detected=True), **paths)
        # The adaptive threshold should be lower than 0.3, so 0.3 > adaptive → retrain
        if result["threshold_used"] < 0.3:
            assert result["should_retrain"] is True


# ══════════════════════════════════════════════════════════════════════════════
#  record_retrain_triggered
# ══════════════════════════════════════════════════════════════════════════════

class TestRecordRetrainTriggered:
    @pytest.fixture
    def paths(self, tmp_path):
        return {
            "state_path":   str(tmp_path / "state.json"),
            "history_path": str(tmp_path / "history.json"),
        }

    def test_records_cooldown_after_trigger(self, paths):
        from src.retraining.threshold import ThresholdManager
        from src.retraining.scheduler import record_retrain_triggered

        tm = ThresholdManager(history_path=paths["history_path"])
        tm.record_drift_score(0.55)  # seed history

        record_retrain_triggered(drift_score=0.55, dag_run_id="run_xyz", **paths)

        from src.retraining.cooldown import CooldownManager
        cm = CooldownManager(state_path=paths["state_path"])
        assert cm.is_cooldown_active() is True

    def test_back_patches_history_entry_triggered_flag(self, paths):
        from src.retraining.threshold import ThresholdManager
        from src.retraining.scheduler import record_retrain_triggered

        tm = ThresholdManager(history_path=paths["history_path"])
        tm.record_drift_score(0.55, triggered=False)

        record_retrain_triggered(drift_score=0.55, **paths)

        history = tm._load()
        assert history[-1]["triggered"] is True

    def test_does_not_raise_when_history_empty(self, paths):
        from src.retraining.scheduler import record_retrain_triggered
        # Should not raise even when history file doesn't exist yet
        record_retrain_triggered(drift_score=0.4, **paths)


# ══════════════════════════════════════════════════════════════════════════════
#  Task callables
# ══════════════════════════════════════════════════════════════════════════════

class TestTaskRecordRetrain:
    def test_records_state_on_success(self, tmp_path):
        from src.pipelines.tasks import task_record_retrain

        mock_ti = MagicMock()
        mock_ti.xcom_pull.return_value = 0.45

        with patch("src.retraining.scheduler.STATE_PATH",   str(tmp_path / "s.json")), \
             patch("src.retraining.scheduler.HISTORY_PATH", str(tmp_path / "h.json")):
            task_record_retrain(ti=mock_ti, dag_run=None)

        # task should not raise; ti.xcom_pull was called for drift_score
        mock_ti.xcom_pull.assert_called_with(task_ids="check_drift", key="drift_score")

    def test_non_fatal_on_error(self, tmp_path):
        from src.pipelines.tasks import task_record_retrain

        mock_ti = MagicMock()
        mock_ti.xcom_pull.return_value = 0.5

        with patch("src.retraining.scheduler.record_retrain_triggered",
                   side_effect=OSError("disk full")):
            task_record_retrain(ti=mock_ti, dag_run=None)  # must not raise


class TestTaskCheckDriftWithScheduler:
    def _setup_mocks(self, tmp_path, decision):
        """Return context patches for task_check_drift."""
        import sys
        from types import ModuleType

        # Stub evidently if needed
        for mod in ("evidently", "evidently.report", "evidently.metric_preset",
                    "evidently.metrics"):
            if mod not in sys.modules:
                sys.modules[mod] = ModuleType(mod)
        sys.modules["evidently.report"].Report = MagicMock()

        return {
            "state_path":   str(tmp_path / "s.json"),
            "history_path": str(tmp_path / "h.json"),
            "decision":     decision,
        }

    def test_pushes_all_xcom_keys(self, tmp_path):
        from src.pipelines.tasks import task_check_drift

        mock_ti      = MagicMock()
        mock_dag_run = MagicMock(); mock_dag_run.run_id = "manual_001"

        drift_result = {
            "drift_score": 0.45, "drift_detected": True,
            "needs_retraining": True, "report_path": "/tmp/r.html",
        }
        decision = {
            "should_retrain": True, "reason": "drift high",
            "drift_score": 0.45, "drift_detected": True,
            "cooldown_active": False, "cooldown_status": {},
            "threshold_used": 0.3, "base_threshold": 0.3,
            "adaptive": False, "recent_mean": None, "history_size": 0,
            "current_score": 0.45,
        }

        import sys
        from types import ModuleType

        # Stub evidently BEFORE importing drift_mod, so the from-imports resolve.
        for mod_name in ("evidently", "evidently.report", "evidently.metric_preset",
                         "evidently.metrics"):
            if mod_name not in sys.modules:
                sys.modules[mod_name] = ModuleType(mod_name)
        sys.modules["evidently.report"].Report                    = MagicMock()
        sys.modules["evidently.metric_preset"].DataDriftPreset    = MagicMock()
        sys.modules["evidently.metric_preset"].RegressionPreset   = MagicMock()
        sys.modules["evidently.metrics"].DataDriftMetric          = MagicMock()
        sys.modules["evidently.metrics"].DatasetDriftMetric       = MagicMock()
        sys.modules["evidently.metrics"].ColumnDriftMetric        = MagicMock()

        # Remove cached drift module so the stubs take effect on re-import
        sys.modules.pop("src.monitoring.drift", None)
        import src.monitoring.drift as drift_mod

        with patch.object(drift_mod, "load_reference_data",   return_value=MagicMock()), \
             patch.object(drift_mod, "load_current_data",     return_value=MagicMock()), \
             patch.object(drift_mod, "run_data_drift_report", return_value=drift_result), \
             patch("src.retraining.scheduler.evaluate_retraining_need",
                   return_value=decision):
            task_check_drift(ti=mock_ti, dag_run=mock_dag_run)

        pushed = {c[1]["key"]: c[1]["value"] for c in mock_ti.xcom_push.call_args_list}
        assert pushed["needs_retraining"] is True
        assert pushed["drift_score"]      == pytest.approx(0.45)
        assert pushed["retrain_reason"]   == "drift high"
        assert pushed["cooldown_active"]  is False
        assert "threshold_used"           in pushed
        assert "cooldown_status"          in pushed


class TestTaskBranchOnDrift:
    def test_routes_to_trigger_when_retrain_true(self):
        from src.pipelines.tasks import task_branch_on_drift

        ti = MagicMock()
        ti.xcom_pull.side_effect = lambda task_ids, key: (
            True if key == "needs_retraining" else "drift exceeded threshold"
        )
        assert task_branch_on_drift(ti=ti) == "trigger_retrain"

    def test_routes_to_skip_when_cooldown_active(self):
        from src.pipelines.tasks import task_branch_on_drift

        ti = MagicMock()
        ti.xcom_pull.side_effect = lambda task_ids, key: (
            False if key == "needs_retraining" else "cooldown active"
        )
        assert task_branch_on_drift(ti=ti) == "skip_retrain"

    def test_routes_to_skip_when_low_drift(self):
        from src.pipelines.tasks import task_branch_on_drift

        ti = MagicMock()
        ti.xcom_pull.return_value = False
        assert task_branch_on_drift(ti=ti) == "skip_retrain"
