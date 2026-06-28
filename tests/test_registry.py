"""
Tests for Step 9: Model Registry
  src/registry/model_registry.py
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock, call


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_version(version="1", run_id="run-abc", stage="None"):
    v = MagicMock()
    v.version           = version
    v.run_id            = run_id
    v.current_stage     = stage
    v.creation_timestamp = int(version) * 1000  # higher version → more recent
    return v


def _client_with_versions(versions: list):
    """Return a mock MlflowClient whose search_model_versions returns `versions`."""
    client = MagicMock()
    client.search_model_versions.return_value = versions
    client.get_run.return_value = MagicMock(
        data=MagicMock(metrics={
            "val_loss": 0.05,
            "test_rmse": 8.0,
            "test_mape": 2.5,
            "test_directional_accuracy": 0.60,
        })
    )
    return client


# ── _model_name ───────────────────────────────────────────────────────────────

class TestModelName:
    def test_known_types_return_correct_name(self):
        from src.registry.model_registry import _model_name
        assert _model_name("LSTM")    == "stock_price_forecaster_lstm"
        assert _model_name("ARM")     == "stock_price_forecaster_arm"
        assert _model_name("TFT")     == "stock_price_forecaster_tft"
        assert _model_name("TimesFM") == "stock_price_forecaster_timesfm"

    def test_unknown_type_raises(self):
        from src.registry.model_registry import _model_name
        with pytest.raises(ValueError, match="Unknown model_type"):
            _model_name("XGBoost")


# ── get_registry_summary ──────────────────────────────────────────────────────

class TestGetRegistrySummary:
    def test_returns_dataframe(self):
        from src.registry.model_registry import get_registry_summary
        v1 = _make_version("1", "run1", "Production")
        v2 = _make_version("2", "run2", "Staging")
        client = _client_with_versions([v1, v2])

        with patch("src.registry.model_registry._client", return_value=client):
            df = get_registry_summary()

        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self):
        from src.registry.model_registry import get_registry_summary
        client = _client_with_versions([_make_version()])
        with patch("src.registry.model_registry._client", return_value=client):
            df = get_registry_summary()

        for col in ("model_name", "model_type", "version", "stage", "run_id"):
            assert col in df.columns

    def test_empty_when_no_versions(self):
        from src.registry.model_registry import get_registry_summary
        client = _client_with_versions([])
        with patch("src.registry.model_registry._client", return_value=client):
            df = get_registry_summary()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_handles_client_error_gracefully(self):
        from src.registry.model_registry import get_registry_summary
        client = MagicMock()
        client.search_model_versions.side_effect = Exception("MLflow unavailable")
        with patch("src.registry.model_registry._client", return_value=client):
            df = get_registry_summary()
        assert isinstance(df, pd.DataFrame)


# ── promote_to_staging ────────────────────────────────────────────────────────

class TestPromoteToStaging:
    def test_transitions_correct_version(self):
        from src.registry.model_registry import promote_to_staging
        v = _make_version("3", "target-run", "None")
        client = _client_with_versions([v])

        with patch("src.registry.model_registry._client", return_value=client):
            version = promote_to_staging("ARM", "target-run")

        assert version == "3"
        client.transition_model_version_stage.assert_called_once_with(
            name="stock_price_forecaster_arm",
            version="3",
            stage="Staging",
            archive_existing_versions=False,
        )

    def test_returns_empty_string_when_run_id_not_found(self):
        from src.registry.model_registry import promote_to_staging
        client = _client_with_versions([])
        with patch("src.registry.model_registry._client", return_value=client):
            result = promote_to_staging("LSTM", "nonexistent-run")
        assert result == ""

    def test_does_not_transition_wrong_run(self):
        from src.registry.model_registry import promote_to_staging
        other = _make_version("1", "other-run", "None")
        client = _client_with_versions([other])
        with patch("src.registry.model_registry._client", return_value=client):
            promote_to_staging("ARM", "my-run")
        client.transition_model_version_stage.assert_not_called()


# ── validate_version ──────────────────────────────────────────────────────────

class TestValidateVersion:
    def test_passes_when_metrics_meet_thresholds(self):
        from src.registry.model_registry import validate_version
        v = _make_version("1", "run1", "Staging")
        client = _client_with_versions([v])
        client.get_run.return_value = MagicMock(
            data=MagicMock(metrics={
                "test_directional_accuracy": 0.60,  # ≥ 0.45 → pass
                "test_mape": 20.0,                   # ≤ 50 → pass
            })
        )
        with patch("src.registry.model_registry._client", return_value=client):
            passed, report = validate_version("ARM", "1")
        assert passed is True
        assert all(v["passed"] for v in report.values() if v["passed"] is not None)

    def test_fails_when_directional_accuracy_too_low(self):
        from src.registry.model_registry import validate_version
        v = _make_version("1", "run1", "Staging")
        client = _client_with_versions([v])
        client.get_run.return_value = MagicMock(
            data=MagicMock(metrics={
                "test_directional_accuracy": 0.30,  # < 0.45 → fail
                "test_mape": 20.0,
            })
        )
        with patch("src.registry.model_registry._client", return_value=client):
            passed, report = validate_version("ARM", "1")
        assert passed is False
        assert report["test_directional_accuracy"]["passed"] is False

    def test_missing_metric_is_skipped_not_failed(self):
        from src.registry.model_registry import validate_version
        v = _make_version("1", "run1", "Staging")
        client = _client_with_versions([v])
        client.get_run.return_value = MagicMock(
            data=MagicMock(metrics={"test_mape": 10.0})  # no directional_accuracy
        )
        with patch("src.registry.model_registry._client", return_value=client):
            passed, report = validate_version("ARM", "1",
                min_thresholds={"test_directional_accuracy": 0.45, "test_mape": 50.0})
        # missing metric = None, should not cause passed=False
        assert report["test_directional_accuracy"]["passed"] is None
        assert report["test_mape"]["passed"] is True

    def test_returns_false_for_unknown_version(self):
        from src.registry.model_registry import validate_version
        client = _client_with_versions([])
        with patch("src.registry.model_registry._client", return_value=client):
            passed, report = validate_version("ARM", "99")
        assert passed is False

    def test_custom_thresholds_override_defaults(self):
        from src.registry.model_registry import validate_version
        v = _make_version("1", "run1", "Staging")
        client = _client_with_versions([v])
        client.get_run.return_value = MagicMock(
            data=MagicMock(metrics={"test_mape": 30.0})
        )
        with patch("src.registry.model_registry._client", return_value=client):
            # strict threshold → should fail
            passed, _ = validate_version("ARM", "1",
                min_thresholds={"test_mape": 10.0})
        assert passed is False


# ── promote_to_production ─────────────────────────────────────────────────────

class TestPromoteToProduction:
    def test_promotes_when_validation_passes(self):
        from src.registry.model_registry import promote_to_production
        v = _make_version("2", "run2", "Staging")
        client = _client_with_versions([v])

        with patch("src.registry.model_registry._client", return_value=client), \
             patch("src.registry.model_registry.validate_version",
                   return_value=(True, {})):
            result = promote_to_production("ARM", "2")

        assert result is True
        client.transition_model_version_stage.assert_called_once_with(
            name="stock_price_forecaster_arm",
            version="2",
            stage="Production",
            archive_existing_versions=True,
        )

    def test_blocks_when_validation_fails(self):
        from src.registry.model_registry import promote_to_production
        client = _client_with_versions([_make_version("1", "r1", "Staging")])

        with patch("src.registry.model_registry._client", return_value=client), \
             patch("src.registry.model_registry.validate_version",
                   return_value=(False, {"test_mape": {"passed": False}})):
            result = promote_to_production("ARM", "1")

        assert result is False
        client.transition_model_version_stage.assert_not_called()

    def test_skip_validation_always_promotes(self):
        from src.registry.model_registry import promote_to_production
        v = _make_version("3", "run3", "Staging")
        client = _client_with_versions([v])

        with patch("src.registry.model_registry._client", return_value=client):
            result = promote_to_production("LSTM", "3", skip_validation=True)

        assert result is True
        client.transition_model_version_stage.assert_called_once()


# ── rollback ──────────────────────────────────────────────────────────────────

class TestRollback:
    def test_restores_most_recent_archived_version(self):
        from src.registry.model_registry import rollback
        v_prod = _make_version("3", "run3", "Production")
        v_arc2 = _make_version("2", "run2", "Archived")
        v_arc1 = _make_version("1", "run1", "Archived")
        client = _client_with_versions([v_prod, v_arc2, v_arc1])

        with patch("src.registry.model_registry._client", return_value=client):
            version = rollback("ARM")

        assert version == "2"  # most recent archived
        calls = client.transition_model_version_stage.call_args_list
        # First call: move Production → Archived
        # Second call: move v2 → Production
        assert calls[0][1]["stage"] == "Archived"
        assert calls[1][1]["stage"] == "Production"
        assert calls[1][1]["version"] == "2"

    def test_returns_empty_string_when_no_archived(self):
        from src.registry.model_registry import rollback
        v_prod = _make_version("1", "run1", "Production")
        client = _client_with_versions([v_prod])

        with patch("src.registry.model_registry._client", return_value=client):
            version = rollback("LSTM")

        assert version == ""
        client.transition_model_version_stage.assert_not_called()

    def test_archives_current_production_before_restore(self):
        from src.registry.model_registry import rollback
        v_prod = _make_version("5", "run5", "Production")
        v_arc  = _make_version("4", "run4", "Archived")
        client = _client_with_versions([v_prod, v_arc])

        with patch("src.registry.model_registry._client", return_value=client):
            rollback("TFT")

        # Production (v5) archived first
        first_call = client.transition_model_version_stage.call_args_list[0]
        assert first_call[1]["version"] == "5"
        assert first_call[1]["stage"]   == "Archived"


# ── compare_versions ──────────────────────────────────────────────────────────

class TestCompareVersions:
    def _client_for_compare(self, metrics_a, metrics_b):
        v1 = _make_version("1", "run1", "Production")
        v2 = _make_version("2", "run2", "Archived")
        client = MagicMock()
        client.search_model_versions.return_value = [v1, v2]

        def get_run(run_id):
            metrics = metrics_a if run_id == "run1" else metrics_b
            return MagicMock(data=MagicMock(metrics=metrics))

        client.get_run.side_effect = get_run
        return client

    def test_returns_required_keys(self):
        from src.registry.model_registry import compare_versions
        client = self._client_for_compare(
            {"test_rmse": 8.0}, {"test_rmse": 10.0}
        )
        with patch("src.registry.model_registry._client", return_value=client):
            result = compare_versions("ARM", "1", "2")
        for key in ("model_name", "version_a", "version_b", "winner", "winner_version"):
            assert key in result

    def test_winner_has_lower_rmse(self):
        from src.registry.model_registry import compare_versions
        client = self._client_for_compare(
            {"test_rmse": 7.0},   # version_a
            {"test_rmse": 12.0},  # version_b
        )
        with patch("src.registry.model_registry._client", return_value=client):
            result = compare_versions("ARM", "1", "2")
        assert result["winner"] == "a"
        assert result["winner_version"] == "1"

    def test_tie_when_equal_metrics(self):
        from src.registry.model_registry import compare_versions
        client = self._client_for_compare(
            {"test_rmse": 9.0},
            {"test_rmse": 9.0},
        )
        with patch("src.registry.model_registry._client", return_value=client):
            result = compare_versions("LSTM", "1", "2")
        assert result["winner"] == "tie"

    def test_raises_for_unknown_version(self):
        from src.registry.model_registry import compare_versions
        client = MagicMock()
        client.search_model_versions.return_value = []
        with patch("src.registry.model_registry._client", return_value=client):
            with pytest.raises(ValueError):
                compare_versions("ARM", "99", "100")


# ── cleanup_old_versions ──────────────────────────────────────────────────────

class TestCleanupOldVersions:
    def test_deletes_beyond_keep_n(self):
        from src.registry.model_registry import cleanup_old_versions
        archived = [_make_version(str(i), f"run{i}", "Archived") for i in range(1, 8)]
        client = _client_with_versions(archived)

        with patch("src.registry.model_registry._client", return_value=client):
            deleted = cleanup_old_versions("ARM", keep_n=5)

        assert deleted == 2
        assert client.delete_model_version.call_count == 2

    def test_no_delete_when_within_keep_n(self):
        from src.registry.model_registry import cleanup_old_versions
        archived = [_make_version(str(i), f"run{i}", "Archived") for i in range(1, 4)]
        client = _client_with_versions(archived)

        with patch("src.registry.model_registry._client", return_value=client):
            deleted = cleanup_old_versions("LSTM", keep_n=5)

        assert deleted == 0
        client.delete_model_version.assert_not_called()

    def test_never_deletes_non_archived_versions(self):
        from src.registry.model_registry import cleanup_old_versions
        versions = [
            _make_version("1", "r1", "Production"),
            _make_version("2", "r2", "Staging"),
            _make_version("3", "r3", "Archived"),
            _make_version("4", "r4", "Archived"),
            _make_version("5", "r5", "Archived"),
            _make_version("6", "r6", "Archived"),
        ]
        client = _client_with_versions(versions)

        with patch("src.registry.model_registry._client", return_value=client):
            deleted = cleanup_old_versions("ARM", keep_n=3)

        assert deleted == 1   # only 1 of 4 archived is beyond keep_n=3
        # Only version "3" should be deleted (oldest)
        deleted_version = client.delete_model_version.call_args[1]["version"]
        assert deleted_version == "3"


# ── get_latest_production ──────────────────────────────────────────────────────

class TestGetLatestProduction:
    def test_returns_dict_for_existing_production(self):
        from src.registry.model_registry import get_latest_production
        v = _make_version("4", "prod-run", "Production")
        client = _client_with_versions([v])

        with patch("src.registry.model_registry._client", return_value=client):
            info = get_latest_production("ARM")

        assert info is not None
        assert info["version"] == "4"
        assert info["stage"]   == "Production"
        assert "model_uri" in info
        assert info["model_uri"] == "models:/stock_price_forecaster_arm/Production"

    def test_returns_none_when_no_production(self):
        from src.registry.model_registry import get_latest_production
        v = _make_version("1", "r1", "Staging")
        client = _client_with_versions([v])

        with patch("src.registry.model_registry._client", return_value=client):
            info = get_latest_production("LSTM")

        assert info is None

    def test_includes_metrics(self):
        from src.registry.model_registry import get_latest_production
        v = _make_version("2", "run2", "Production")
        client = _client_with_versions([v])
        client.get_run.return_value = MagicMock(
            data=MagicMock(metrics={"test_rmse": 7.5, "val_loss": 0.03})
        )
        with patch("src.registry.model_registry._client", return_value=client):
            info = get_latest_production("TFT")

        assert info["metrics"]["test_rmse"] == 7.5


# ── register_best_model ───────────────────────────────────────────────────────

class TestRegisterBestModel:
    def _best_info(self):
        return {"model_type": "ARM", "run_id": "target-run"}

    def test_returns_result_dict_with_required_keys(self):
        from src.registry.model_registry import register_best_model

        with patch("src.registry.model_registry.promote_to_staging",   return_value="3"), \
             patch("src.registry.model_registry.promote_to_production", return_value=True), \
             patch("src.registry.model_registry._client",
                   return_value=_client_with_versions([])), \
             patch("src.registry.model_registry.set_champion_challenger"), \
             patch("src.registry.model_registry.cleanup_old_versions",  return_value=0):
            result = register_best_model(self._best_info())

        for key in ("model_type", "version", "promoted", "stage"):
            assert key in result

    def test_promoted_true_when_all_steps_succeed(self):
        from src.registry.model_registry import register_best_model

        with patch("src.registry.model_registry.promote_to_staging",   return_value="3"), \
             patch("src.registry.model_registry.promote_to_production", return_value=True), \
             patch("src.registry.model_registry._client",
                   return_value=_client_with_versions([])), \
             patch("src.registry.model_registry.set_champion_challenger"), \
             patch("src.registry.model_registry.cleanup_old_versions",  return_value=0):
            result = register_best_model(self._best_info())

        assert result["promoted"] is True
        assert result["stage"] == "Production"

    def test_promoted_false_when_validation_fails(self):
        from src.registry.model_registry import register_best_model

        with patch("src.registry.model_registry.promote_to_staging",   return_value="2"), \
             patch("src.registry.model_registry.promote_to_production", return_value=False):
            result = register_best_model(self._best_info())

        assert result["promoted"] is False
        assert result["stage"] == "Staging"

    def test_raises_on_missing_model_type(self):
        from src.registry.model_registry import register_best_model
        with pytest.raises(ValueError, match="model_type"):
            register_best_model({"run_id": "abc"})

    def test_raises_on_missing_run_id(self):
        from src.registry.model_registry import register_best_model
        with pytest.raises(ValueError, match="run_id"):
            register_best_model({"model_type": "ARM"})

    def test_staging_failure_returns_early(self):
        from src.registry.model_registry import register_best_model

        with patch("src.registry.model_registry.promote_to_staging", return_value=""):
            result = register_best_model(self._best_info())

        assert result["promoted"] is False
        assert result["version"] == ""

    def test_calls_cleanup_after_promotion(self):
        from src.registry.model_registry import register_best_model

        with patch("src.registry.model_registry.promote_to_staging",   return_value="5"), \
             patch("src.registry.model_registry.promote_to_production", return_value=True), \
             patch("src.registry.model_registry._client",
                   return_value=_client_with_versions([])), \
             patch("src.registry.model_registry.set_champion_challenger"), \
             patch("src.registry.model_registry.cleanup_old_versions",
                   return_value=0) as mock_cleanup:
            register_best_model(self._best_info())

        mock_cleanup.assert_called_once()


# ── set_champion_challenger ───────────────────────────────────────────────────

class TestSetChampionChallenger:
    def test_sets_both_aliases(self):
        from src.registry.model_registry import set_champion_challenger
        client = MagicMock()

        with patch("src.registry.model_registry._client", return_value=client):
            set_champion_challenger("ARM", "5", "4")

        calls = [c[0] for c in client.set_registered_model_alias.call_args_list]
        aliases = {c[1]: c[2] for c in calls}
        assert aliases.get("champion")   == "5"
        assert aliases.get("challenger") == "4"

    def test_gracefully_handles_mlflow_1x_missing_method(self):
        from src.registry.model_registry import set_champion_challenger
        client = MagicMock(spec=[])  # no attributes → AttributeError on any access

        with patch("src.registry.model_registry._client", return_value=client):
            # Should not raise
            set_champion_challenger("LSTM", "2", "1")
