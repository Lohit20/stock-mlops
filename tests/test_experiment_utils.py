"""
Tests for src/training/experiment_utils.py and src/training/hyperparam_search.py

All tests mock MLflow so they run offline without a tracking server.
"""

import os
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock, call


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_y(n=50, horizon=10):
    y_true = 100 + np.cumsum(np.random.randn(n * horizon)).reshape(n, horizon)
    y_pred = y_true + np.random.randn(n, horizon) * 2
    return y_true, y_pred


# ── get_git_sha ───────────────────────────────────────────────────────────────

class TestGetGitSha:
    def test_returns_string(self):
        from src.training.experiment_utils import get_git_sha
        result = get_git_sha()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_returns_unknown_when_git_fails(self):
        from src.training.experiment_utils import get_git_sha
        with patch("subprocess.check_output", side_effect=Exception("no git")):
            result = get_git_sha()
        assert result == "unknown"


# ── tag_current_run ───────────────────────────────────────────────────────────

class TestTagCurrentRun:
    def test_always_sets_git_sha_tag(self):
        from src.training.experiment_utils import tag_current_run

        with patch("mlflow.set_tags") as mock_tags, \
             patch("src.training.experiment_utils.get_git_sha", return_value="abc1234"):
            tag_current_run()

        tags = mock_tags.call_args[0][0]
        assert "git_sha" in tags
        assert tags["git_sha"] == "abc1234"

    def test_sets_data_hash_when_provided(self):
        from src.training.experiment_utils import tag_current_run

        with patch("mlflow.set_tags") as mock_tags, \
             patch("src.training.experiment_utils.get_git_sha", return_value="abc"):
            tag_current_run(data_hash="deadbeef", data_commit="cafebabe")

        tags = mock_tags.call_args[0][0]
        assert tags["data_hash"]   == "deadbeef"
        assert tags["data_commit"] == "cafebabe"

    def test_merges_extra_tags(self):
        from src.training.experiment_utils import tag_current_run

        with patch("mlflow.set_tags") as mock_tags, \
             patch("src.training.experiment_utils.get_git_sha", return_value="abc"):
            tag_current_run(extra_tags={"model_type": "LSTM", "trial": "3"})

        tags = mock_tags.call_args[0][0]
        assert tags["model_type"] == "LSTM"
        assert tags["trial"]      == "3"

    def test_omits_data_hash_when_none(self):
        from src.training.experiment_utils import tag_current_run

        with patch("mlflow.set_tags") as mock_tags, \
             patch("src.training.experiment_utils.get_git_sha", return_value="abc"):
            tag_current_run()

        tags = mock_tags.call_args[0][0]
        assert "data_hash"   not in tags
        assert "data_commit" not in tags


# ── make_model_signature ──────────────────────────────────────────────────────

class TestMakeModelSignature:
    def test_returns_signature_object(self):
        from src.training.experiment_utils import make_model_signature
        X = np.random.randn(4, 10, 3).astype("float32")
        y = np.random.randn(4, 5).astype("float32")
        sig = make_model_signature(X, y)
        assert sig is not None

    def test_returns_none_when_mlflow_not_available(self):
        from src.training.experiment_utils import make_model_signature
        with patch("mlflow.models.infer_signature", side_effect=Exception("no mlflow")):
            sig = make_model_signature(np.ones((2, 3)), np.ones((2, 1)))
        assert sig is None


# ── log_prediction_plot ───────────────────────────────────────────────────────

class TestLogPredictionPlot:
    def test_calls_log_artifact(self):
        from src.training.experiment_utils import log_prediction_plot
        y_true = np.random.randn(100)
        y_pred = np.random.randn(100)

        with patch("mlflow.log_artifact") as mock_artifact:
            log_prediction_plot(y_true, y_pred, title="Test Plot")

        assert mock_artifact.call_count == 1
        # artifact_path kwarg should be "plots"
        _, kwargs = mock_artifact.call_args
        assert kwargs.get("artifact_path") == "plots"

    def test_temp_file_is_cleaned_up(self):
        from src.training.experiment_utils import log_prediction_plot
        import tempfile
        created_files = []
        original_NamedTemporaryFile = tempfile.NamedTemporaryFile

        def tracking_tmpfile(**kwargs):
            f = original_NamedTemporaryFile(**kwargs)
            created_files.append(f.name)
            return f

        with patch("mlflow.log_artifact"), \
             patch("tempfile.NamedTemporaryFile", side_effect=tracking_tmpfile):
            try:
                log_prediction_plot(np.ones(10), np.ones(10))
            except Exception:
                pass  # might fail if matplotlib is not usable

        for path in created_files:
            assert not os.path.exists(path), f"Temp file not cleaned up: {path}"

    def test_2d_input_flattened(self):
        from src.training.experiment_utils import log_prediction_plot
        y_true = np.ones((5, 10))  # 2-D
        y_pred = np.ones((5, 10))

        with patch("mlflow.log_artifact"):
            # Should not raise
            log_prediction_plot(y_true, y_pred)

    def test_respects_max_samples(self):
        """Verify plot is created even when y arrays are very long."""
        from src.training.experiment_utils import log_prediction_plot
        y = np.random.randn(10000)

        with patch("mlflow.log_artifact"):
            log_prediction_plot(y, y, max_samples=50)


# ── log_metrics_table ─────────────────────────────────────────────────────────

class TestLogMetricsTable:
    def _sample_metrics(self):
        return {
            "AAPL": {"mse": 1.2, "mae": 0.8, "rmse": 1.1, "mape": 3.4},
            "MSFT": {"mse": 0.9, "mae": 0.6, "rmse": 0.95, "mape": 2.1},
            "GOOGL": {"mse": 1.5, "mae": 0.9, "rmse": 1.22, "mape": 4.0},
        }

    def test_returns_dataframe_with_correct_columns(self):
        from src.training.experiment_utils import log_metrics_table
        with patch("mlflow.log_artifact"):
            df = log_metrics_table(self._sample_metrics())
        assert isinstance(df, pd.DataFrame)
        assert "symbol" in df.columns
        assert "rmse"   in df.columns

    def test_one_row_per_symbol(self):
        from src.training.experiment_utils import log_metrics_table
        metrics = self._sample_metrics()
        with patch("mlflow.log_artifact"):
            df = log_metrics_table(metrics)
        assert len(df) == len(metrics)

    def test_calls_log_artifact_once(self):
        from src.training.experiment_utils import log_metrics_table
        with patch("mlflow.log_artifact") as mock_art:
            log_metrics_table(self._sample_metrics())
        assert mock_art.call_count == 1

    def test_symbols_sorted_alphabetically(self):
        from src.training.experiment_utils import log_metrics_table
        with patch("mlflow.log_artifact"):
            df = log_metrics_table(self._sample_metrics())
        symbols = df["symbol"].tolist()
        assert symbols == sorted(symbols)

    def test_temp_file_cleaned_up(self):
        from src.training.experiment_utils import log_metrics_table
        import tempfile
        created = []
        orig = tempfile.NamedTemporaryFile

        def tracker(**kw):
            f = orig(**kw)
            created.append(f.name)
            return f

        with patch("mlflow.log_artifact"), \
             patch("tempfile.NamedTemporaryFile", side_effect=tracker):
            log_metrics_table(self._sample_metrics())

        for p in created:
            assert not os.path.exists(p)


# ── get_best_run ──────────────────────────────────────────────────────────────

class TestGetBestRun:
    def _mock_client(self, runs):
        client     = MagicMock()
        experiment = MagicMock()
        experiment.experiment_id = "1"
        client.get_experiment_by_name.return_value = experiment
        client.search_runs.return_value = runs
        return client

    def _make_run(self, run_id, model_type, val_loss):
        run = MagicMock()
        run.info.run_id               = run_id
        run.data.params               = {"model_type": model_type}
        run.data.metrics              = {"val_loss": val_loss}
        run.data.tags                 = {"git_sha": "abc1234"}
        return run

    def test_returns_best_run_dict(self):
        from src.training.experiment_utils import get_best_run

        runs = [
            self._make_run("run1", "LSTM", 0.05),
            self._make_run("run2", "ARM",  0.12),
        ]
        with patch("mlflow.tracking.MlflowClient",
                   return_value=self._mock_client(runs)), \
             patch("mlflow.set_tracking_uri"):
            result = get_best_run("stock_price_forecasting")

        assert result is not None
        assert result["run_id"] == "run1"
        assert result["val_loss"] == pytest.approx(0.05)

    def test_returns_none_when_experiment_missing(self):
        from src.training.experiment_utils import get_best_run

        client = MagicMock()
        client.get_experiment_by_name.return_value = None
        with patch("mlflow.tracking.MlflowClient", return_value=client), \
             patch("mlflow.set_tracking_uri"):
            result = get_best_run("nonexistent")

        assert result is None

    def test_returns_none_when_no_finished_runs(self):
        from src.training.experiment_utils import get_best_run

        with patch("mlflow.tracking.MlflowClient",
                   return_value=self._mock_client([])), \
             patch("mlflow.set_tracking_uri"):
            result = get_best_run("stock_price_forecasting")

        assert result is None

    def test_result_contains_required_keys(self):
        from src.training.experiment_utils import get_best_run

        runs = [self._make_run("run1", "LSTM", 0.05)]
        with patch("mlflow.tracking.MlflowClient",
                   return_value=self._mock_client(runs)), \
             patch("mlflow.set_tracking_uri"):
            result = get_best_run("stock_price_forecasting")

        for key in ("run_id", "model_type", "val_loss", "params", "metrics", "tags"):
            assert key in result


# ── get_experiment_summary ────────────────────────────────────────────────────

class TestGetExperimentSummary:
    def test_returns_empty_dataframe_when_experiment_missing(self):
        from src.training.experiment_utils import get_experiment_summary

        client = MagicMock()
        client.get_experiment_by_name.return_value = None
        with patch("mlflow.tracking.MlflowClient", return_value=client), \
             patch("mlflow.set_tracking_uri"):
            df = get_experiment_summary("nonexistent")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_returns_one_row_per_run(self):
        from src.training.experiment_utils import get_experiment_summary

        client     = MagicMock()
        experiment = MagicMock()
        experiment.experiment_id = "1"
        client.get_experiment_by_name.return_value = experiment

        fake_runs = []
        for i, mtype in enumerate(["LSTM", "ARM", "TimesFM"]):
            r = MagicMock()
            r.info.run_id           = f"run{i}"
            r.data.params           = {"model_type": mtype}
            r.data.metrics          = {"val_loss": 0.1 * (i + 1)}
            r.data.tags             = {"git_sha": "abc"}
            fake_runs.append(r)
        client.search_runs.return_value = fake_runs

        with patch("mlflow.tracking.MlflowClient", return_value=client), \
             patch("mlflow.set_tracking_uri"):
            df = get_experiment_summary("stock_price_forecasting")

        assert len(df) == 3
        assert "model_type" in df.columns


# ── Hyperparameter search ─────────────────────────────────────────────────────

class TestHyperparamSearch:
    """Unit tests for HP search pipeline — mock out actual training."""

    def _patch_train(self, val_loss=0.05):
        """Return a mock train() that returns a minimal result dict."""
        mock_train = MagicMock(return_value={
            "run_id":       "mock-run",
            "val_loss":     val_loss,
            "test_metrics": {"mse": 0.1, "mae": 0.05, "rmse": 0.1,
                             "mape": 2.0, "directional_accuracy": 0.55},
            "model_name":   "mock_model",
        })
        return mock_train

    def test_search_arm_returns_best_config(self):
        # Pre-import so the module is in sys.modules before patching
        import src.training.train_arm  # noqa: F401
        from src.training.hyperparam_search import search_arm

        import sys
        fake_mlflow  = MagicMock()
        parent_ctx   = MagicMock()
        parent_ctx.__enter__ = lambda s: s
        parent_ctx.__exit__  = MagicMock(return_value=False)
        parent_ctx.info.run_id = "parent-id"
        fake_mlflow.start_run.return_value = parent_ctx
        fake_mlflow.active_run.return_value = parent_ctx

        tiny_space = [{"p": 2, "d": 1, "q": 0}, {"p": 5, "d": 1, "q": 0}]

        with patch.dict(sys.modules, {"mlflow": fake_mlflow}), \
             patch("src.training.hyperparam_search.tag_current_run"), \
             patch("src.training.train_arm.train", self._patch_train(0.08)):

            result = search_arm(search_space=tiny_space)

        assert "val_loss" in result
        assert "config"   in result
        assert result["config"] in tiny_space

    def test_search_arm_raises_when_all_trials_fail(self):
        # Pre-import so the module is in sys.modules before patching
        import src.training.train_arm  # noqa: F401
        from src.training.hyperparam_search import search_arm

        import sys
        fake_mlflow = MagicMock()
        ctx         = MagicMock()
        ctx.__enter__ = lambda s: s
        ctx.__exit__  = MagicMock(return_value=False)
        ctx.info.run_id = "parent-id"
        fake_mlflow.start_run.return_value = ctx

        tiny_space = [{"p": 2, "d": 1, "q": 0}]

        with patch.dict(sys.modules, {"mlflow": fake_mlflow}), \
             patch("src.training.hyperparam_search.tag_current_run"), \
             patch("src.training.train_arm.train",
                   side_effect=RuntimeError("simulated failure")):
            with pytest.raises(RuntimeError, match="All ARM HP search trials failed"):
                search_arm(search_space=tiny_space)

    def test_search_space_constants_have_required_keys(self):
        from src.training.hyperparam_search import LSTM_SEARCH_SPACE, ARM_SEARCH_SPACE

        for cfg in LSTM_SEARCH_SPACE:
            assert "n_steps_in"  in cfg
            assert "batch_size"  in cfg
            assert "epochs"      in cfg
            assert "feature_cols" in cfg

        for cfg in ARM_SEARCH_SPACE:
            assert "p" in cfg
            assert "d" in cfg
            assert "q" in cfg
