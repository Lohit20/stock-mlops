"""
Tests for Step 8: Model Comparison
  - src/evaluation/evaluator.py
  - src/evaluation/report.py
  - src/evaluation/compare_models.py
"""

import os
import json
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


# ── Shared helpers ────────────────────────────────────────────────────────────

def _make_price_df(n_symbols=3, n_rows=200):
    dfs = []
    for i in range(n_symbols):
        sym    = f"SYM{i}"
        dates  = pd.date_range("2022-01-01", periods=n_rows, freq="B")
        prices = 100 + np.cumsum(np.random.randn(n_rows))
        dfs.append(pd.DataFrame({
            "timestamp":    dates, "symbol": sym,
            "close":        prices,
            "returns":      np.random.randn(n_rows) * 0.01,
            "volatility_7": np.abs(np.random.randn(n_rows)) * 0.02,
            "rsi_14":       np.random.uniform(30, 70, n_rows),
            "sma_7": prices, "sma_30": prices,
            "ema_7": prices, "ema_30": prices,
            "bb_width": np.random.uniform(0.01, 0.05, n_rows),
        }))
    return pd.concat(dfs, ignore_index=True)


def _make_per_symbol_df():
    rows = []
    for model in ["LSTM", "ARM", "TimesFM"]:
        for sym in ["AAPL", "MSFT"]:
            rows.append({
                "model_type":           model,
                "symbol":               sym,
                "rmse":                 np.random.uniform(5, 20),
                "mae":                  np.random.uniform(3, 15),
                "mse":                  np.random.uniform(25, 400),
                "mape":                 np.random.uniform(1, 5),
                "directional_accuracy": np.random.uniform(0.4, 0.7),
            })
    return pd.DataFrame(rows)


def _make_agg_df():
    return pd.DataFrame([
        {"model_type": "ARM",     "rmse": 8.0,  "mape": 2.5, "directional_accuracy": 0.55},
        {"model_type": "TimesFM", "rmse": 10.0, "mape": 3.0, "directional_accuracy": 0.52},
        {"model_type": "LSTM",    "rmse": 12.0, "mape": 3.5, "directional_accuracy": 0.50},
    ])


# ── evaluator.load_test_data ──────────────────────────────────────────────────

class TestLoadTestData:
    @patch("src.evaluation.evaluator.pd.read_csv")
    def test_returns_dict_keyed_by_symbol(self, mock_csv, tmp_path):
        mock_csv.return_value = _make_price_df(n_symbols=3, n_rows=200)
        from src.evaluation.evaluator import load_test_data
        result = load_test_data(str(tmp_path), n_steps_in=90, n_steps_out=30)
        assert isinstance(result, dict)
        assert len(result) == 3

    @patch("src.evaluation.evaluator.pd.read_csv")
    def test_context_and_y_true_shapes(self, mock_csv, tmp_path):
        mock_csv.return_value = _make_price_df(n_symbols=2, n_rows=200)
        from src.evaluation.evaluator import load_test_data
        result = load_test_data(str(tmp_path), n_steps_in=90, n_steps_out=30)
        for sym, data in result.items():
            assert data["context"].shape == (90,)
            assert data["y_true"].shape  == (30,)

    @patch("src.evaluation.evaluator.pd.read_csv")
    def test_skips_symbols_with_too_few_rows(self, mock_csv, tmp_path):
        df_long  = _make_price_df(n_symbols=2, n_rows=200)
        df_short = _make_price_df(n_symbols=1, n_rows=50)
        df_short["symbol"] = "SHORT"
        mock_csv.return_value = pd.concat([df_long, df_short], ignore_index=True)
        from src.evaluation.evaluator import load_test_data
        result = load_test_data(str(tmp_path), n_steps_in=90, n_steps_out=30)
        assert "SHORT" not in result


# ── evaluator._predict_timesfm ────────────────────────────────────────────────

class TestPredictTimesfm:
    def _make_test_data(self, n_symbols=2, n_rows=200):
        df  = _make_price_df(n_symbols=n_symbols, n_rows=n_rows)
        out = {}
        for sym, grp in df.groupby("symbol"):
            prices = grp["close"].values
            out[sym] = {
                "context": prices[:150],
                "y_true":  prices[150:180],
                "df":      grp,
            }
        return out

    def test_returns_predictions_for_each_symbol(self):
        from src.evaluation.evaluator import _predict_timesfm
        test_data = self._make_test_data(n_symbols=3)
        result    = _predict_timesfm(test_data, n_steps_out=30)
        assert len(result) == 3

    def test_predictions_have_correct_length(self):
        from src.evaluation.evaluator import _predict_timesfm
        test_data = self._make_test_data(n_symbols=2)
        result    = _predict_timesfm(test_data, n_steps_out=20)
        for sym, data in result.items():
            assert len(data["y_pred"]) == 20, f"Wrong length for {sym}"

    def test_y_true_y_pred_shapes_match(self):
        from src.evaluation.evaluator import _predict_timesfm
        test_data = self._make_test_data(n_symbols=2)
        result    = _predict_timesfm(test_data, n_steps_out=15)
        for sym, data in result.items():
            assert data["y_true"].shape == (30,)
            assert data["y_pred"].shape == (15,)


# ── evaluator._predict_arm ────────────────────────────────────────────────────

class TestPredictArm:
    def _make_test_data(self):
        df  = _make_price_df(n_symbols=2, n_rows=150)
        out = {}
        for sym, grp in df.groupby("symbol"):
            prices = grp["close"].values
            out[sym] = {"context": prices[:120], "y_true": prices[120:150], "df": grp}
        return out

    def test_calls_model_predict_per_symbol(self):
        from src.evaluation.evaluator import _predict_arm

        mock_model = MagicMock()
        mock_model.predict.return_value = pd.DataFrame({
            "symbol":          ["SYM0"] * 10,
            "forecast_day":    list(range(1, 11)),
            "predicted_close": [100.0] * 10,
        })
        test_data = self._make_test_data()
        result = _predict_arm(mock_model, test_data, n_steps_out=10)
        assert mock_model.predict.call_count == len(test_data)

    def test_returns_dict_with_y_pred(self):
        from src.evaluation.evaluator import _predict_arm

        mock_model = MagicMock()
        mock_model.predict.return_value = pd.DataFrame({
            "symbol":          ["SYM0"] * 10,
            "forecast_day":    list(range(1, 11)),
            "predicted_close": np.linspace(100, 110, 10),
        })
        test_data = self._make_test_data()
        result = _predict_arm(mock_model, test_data, n_steps_out=10)
        for sym, data in result.items():
            assert "y_true" in data
            assert "y_pred" in data


# ── evaluator._aggregate ─────────────────────────────────────────────────────

class TestAggregate:
    def test_returns_list_of_dicts(self):
        from src.evaluation.evaluator import _aggregate
        per_sym = {
            "AAPL": {"y_true": np.ones(10), "y_pred": np.ones(10) * 1.05},
            "MSFT": {"y_true": np.ones(10), "y_pred": np.ones(10) * 0.98},
        }
        rows = _aggregate("LSTM", per_sym)
        assert len(rows) == 2
        assert all("rmse" in r for r in rows)
        assert all("model_type" in r for r in rows)
        assert all(r["model_type"] == "LSTM" for r in rows)

    def test_rmse_is_non_negative(self):
        from src.evaluation.evaluator import _aggregate
        per_sym = {
            "SYM0": {"y_true": np.random.randn(30), "y_pred": np.random.randn(30)},
        }
        rows = _aggregate("ARM", per_sym)
        assert rows[0]["rmse"] >= 0


# ── report.generate_model_card ────────────────────────────────────────────────

class TestGenerateModelCard:
    def test_card_has_required_keys(self):
        from src.evaluation.report import generate_model_card
        best   = {"model_type": "ARM", "run_id": "abc123"}
        agg    = _make_agg_df()
        ps     = _make_per_symbol_df()
        card   = generate_model_card(best, agg, ps)
        for key in ("model_id", "model_type", "run_id", "timestamp",
                    "dataset", "evaluation", "intended_use", "limitations"):
            assert key in card, f"Missing key: {key}"

    def test_card_model_type_matches_best(self):
        from src.evaluation.report import generate_model_card
        best = {"model_type": "TimesFM", "run_id": "xyz"}
        card = generate_model_card(best, _make_agg_df(), _make_per_symbol_df())
        assert card["model_type"] == "TimesFM"

    def test_evaluation_section_has_metrics(self):
        from src.evaluation.report import generate_model_card
        best = {"model_type": "ARM", "run_id": "abc"}
        card = generate_model_card(best, _make_agg_df(), _make_per_symbol_df())
        ev   = card["evaluation"]
        assert "avg_rmse"         in ev
        assert "horizon_days"     in ev
        assert "context_days"     in ev
        assert "symbols_won"      in ev
        assert "symbols_total"    in ev

    def test_limitations_is_non_empty_list(self):
        from src.evaluation.report import generate_model_card
        best = {"model_type": "LSTM", "run_id": "run1"}
        card = generate_model_card(best, _make_agg_df(), _make_per_symbol_df())
        assert isinstance(card["limitations"], list)
        assert len(card["limitations"]) > 0

    def test_model_id_contains_model_type(self):
        from src.evaluation.report import generate_model_card
        best = {"model_type": "TFT", "run_id": "run2"}
        card = generate_model_card(best, _make_agg_df(), _make_per_symbol_df())
        assert "tft" in card["model_id"]


# ── report.comparison_bar_chart ───────────────────────────────────────────────

class TestComparisonBarChart:
    def test_calls_log_artifact(self):
        from src.evaluation.report import comparison_bar_chart
        agg = _make_agg_df()
        with patch("mlflow.log_artifact") as mock_art:
            comparison_bar_chart(agg)
        assert mock_art.call_count == 1

    def test_temp_file_cleaned_up(self):
        from src.evaluation.report import comparison_bar_chart
        import tempfile
        created = []
        orig = tempfile.NamedTemporaryFile

        def track(**kw):
            f = orig(**kw)
            created.append(f.name)
            return f

        with patch("mlflow.log_artifact"), \
             patch("tempfile.NamedTemporaryFile", side_effect=track):
            comparison_bar_chart(_make_agg_df())

        for p in created:
            assert not os.path.exists(p)


# ── report.win_count_chart ────────────────────────────────────────────────────

class TestWinCountChart:
    def test_calls_log_artifact(self):
        from src.evaluation.report import win_count_chart
        with patch("mlflow.log_artifact") as mock_art:
            win_count_chart(_make_per_symbol_df())
        assert mock_art.call_count == 1

    def test_empty_df_is_no_op(self):
        from src.evaluation.report import win_count_chart
        with patch("mlflow.log_artifact") as mock_art:
            win_count_chart(pd.DataFrame())
        assert mock_art.call_count == 0


# ── report.log_model_card ─────────────────────────────────────────────────────

class TestLogModelCard:
    def test_logs_valid_json(self):
        from src.evaluation.report import log_model_card
        card     = {"model_type": "ARM", "run_id": "abc"}
        captured = {}

        def fake_log_artifact(path, artifact_path=""):
            with open(path) as f:
                captured["data"] = json.load(f)

        with patch("mlflow.log_artifact", side_effect=fake_log_artifact):
            log_model_card(card)

        assert captured["data"]["model_type"] == "ARM"

    def test_temp_file_removed(self):
        from src.evaluation.report import log_model_card
        import tempfile
        created = []
        orig    = tempfile.NamedTemporaryFile

        def track(**kw):
            f = orig(**kw)
            created.append(f.name)
            return f

        with patch("mlflow.log_artifact"), \
             patch("tempfile.NamedTemporaryFile", side_effect=track):
            log_model_card({"model_type": "X"})

        for p in created:
            assert not os.path.exists(p)


# ── compare_models.rank_models ────────────────────────────────────────────────

class TestRankModels:
    def _make_historical_df(self):
        return pd.DataFrame([
            {"model_type": "LSTM", "run_id": "r1", "val_loss": 0.10,
             "test_rmse": 12.0, "test_mape": 3.5, "test_directional_accuracy": 0.50,
             "val_mae": 0.05, "start_time": 0, "git_sha": "abc", "data_hash": "xyz"},
            {"model_type": "ARM",  "run_id": "r2", "val_loss": 0.08,
             "test_rmse": 8.0,  "test_mape": 2.5, "test_directional_accuracy": 0.55,
             "val_mae": 0.04, "start_time": 0, "git_sha": "abc", "data_hash": "xyz"},
        ])

    def test_returns_dataframe_and_dict(self):
        from src.evaluation.compare_models import rank_models
        ranked, best = rank_models(self._make_historical_df(), pd.DataFrame())
        assert isinstance(ranked, pd.DataFrame)
        assert isinstance(best, dict)

    def test_winner_has_lowest_val_loss_when_no_fresh(self):
        from src.evaluation.compare_models import rank_models
        _, best = rank_models(self._make_historical_df(), pd.DataFrame())
        assert best["model_type"] == "ARM"   # val_loss=0.08

    def test_fresh_rmse_overrides_when_available(self):
        from src.evaluation.compare_models import rank_models
        # LSTM has better fresh_rmse even though higher val_loss
        fresh = pd.DataFrame([
            {"model_type": "LSTM", "rmse": 5.0, "mape": 1.5, "directional_accuracy": 0.65},
            {"model_type": "ARM",  "rmse": 9.0, "mape": 2.8, "directional_accuracy": 0.52},
        ])
        _, best = rank_models(self._make_historical_df(), fresh)
        assert best["model_type"] == "LSTM"

    def test_rank_column_starts_at_1(self):
        from src.evaluation.compare_models import rank_models
        ranked, _ = rank_models(self._make_historical_df(), pd.DataFrame())
        assert 1 in ranked["rank"].values
        assert ranked["rank"].min() == 1

    def test_all_models_get_a_rank(self):
        from src.evaluation.compare_models import rank_models
        ranked, _ = rank_models(self._make_historical_df(), pd.DataFrame())
        assert ranked["rank"].nunique() == len(ranked)


# ── compare_models.promote_best_model ─────────────────────────────────────────

class TestPromoteBestModel:
    def test_promotes_correct_version(self):
        from src.evaluation.compare_models import promote_best_model

        client   = MagicMock()
        version  = MagicMock()
        version.run_id        = "target-run-id"
        version.version       = "3"
        version.current_stage = "None"
        client.search_model_versions.return_value = [version]

        with patch("mlflow.tracking.MlflowClient", return_value=client):
            promote_best_model({
                "model_type": "ARM",
                "run_id":     "target-run-id",
            })

        client.transition_model_version_stage.assert_called_once()
        call_kwargs = client.transition_model_version_stage.call_args[1]
        assert call_kwargs["stage"] == "Production"
        assert call_kwargs["version"] == "3"

    def test_logs_warning_when_run_id_not_found(self):
        from src.evaluation.compare_models import promote_best_model

        client = MagicMock()
        client.search_model_versions.return_value = []

        with patch("mlflow.tracking.MlflowClient", return_value=client):
            promote_best_model({"model_type": "LSTM", "run_id": "unknown"})

        client.transition_model_version_stage.assert_not_called()

    def test_handles_mlflow_error_gracefully(self):
        from src.evaluation.compare_models import promote_best_model

        client = MagicMock()
        client.search_model_versions.side_effect = Exception("MLflow down")

        with patch("mlflow.tracking.MlflowClient", return_value=client):
            # Should not raise
            promote_best_model({"model_type": "ARM", "run_id": "r1"})
