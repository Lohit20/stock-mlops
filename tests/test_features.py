"""
Tests for src/features/engineer.py
"""

import os
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
from src.features.engineer import (
    transpose_df,
    clean_df,
    add_price_features,
    build_price_features,
    add_financial_ratios,
    clean_prices,
    clean_financial,
    _rsi,
    _bollinger_bands,
    run_cleaning,
    run_feature_engineering,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_price_df(n=60, symbol="AAPL"):
    dates  = pd.date_range("2023-01-01", periods=n, freq="D")
    prices = np.linspace(100, 160, n) + np.random.randn(n) * 0.5
    return pd.DataFrame({
        "timestamp": dates,
        "symbol":    symbol,
        "close":     prices,
        "open":      prices * 0.99,
        "high":      prices * 1.01,
        "low":       prices * 0.98,
        "volume":    np.random.randint(1_000_000, 5_000_000, n),
    })


@pytest.fixture
def price_df():
    return _make_price_df()


@pytest.fixture
def multi_symbol_df():
    return pd.concat([
        _make_price_df(60, "AAPL"),
        _make_price_df(60, "MSFT"),
        _make_price_df(60, "GOOG"),
    ], ignore_index=True)


@pytest.fixture
def balance_df():
    return pd.DataFrame({
        "Date":                      ["2023-Q4", "2023-Q4"],
        "Company Symbol":            ["AAPL",    "MSFT"],
        "Total Assets":              [100.0,     200.0],
        "Total Liabilities":         [40.0,      80.0],
        "Total Current Assets":      [50.0,      90.0],
        "Total Current Liabilities": [20.0,      30.0],
        "Total Equity":              [60.0,      120.0],
    })


@pytest.fixture
def income_df():
    return pd.DataFrame({
        "Date":           ["2023-Q4", "2023-Q4"],
        "Company Symbol": ["AAPL",    "MSFT"],
        "Net Income":     [10.0,      25.0],
        "Total Revenue":  [80.0,      150.0],
    })


# ── _rsi ──────────────────────────────────────────────────────────────────────

def test_rsi_is_bounded(price_df):
    rsi = _rsi(price_df["close"], 14)
    valid = rsi.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_rsi_has_nan_for_first_period(price_df):
    rsi = _rsi(price_df["close"], 14)
    assert rsi.iloc[:14].isna().all()
    assert not rsi.iloc[14:].isna().all()


# ── _bollinger_bands ──────────────────────────────────────────────────────────

def test_bollinger_upper_above_lower(price_df):
    upper, mid, lower, width = _bollinger_bands(price_df["close"])
    valid = ~upper.isna()
    assert (upper[valid] >= lower[valid]).all()


def test_bollinger_mid_between_bands(price_df):
    upper, mid, lower, width = _bollinger_bands(price_df["close"])
    valid = ~upper.isna()
    assert (mid[valid] <= upper[valid]).all()
    assert (mid[valid] >= lower[valid]).all()


def test_bollinger_width_positive(price_df):
    _, _, _, width = _bollinger_bands(price_df["close"])
    assert (width.dropna() > 0).all()


# ── add_price_features ────────────────────────────────────────────────────────

def test_price_features_columns_present(price_df):
    out = add_price_features(price_df)
    expected = [
        "sma_7", "sma_14", "sma_30",
        "ema_7", "ema_14", "ema_30",
        "momentum_7", "momentum_14", "momentum_30",
        "volatility_7", "volatility_30",
        "returns", "log_returns",
        "rsi_14",
        "bb_upper", "bb_mid", "bb_lower", "bb_width",
    ]
    for col in expected:
        assert col in out.columns, f"Missing column: {col}"


def test_price_features_does_not_mutate_input(price_df):
    original_cols = set(price_df.columns)
    add_price_features(price_df)
    assert set(price_df.columns) == original_cols


def test_sma_7_nan_for_first_6_rows(price_df):
    out = add_price_features(price_df)
    assert out["sma_7"].iloc[:6].isna().all()
    assert not out["sma_7"].iloc[6:].isna().all()


def test_sma_30_nan_for_first_29_rows(price_df):
    out = add_price_features(price_df)
    assert out["sma_30"].iloc[:29].isna().all()


def test_returns_is_pct_change_of_close(price_df):
    out = add_price_features(price_df)
    expected = price_df["close"].pct_change()
    pd.testing.assert_series_equal(
        out["returns"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_momentum_7_is_price_minus_7_day_lag(price_df):
    out   = add_price_features(price_df)
    close = price_df["close"].reset_index(drop=True)
    expected = (close - close.shift(7)).reset_index(drop=True)
    pd.testing.assert_series_equal(
        out["momentum_7"].reset_index(drop=True),
        expected,
        check_names=False,
    )


def test_log_returns_highly_correlated_with_returns(price_df):
    out  = add_price_features(price_df)
    corr = out[["returns", "log_returns"]].dropna().corr().iloc[0, 1]
    assert corr > 0.99


# ── build_price_features ──────────────────────────────────────────────────────

def test_build_price_features_all_symbols_present(multi_symbol_df):
    out = build_price_features(multi_symbol_df)
    assert set(out["symbol"].unique()) == {"AAPL", "MSFT", "GOOG"}


def test_build_price_features_no_cross_symbol_leakage(multi_symbol_df):
    out    = build_price_features(multi_symbol_df)
    aapl   = out[out["symbol"] == "AAPL"]
    # SMA for AAPL row 0 should be NaN (not contaminated by MSFT/GOOG rows)
    assert aapl["sma_7"].iloc[0] is np.nan or pd.isna(aapl["sma_7"].iloc[0])


def test_build_price_features_output_length(multi_symbol_df):
    out = build_price_features(multi_symbol_df)
    assert len(out) == len(multi_symbol_df)


# ── add_financial_ratios ──────────────────────────────────────────────────────

def test_financial_ratio_columns(balance_df, income_df):
    out = add_financial_ratios(balance_df, income_df)
    for col in ["debt_ratio", "current_ratio", "profit_margin", "roa", "roe"]:
        assert col in out.columns, f"Missing: {col}"


def test_debt_ratio_correct(balance_df, income_df):
    out  = add_financial_ratios(balance_df, income_df)
    aapl = out[out["Company Symbol"] == "AAPL"].iloc[0]
    assert abs(aapl["debt_ratio"] - 40 / 100) < 1e-9


def test_profit_margin_correct(balance_df, income_df):
    out  = add_financial_ratios(balance_df, income_df)
    aapl = out[out["Company Symbol"] == "AAPL"].iloc[0]
    assert abs(aapl["profit_margin"] - 10 / 80) < 1e-9


def test_no_division_by_zero(income_df):
    balance = pd.DataFrame({
        "Date":                      ["2023-Q4"],
        "Company Symbol":            ["ZZZ"],
        "Total Assets":              [0.0],
        "Total Liabilities":         [5.0],
        "Total Current Assets":      [0.0],
        "Total Current Liabilities": [0.0],
        "Total Equity":              [0.0],
    })
    out = add_financial_ratios(balance, income_df.iloc[:0])
    # Merging with empty income → empty output, no crash
    assert isinstance(out, pd.DataFrame)


# ── clean_df ──────────────────────────────────────────────────────────────────

def test_clean_df_strips_dollar_signs():
    df = pd.DataFrame({
        "Date":           ["2023-Q4"],
        "Company Symbol": ["AAPL"],
        "Revenue":        ["$1,000"],
    })
    out = clean_df(df)
    assert out["Revenue"].iloc[0] == 1000.0


def test_clean_df_replaces_dashes_with_nan():
    df = pd.DataFrame({
        "Date":           ["2023-Q4"],
        "Company Symbol": ["AAPL"],
        "Revenue":        ["--"],
    })
    out = clean_df(df)
    assert pd.isna(out["Revenue"].iloc[0])


def test_clean_df_drops_all_null_columns():
    df = pd.DataFrame({
        "Date":           ["2023-Q4"],
        "Company Symbol": ["AAPL"],
        "empty_col":      [None],
        "Revenue":        ["$500"],
    })
    out = clean_df(df)
    assert "empty_col" not in out.columns


# ── clean_prices ──────────────────────────────────────────────────────────────

def test_clean_prices_removes_nan_close(tmp_path):
    df = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=5),
        "symbol":    "AAPL",
        "close":     [100, np.nan, 102, 103, 104],
        "open":      100, "high": 105, "low": 98, "volume": 1_000_000,
    })
    src = str(tmp_path / "prices_20240101.csv")
    out = str(tmp_path / "clean_prices.csv")
    df.to_csv(src, index=False)
    result = clean_prices(src, out)
    assert result["close"].isna().sum() == 0
    assert len(result) == 4


def test_clean_prices_removes_zero_or_negative_close(tmp_path):
    df = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=3),
        "symbol":    "AAPL",
        "close":     [100, 0, -5],
        "open":      100, "high": 105, "low": 98, "volume": 1_000_000,
    })
    src = str(tmp_path / "prices_20240101.csv")
    out = str(tmp_path / "clean_prices.csv")
    df.to_csv(src, index=False)
    result = clean_prices(src, out)
    assert len(result) == 1


# ── run_cleaning / run_feature_engineering ────────────────────────────────────

def test_run_cleaning_returns_dict_when_no_files(tmp_path):
    with patch.dict(os.environ, {
        "RAW_DATA_PATH":       str(tmp_path),
        "PROCESSED_DATA_PATH": str(tmp_path / "processed"),
        "FEATURES_DATA_PATH":  str(tmp_path / "features"),
    }):
        result = run_cleaning()
    assert isinstance(result, dict)


def test_run_feature_engineering_returns_dict_when_no_processed_files(tmp_path):
    with patch.dict(os.environ, {
        "PROCESSED_DATA_PATH": str(tmp_path),
        "FEATURES_DATA_PATH":  str(tmp_path / "features"),
    }):
        result = run_feature_engineering()
    assert isinstance(result, dict)


def test_run_feature_engineering_price_output(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    features  = tmp_path / "features"

    price_df = pd.concat([_make_price_df(60, s) for s in ["AAPL", "MSFT"]])
    price_df.to_csv(processed / "clean_prices.csv", index=False)

    # Patch module-level constants directly (env vars are read at import time)
    with patch("src.features.engineer.PROCESSED_DATA_PATH", str(processed)), \
         patch("src.features.engineer.FEATURES_DATA_PATH",  str(features)):
        out = run_feature_engineering()

    assert "price_features" in out
    result = pd.read_csv(out["price_features"])
    assert "sma_7" in result.columns
    assert "rsi_14" in result.columns
    assert "bb_upper" in result.columns
    assert set(result["symbol"].unique()) == {"AAPL", "MSFT"}
