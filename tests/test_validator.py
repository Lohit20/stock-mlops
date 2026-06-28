"""
Tests for src/ingestion/validator.py
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch
from src.ingestion.validator import (
    check_missing_values,
    check_row_count,
    check_required_columns,
    check_duplicate_rows,
    check_data_freshness,
    check_negative_prices,
    check_symbol_coverage,
    check_price_variance,
    validate_price_data,
    validate_balance_sheet,
    validate_income_statement,
    run_validation,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def price_df():
    n = 500
    dates = pd.date_range(end=datetime.today(), periods=n, freq="D")
    return pd.DataFrame({
        "timestamp": dates,
        "symbol":    np.resize(["AAPL", "MSFT", "GOOG"], n),
        "close":     np.random.uniform(100, 500, n),
        "open":      np.random.uniform(100, 500, n),
        "high":      np.random.uniform(100, 500, n),
        "low":       np.random.uniform(100, 500, n),
        "volume":    np.random.randint(1_000_000, 10_000_000, n),
    })


@pytest.fixture
def balance_df():
    n = 2000
    return pd.DataFrame({
        "Company Symbol": [f"SYM{i}" for i in range(n)],
        "Period Ending:":  ["2023-Q4"] * n,
        "Total Assets":   np.random.uniform(100, 1000, n),
    })


@pytest.fixture
def income_df():
    n = 1000
    return pd.DataFrame({
        "Company Symbol": [f"SYM{i}" for i in range(n)],
        "Period Ending:":  ["2023-Q4"] * n,
        "Net Income":     np.random.uniform(10, 200, n),
    })


# ── check_missing_values ──────────────────────────────────────────────────────

def test_missing_values_passes_below_threshold():
    df = pd.DataFrame({"a": [1, 2, None], "b": [1, 2, 3]})
    result = check_missing_values(df, threshold=0.5)
    assert result.passed


def test_missing_values_fails_above_threshold():
    df = pd.DataFrame({"a": [None, None, 1], "b": [1, 2, 3]})
    result = check_missing_values(df, threshold=0.5)
    assert not result.passed
    assert "a" in result.details


def test_missing_values_all_null_column_fails():
    df = pd.DataFrame({"a": [None, None, None]})
    result = check_missing_values(df, threshold=0.5)
    assert not result.passed


# ── check_row_count ───────────────────────────────────────────────────────────

def test_row_count_passes():
    df = pd.DataFrame({"a": range(200)})
    assert check_row_count(df, min_rows=100).passed


def test_row_count_fails():
    df = pd.DataFrame({"a": range(50)})
    result = check_row_count(df, min_rows=100)
    assert not result.passed
    assert result.details["row_count"] == 50


# ── check_required_columns ────────────────────────────────────────────────────

def test_required_columns_passes():
    df = pd.DataFrame({"a": [1], "b": [2]})
    assert check_required_columns(df, ["a", "b"]).passed


def test_required_columns_fails():
    df = pd.DataFrame({"a": [1]})
    result = check_required_columns(df, ["a", "b"])
    assert not result.passed
    assert "b" in result.details["missing_columns"]


# ── check_duplicate_rows ──────────────────────────────────────────────────────

def test_duplicate_rows_passes():
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert check_duplicate_rows(df, threshold=0.1).passed


def test_duplicate_rows_fails():
    df = pd.DataFrame({"a": [1, 1, 1, 1, 2]})
    result = check_duplicate_rows(df, threshold=0.1)
    assert not result.passed


# ── check_data_freshness ──────────────────────────────────────────────────────

def test_freshness_passes_for_today():
    df = pd.DataFrame({"timestamp": [datetime.today()]})
    assert check_data_freshness(df, "timestamp", max_age_days=1).passed


def test_freshness_fails_for_old_data():
    old_date = datetime.today() - timedelta(days=10)
    df = pd.DataFrame({"timestamp": [old_date]})
    result = check_data_freshness(df, "timestamp", max_age_days=1)
    assert not result.passed
    assert result.details["age_days"] >= 10


def test_freshness_fails_for_missing_column():
    df = pd.DataFrame({"price": [100]})
    result = check_data_freshness(df, "timestamp")
    assert not result.passed


def test_freshness_passes_for_friday_data_on_monday():
    # Friday data is 3 days old on Monday — should pass with max_age_days=1 (allows +2 for weekend)
    friday = datetime.today() - timedelta(days=3)
    df = pd.DataFrame({"timestamp": [friday]})
    assert check_data_freshness(df, "timestamp", max_age_days=1).passed


# ── check_negative_prices ─────────────────────────────────────────────────────

def test_negative_prices_fails():
    df = pd.DataFrame({"close": [100, -5, 200]})
    result = check_negative_prices(df, ["close"])
    assert not result.passed
    assert result.details["close"] == 1


def test_zero_price_fails():
    df = pd.DataFrame({"close": [100, 0, 200]})
    assert not check_negative_prices(df, ["close"]).passed


def test_all_positive_passes():
    df = pd.DataFrame({"close": [100, 200, 300]})
    assert check_negative_prices(df, ["close"]).passed


def test_missing_price_col_skipped():
    df = pd.DataFrame({"volume": [100]})
    result = check_negative_prices(df, ["close"])
    assert result.passed


# ── check_symbol_coverage ─────────────────────────────────────────────────────

def test_symbol_coverage_passes():
    df = pd.DataFrame({"symbol": [f"SYM{i}" for i in range(200)]})
    assert check_symbol_coverage(df, "symbol", min_symbols=100).passed


def test_symbol_coverage_fails():
    df = pd.DataFrame({"symbol": ["AAPL", "MSFT"]})
    result = check_symbol_coverage(df, "symbol", min_symbols=100)
    assert not result.passed
    assert result.details["symbol_count"] == 2


# ── check_price_variance ──────────────────────────────────────────────────────

def test_price_variance_fails_on_flat_prices():
    df = pd.DataFrame({"close": [100.0] * 50})
    assert not check_price_variance(df, "close").passed


def test_price_variance_passes():
    df = pd.DataFrame({"close": np.linspace(100, 200, 50)})
    assert check_price_variance(df, "close").passed


# ── validate_price_data ───────────────────────────────────────────────────────

def test_validate_price_data_passes(price_df):
    with patch("src.ingestion.validator.MIN_PRICE_SYMBOLS", 3):
        passed, checks = validate_price_data(price_df)
    assert passed


def test_validate_price_data_fails_on_missing_column(price_df):
    price_df = price_df.drop(columns=["symbol"])
    passed, _ = validate_price_data(price_df)
    assert not passed


# ── validate_balance_sheet ────────────────────────────────────────────────────

def test_validate_balance_sheet_passes(balance_df):
    passed, _ = validate_balance_sheet(balance_df)
    assert passed


def test_validate_balance_sheet_fails_on_few_rows():
    df = pd.DataFrame({"Company Symbol": ["AAPL"], "Period Ending:": ["2023"]})
    passed, _ = validate_balance_sheet(df)
    assert not passed


# ── validate_income_statement ─────────────────────────────────────────────────

def test_validate_income_statement_passes(income_df):
    passed, _ = validate_income_statement(income_df)
    assert passed


# ── run_validation ────────────────────────────────────────────────────────────

def test_run_validation_passes_with_no_files(tmp_path):
    # No files found → warnings logged, but overall passes (nothing to fail)
    passed = run_validation(str(tmp_path))
    assert passed


def test_run_validation_fails_on_bad_price_file(tmp_path, price_df):
    price_df["close"] = -1  # force failure
    price_df.to_csv(tmp_path / "prices_20240101_120000.csv", index=False)
    with patch("src.ingestion.validator.MIN_PRICE_SYMBOLS", 1):
        passed = run_validation(str(tmp_path))
    assert not passed
