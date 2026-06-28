"""
Data Validator.

Runs after every scrape before data enters the pipeline.
If any check fails → pipeline stops and an alert fires.

Checks per data type:
  Prices (yfinance):     freshness, row count, required columns, negatives, symbol coverage
  Balance sheet:         row count, required columns, missing values, duplicates
  Income statement:      row count, required columns, missing values, duplicates
"""

import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Tuple, List
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

RAW_DATA_PATH = os.getenv("RAW_DATA_PATH", "data/raw")

ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL", "")   # Slack/Teams webhook (optional)
MIN_PRICE_SYMBOLS = int(os.getenv("MIN_PRICE_SYMBOLS", "100"))
MISSING_THRESHOLD  = float(os.getenv("MISSING_THRESHOLD", "0.8"))
DUPLICATE_THRESHOLD = float(os.getenv("DUPLICATE_THRESHOLD", "0.1"))


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    passed:  bool
    message: str
    details: dict


# ── Generic checks ────────────────────────────────────────────────────────────

def check_missing_values(df: pd.DataFrame, threshold: float = 0.5) -> ValidationResult:
    """Fail if more than `threshold` fraction of any column is missing."""
    missing_ratio = df.isnull().mean()
    bad_columns   = missing_ratio[missing_ratio > threshold]
    if len(bad_columns) > 0:
        return ValidationResult(
            passed=False,
            message=f"{len(bad_columns)} columns exceed {threshold*100:.0f}% missing values",
            details=bad_columns.to_dict(),
        )
    return ValidationResult(passed=True, message="Missing values OK", details={})


def check_row_count(df: pd.DataFrame, min_rows: int = 100) -> ValidationResult:
    """Fail if dataset has fewer rows than expected."""
    if len(df) < min_rows:
        return ValidationResult(
            passed=False,
            message=f"Only {len(df)} rows, expected >= {min_rows}",
            details={"row_count": len(df)},
        )
    return ValidationResult(passed=True, message=f"Row count OK: {len(df)}", details={})


def check_required_columns(df: pd.DataFrame, required_cols: list) -> ValidationResult:
    """Fail if any expected columns are missing."""
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return ValidationResult(
            passed=False,
            message=f"Missing required columns: {missing}",
            details={"missing_columns": missing},
        )
    return ValidationResult(passed=True, message="All required columns present", details={})


def check_duplicate_rows(df: pd.DataFrame, threshold: float = 0.1) -> ValidationResult:
    """Fail if more than `threshold` fraction of rows are duplicates."""
    dup_ratio = df.duplicated().sum() / len(df)
    if dup_ratio > threshold:
        return ValidationResult(
            passed=False,
            message=f"{dup_ratio*100:.1f}% duplicate rows (max {threshold*100:.0f}%)",
            details={"duplicate_ratio": dup_ratio},
        )
    return ValidationResult(
        passed=True,
        message=f"Duplicate check OK ({dup_ratio*100:.1f}%)",
        details={},
    )


# ── Domain-specific checks ────────────────────────────────────────────────────

def check_data_freshness(
    df: pd.DataFrame,
    date_col: str,
    max_age_days: int = 1,
) -> ValidationResult:
    """
    Fail if the most recent record is older than max_age_days.
    Skips weekends — markets are closed Sat/Sun so Friday data is valid on Monday.
    """
    if date_col not in df.columns:
        return ValidationResult(
            passed=False,
            message=f"Date column '{date_col}' not found",
            details={},
        )

    try:
        dates      = pd.to_datetime(df[date_col], errors="coerce").dropna()
        most_recent = dates.max()
    except Exception as exc:
        return ValidationResult(
            passed=False,
            message=f"Could not parse dates in '{date_col}': {exc}",
            details={},
        )

    now       = pd.Timestamp.now(tz=most_recent.tzinfo)
    age_days  = (now - most_recent).days

    # Allow up to max_age_days + 2 weekend days
    allowed = max_age_days + 2
    if age_days > allowed:
        return ValidationResult(
            passed=False,
            message=f"Data is {age_days} days old (max {allowed}). Most recent: {most_recent.date()}",
            details={"most_recent": str(most_recent.date()), "age_days": age_days},
        )
    return ValidationResult(
        passed=True,
        message=f"Data freshness OK (most recent: {most_recent.date()}, {age_days}d ago)",
        details={"most_recent": str(most_recent.date())},
    )


def check_negative_prices(df: pd.DataFrame, price_cols: list) -> ValidationResult:
    """Fail if any price column contains values <= 0."""
    bad = {}
    for col in price_cols:
        if col not in df.columns:
            continue
        n_bad = (df[col] <= 0).sum()
        if n_bad > 0:
            bad[col] = int(n_bad)

    if bad:
        return ValidationResult(
            passed=False,
            message=f"Negative/zero prices found in: {list(bad.keys())}",
            details=bad,
        )
    return ValidationResult(passed=True, message="All prices positive", details={})


def check_symbol_coverage(df: pd.DataFrame, symbol_col: str, min_symbols: int = 100) -> ValidationResult:
    """Fail if fewer than min_symbols unique tickers are present."""
    if symbol_col not in df.columns:
        return ValidationResult(
            passed=False,
            message=f"Symbol column '{symbol_col}' not found",
            details={},
        )
    n = df[symbol_col].nunique()
    if n < min_symbols:
        return ValidationResult(
            passed=False,
            message=f"Only {n} symbols present, expected >= {min_symbols}",
            details={"symbol_count": n},
        )
    return ValidationResult(
        passed=True,
        message=f"Symbol coverage OK: {n} symbols",
        details={"symbol_count": n},
    )


def check_price_variance(df: pd.DataFrame, price_col: str = "close") -> ValidationResult:
    """
    Fail if all prices are identical — indicates a stuck/stale feed.
    """
    if price_col not in df.columns:
        return ValidationResult(passed=True, message=f"Column '{price_col}' not present, skipped", details={})

    if df[price_col].std() == 0:
        return ValidationResult(
            passed=False,
            message=f"Zero variance in '{price_col}' — data may be stale or stuck",
            details={"std": 0},
        )
    return ValidationResult(passed=True, message="Price variance OK", details={})


# ── Domain validators ─────────────────────────────────────────────────────────

def validate_price_data(df: pd.DataFrame) -> Tuple[bool, List[ValidationResult]]:
    """
    Validate yfinance price output.
    Expected columns: timestamp, symbol, close, open, high, low, volume
    """
    checks = [
        check_row_count(df, min_rows=100),
        check_required_columns(df, ["timestamp", "symbol", "close"]),
        check_missing_values(df, threshold=0.1),
        check_negative_prices(df, price_cols=["close", "open", "high", "low"]),
        check_symbol_coverage(df, symbol_col="symbol", min_symbols=MIN_PRICE_SYMBOLS),
        check_data_freshness(df, date_col="timestamp", max_age_days=1),
        check_price_variance(df, price_col="close"),
    ]
    return _log_and_return(checks)


def validate_balance_sheet(df: pd.DataFrame) -> Tuple[bool, List[ValidationResult]]:
    """Validate scraped balance sheet data."""
    checks = [
        check_row_count(df, min_rows=1000),
        check_required_columns(df, ["Company Symbol", "Period Ending:"]),
        check_missing_values(df, threshold=MISSING_THRESHOLD),
        check_duplicate_rows(df, threshold=DUPLICATE_THRESHOLD),
    ]
    return _log_and_return(checks)


def validate_income_statement(df: pd.DataFrame) -> Tuple[bool, List[ValidationResult]]:
    """Validate scraped income statement data."""
    checks = [
        check_row_count(df, min_rows=500),
        check_required_columns(df, ["Company Symbol", "Period Ending:"]),
        check_missing_values(df, threshold=MISSING_THRESHOLD),
        check_duplicate_rows(df, threshold=DUPLICATE_THRESHOLD),
    ]
    return _log_and_return(checks)


def _log_and_return(checks: List[ValidationResult]) -> Tuple[bool, List[ValidationResult]]:
    all_passed = all(c.passed for c in checks)
    for c in checks:
        if c.passed:
            logger.info(f"  ✅ {c.message}")
        else:
            logger.error(f"  ❌ {c.message} | {c.details}")
    return all_passed, checks


# ── Alerting ──────────────────────────────────────────────────────────────────

def send_alert(subject: str, body: str):
    """
    Fire an alert when validation fails.
    Logs at ERROR level always. Sends to webhook if ALERT_WEBHOOK_URL is set.
    """
    logger.error(f"ALERT — {subject}: {body}")

    if not ALERT_WEBHOOK_URL:
        return

    try:
        import requests
        payload = {"text": f"*{subject}*\n{body}"}
        resp    = requests.post(ALERT_WEBHOOK_URL, json=payload, timeout=10)
        if resp.status_code != 200:
            logger.warning(f"Alert webhook returned {resp.status_code}")
    except Exception as exc:
        logger.warning(f"Alert webhook failed: {exc}")


# ── Main validation entry point ───────────────────────────────────────────────

def run_validation(raw_data_path: str = RAW_DATA_PATH) -> bool:
    """
    Validate all latest raw data files.
    Called by Airflow after scraping.

    Returns:
        True if all validations pass, False otherwise.
    """
    overall_passed = True

    # ── 1. Price data ─────────────────────────────────────────────────────────
    price_files = sorted(glob.glob(os.path.join(raw_data_path, "prices_*.csv")))
    if price_files:
        latest_price = price_files[-1]
        logger.info(f"Validating price data: {os.path.basename(latest_price)}")
        df     = pd.read_csv(latest_price)
        passed, _ = validate_price_data(df)
        if not passed:
            send_alert(
                "Price data validation FAILED",
                f"File: {latest_price}. Pipeline will stop.",
            )
            overall_passed = False
    else:
        logger.warning("No price files found in raw data — skipping price validation")

    # ── 2. Balance sheet ──────────────────────────────────────────────────────
    bs_files = sorted(glob.glob(os.path.join(raw_data_path, "balanceSheetTable_*.csv")))
    if bs_files:
        latest_bs = bs_files[-1]
        logger.info(f"Validating balance sheet: {os.path.basename(latest_bs)}")
        df     = pd.read_csv(latest_bs)
        passed, _ = validate_balance_sheet(df)
        if not passed:
            send_alert(
                "Balance sheet validation FAILED",
                f"File: {latest_bs}. Pipeline will stop.",
            )
            overall_passed = False
    else:
        logger.warning("No balance sheet files found — skipping balance sheet validation")

    # ── 3. Income statement ───────────────────────────────────────────────────
    is_files = sorted(glob.glob(os.path.join(raw_data_path, "incomeStatementTable_*.csv")))
    if is_files:
        latest_is = is_files[-1]
        logger.info(f"Validating income statement: {os.path.basename(latest_is)}")
        df     = pd.read_csv(latest_is)
        passed, _ = validate_income_statement(df)
        if not passed:
            send_alert(
                "Income statement validation FAILED",
                f"File: {latest_is}. Pipeline will stop.",
            )
            overall_passed = False
    else:
        logger.warning("No income statement files found — skipping income statement validation")

    if overall_passed:
        logger.info("✅ All validations passed")
    else:
        logger.error("❌ One or more validations failed — pipeline should stop")

    return overall_passed


if __name__ == "__main__":
    passed = run_validation()
    raise SystemExit(0 if passed else 1)
