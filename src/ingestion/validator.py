"""
Data Validator.

What this file does:
- Checks scraped data quality BEFORE it enters the pipeline
- If data is bad → pipeline stops and alerts you
- If data is good → pipeline continues

Think of it as a bouncer at a club:
Bad data doesn't get in.
"""

import pandas as pd
import numpy as np
from loguru import logger
from dataclasses import dataclass
from typing import Tuple


@dataclass
class ValidationResult:
    """Holds the result of a validation check."""
    passed: bool          # True = data is good, False = data is bad
    message: str          # Human-readable explanation
    details: dict         # Extra information about the check


def check_missing_values(df: pd.DataFrame, threshold: float = 0.5) -> ValidationResult:
    """
    Fail if more than 50% of any column is missing.

    Args:
        df: DataFrame to check
        threshold: Maximum allowed fraction of missing values (0.5 = 50%)
    """
    missing_ratio = df.isnull().mean()
    bad_columns = missing_ratio[missing_ratio > threshold]

    if len(bad_columns) > 0:
        return ValidationResult(
            passed=False,
            message=f"{len(bad_columns)} columns exceed {threshold*100}% missing values",
            details=bad_columns.to_dict()
        )
    return ValidationResult(passed=True, message="Missing values check passed", details={})


def check_row_count(df: pd.DataFrame, min_rows: int = 100) -> ValidationResult:
    """Fail if dataset has fewer rows than expected."""
    if len(df) < min_rows:
        return ValidationResult(
            passed=False,
            message=f"Only {len(df)} rows found, expected at least {min_rows}",
            details={"row_count": len(df)}
        )
    return ValidationResult(passed=True, message=f"Row count OK: {len(df)} rows", details={})


def check_required_columns(df: pd.DataFrame, required_cols: list) -> ValidationResult:
    """Fail if any expected columns are missing from the dataset."""
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        return ValidationResult(
            passed=False,
            message=f"Missing required columns: {missing}",
            details={"missing_columns": missing}
        )
    return ValidationResult(passed=True, message="All required columns present", details={})


def check_duplicate_rows(df: pd.DataFrame, threshold: float = 0.1) -> ValidationResult:
    """Fail if more than 10% of rows are duplicates."""
    dup_ratio = df.duplicated().sum() / len(df)
    if dup_ratio > threshold:
        return ValidationResult(
            passed=False,
            message=f"{dup_ratio*100:.1f}% duplicate rows found (max {threshold*100}%)",
            details={"duplicate_ratio": dup_ratio}
        )
    return ValidationResult(passed=True, message=f"Duplicate check passed ({dup_ratio*100:.1f}%)", details={})


def validate_balance_sheet(df: pd.DataFrame) -> Tuple[bool, list]:
    """
    Run all validation checks on balance sheet data.

    Returns:
        (all_passed, list of ValidationResult)
    """
    required_cols = ["Company Symbol", "Period Ending:"]
    checks = [
        check_row_count(df, min_rows=1000),
        check_required_columns(df, required_cols),
        check_missing_values(df, threshold=0.8),
        check_duplicate_rows(df, threshold=0.1),
    ]

    all_passed = all(c.passed for c in checks)
    for check in checks:
        if check.passed:
            logger.info(f"✅ {check.message}")
        else:
            logger.error(f"❌ {check.message} | Details: {check.details}")

    return all_passed, checks


def validate_income_statement(df: pd.DataFrame) -> Tuple[bool, list]:
    """Run all validation checks on income statement data."""
    required_cols = ["Company Symbol", "Period Ending:"]
    checks = [
        check_row_count(df, min_rows=500),
        check_required_columns(df, required_cols),
        check_missing_values(df, threshold=0.8),
        check_duplicate_rows(df, threshold=0.1),
    ]

    all_passed = all(c.passed for c in checks)
    for check in checks:
        if check.passed:
            logger.info(f"✅ {check.message}")
        else:
            logger.error(f"❌ {check.message} | Details: {check.details}")

    return all_passed, checks


def validate_price_data(df: pd.DataFrame) -> Tuple[bool, list]:
    """Run all validation checks on price time series data."""
    checks = [
        check_row_count(df, min_rows=100),
        check_required_columns(df, ["timestamp", "price"]),
        check_missing_values(df, threshold=0.1),
    ]

    # Extra check: prices must be positive
    if "price" in df.columns:
        negative_prices = (df["price"] <= 0).sum()
        if negative_prices > 0:
            checks.append(ValidationResult(
                passed=False,
                message=f"{negative_prices} negative/zero prices found",
                details={"count": negative_prices}
            ))
        else:
            checks.append(ValidationResult(
                passed=True,
                message="All prices are positive",
                details={}
            ))

    all_passed = all(c.passed for c in checks)
    for check in checks:
        if check.passed:
            logger.info(f"✅ {check.message}")
        else:
            logger.error(f"❌ {check.message} | Details: {check.details}")

    return all_passed, checks


if __name__ == "__main__":
    # Quick test with existing data
    df = pd.read_csv("data/raw/balanceSheetTable.csv")
    passed, results = validate_balance_sheet(df)
    print(f"\nValidation {'PASSED' if passed else 'FAILED'}")
