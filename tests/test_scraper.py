"""
Tests for src/ingestion/scraper.py
"""

import os
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch, call
from src.ingestion.scraper import (
    scrape_company_financials,
    save_to_csv,
    run_scraper,
)


# ── save_to_csv ───────────────────────────────────────────────────────────────

def test_save_to_csv_creates_file(tmp_path):
    path = str(tmp_path / "out.csv")
    # First row is the header
    rows = [["symbol", "value", "other"], ["AAPL", "100", "200"], ["MSFT", "150", "300"]]
    save_to_csv(rows, path)
    assert os.path.exists(path)
    df = pd.read_csv(path)
    assert len(df) == 2
    assert list(df.columns) == ["symbol", "value", "other"]


def test_save_to_csv_appends(tmp_path):
    path = str(tmp_path / "out.csv")
    save_to_csv([["symbol", "value"], ["AAPL", "100"]], path)
    save_to_csv([["symbol", "value"], ["MSFT", "200"]], path)
    df = pd.read_csv(path)
    assert len(df) == 2


def test_save_to_csv_no_duplicate_header(tmp_path):
    path = str(tmp_path / "out.csv")
    rows = [["symbol", "value"], ["AAPL", "100"]]
    save_to_csv(rows, path)
    save_to_csv(rows, path)
    with open(path) as f:
        lines = f.readlines()
    # One header line + two data rows = 3 total lines
    assert len(lines) == 3


# ── scrape_company_financials ─────────────────────────────────────────────────

def _make_driver(script_return):
    """Return a mock Selenium driver that returns script_return on execute_script."""
    driver = MagicMock()
    driver.execute_script.return_value = script_return
    return driver


def test_scrape_returns_data_for_both_panels():
    fake_data = [
        ["Period Ending:", "2023-Q4", "2022-Q4"],
        ["Total Assets", "100M", "90M"],
    ]
    driver = _make_driver(fake_data)
    panels = ["balanceSheetTable", "incomeStatementTable"]
    result = scrape_company_financials(driver, "AAPL", panels)
    assert "balanceSheetTable" in result
    assert "incomeStatementTable" in result
    assert result["balanceSheetTable"][0][0] == "Company Symbol"


def test_scrape_empty_result_when_no_table():
    driver = _make_driver([])  # empty list = no table found
    result = scrape_company_financials(driver, "AAPL", ["balanceSheetTable"])
    assert result == {}


def test_scrape_retries_on_timeout():
    from selenium.common.exceptions import TimeoutException
    driver = MagicMock()
    driver.get.side_effect = [TimeoutException, TimeoutException, None]
    driver.execute_script.return_value = [["Period Ending:", "2023"]]
    result = scrape_company_financials(driver, "AAPL", ["balanceSheetTable"])
    assert driver.get.call_count == 3


def test_scrape_symbol_prepended_to_rows():
    fake_data = [
        ["Period Ending:", "2023-Q4"],
        ["Total Assets", "100M"],
    ]
    driver = _make_driver(fake_data)
    result = scrape_company_financials(driver, "TSLA", ["balanceSheetTable"])
    rows = result["balanceSheetTable"]
    assert rows[1][0] == "TSLA"


# ── run_scraper ───────────────────────────────────────────────────────────────

@patch("src.ingestion.scraper.get_driver")
@patch("src.ingestion.scraper.scrape_company_financials")
@patch("src.ingestion.scraper.save_to_csv")
def test_run_scraper_calls_save_for_each_panel(mock_save, mock_scrape, mock_driver, tmp_path):
    mock_driver.return_value = MagicMock()
    mock_scrape.return_value = {
        "balanceSheetTable":    [["Company Symbol", "Period Ending:"], ["AAPL", "2023"]],
        "incomeStatementTable": [["Company Symbol", "Period Ending:"], ["AAPL", "2023"]],
    }
    with patch.dict(os.environ, {"RAW_DATA_PATH": str(tmp_path)}):
        run_scraper(symbols=["AAPL"])
    # save_to_csv called once per panel with full rows (header + data)
    assert mock_save.call_count == 2
    for call_args in mock_save.call_args_list:
        rows = call_args[0][0]
        assert rows[0][0] == "Company Symbol"  # header row preserved


@patch("src.ingestion.scraper.get_driver")
@patch("src.ingestion.scraper.scrape_company_financials", return_value={})
def test_run_scraper_handles_empty_scrape(mock_scrape, mock_driver, tmp_path):
    mock_driver.return_value = MagicMock()
    with patch.dict(os.environ, {"RAW_DATA_PATH": str(tmp_path)}):
        result = run_scraper(symbols=["AAPL"])
    assert isinstance(result, dict)


def test_run_scraper_raises_if_no_screener_and_no_symbols(tmp_path):
    # Patch the module-level constant directly — env var patch alone doesn't
    # work because RAW_DATA_PATH is read at import time, not at call time.
    with patch("src.ingestion.scraper.RAW_DATA_PATH", str(tmp_path)):
        with pytest.raises(FileNotFoundError):
            run_scraper(symbols=None)
