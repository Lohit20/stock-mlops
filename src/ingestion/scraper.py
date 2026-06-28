"""
Live NASDAQ scraper using Selenium.

What this file does:
- Connects to NASDAQ website
- Scrapes balance sheet + income statement for all companies
- Saves raw data to data/raw/ with a timestamp
- Called daily by Airflow
"""

import os
import pandas as pd
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

RAW_DATA_PATH = os.getenv("RAW_DATA_PATH", "data/raw")
NASDAQ_BASE_URL = os.getenv("NASDAQ_BASE_URL", "https://www.nasdaq.com/market-activity/stocks")


def get_driver() -> webdriver.Chrome:
    """Create a headless Chrome driver (no browser window opens)."""
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")       # Run without opening browser window
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    return webdriver.Chrome(options=options)


def scrape_company_financials(driver: webdriver.Chrome, symbol: str, panel_names: list) -> dict:
    """
    Scrape financial tables for a single company symbol.

    Args:
        driver: Selenium Chrome driver
        symbol: Stock ticker e.g. 'AAPL'
        panel_names: ['balanceSheetTable', 'incomeStatementTable']

    Returns:
        Dictionary with panel_name -> list of rows
    """
    url = f"{NASDAQ_BASE_URL}/{symbol}/financials"
    table_data = {}

    for attempt in range(3):  # Retry up to 3 times
        try:
            driver.get(url)
            for panel_name in panel_names:
                # JavaScript extracts data from the rendered page
                # (XPath doesn't work because NASDAQ uses JavaScript rendering)
                script = f'''
                    var table = document.querySelector(
                        'div[data-panel-name="{panel_name}"] table.financials__table'
                    );
                    var data = [];
                    if (table) {{
                        var rows = table.querySelectorAll('tbody tr');
                        var headers = table.querySelectorAll('thead tr th[scope="col"]');
                        var header_data = Array.from(headers, h => h.textContent.trim());
                        data.push(header_data);
                        rows.forEach(row => {{
                            var cells = row.querySelectorAll('th[scope="row"], td');
                            data.push(Array.from(cells, c => c.textContent.trim()));
                        }});
                    }}
                    return data;
                '''
                result = driver.execute_script(script)
                if result:
                    # Prepend company symbol to each row
                    result = [[symbol] + row for row in result]
                    result[0][0] = "Company Symbol"
                    table_data[panel_name] = result
                else:
                    logger.warning(f"No data for {symbol} in {panel_name}")

            break  # Success — exit retry loop

        except TimeoutException:
            logger.warning(f"Timeout for {symbol}, attempt {attempt + 1}/3")
            if attempt == 2:
                logger.error(f"Failed to scrape {symbol} after 3 attempts")

    return table_data


def save_to_csv(data: list, file_path: str):
    """Append scraped rows to CSV file."""
    df = pd.DataFrame(data)
    header = not os.path.exists(file_path)  # Write header only for new file
    df.to_csv(file_path, mode='a', header=header, index=False)


def run_scraper(symbols: list = None):
    """
    Main scraping function — called by Airflow daily.

    Args:
        symbols: List of stock tickers. If None, reads from nasdaq_screener.csv
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(RAW_DATA_PATH, exist_ok=True)

    # Load symbols if not provided
    if symbols is None:
        screener_path = os.path.join(RAW_DATA_PATH, "nasdaq_screener.csv")
        if not os.path.exists(screener_path):
            raise FileNotFoundError(f"nasdaq_screener.csv not found at {screener_path}")
        symbols = pd.read_csv(screener_path)["Symbol"].tolist()

    panel_names = ["balanceSheetTable", "incomeStatementTable"]
    output_files = {
        panel: os.path.join(RAW_DATA_PATH, f"{panel}_{timestamp}.csv")
        for panel in panel_names
    }

    logger.info(f"Starting scrape for {len(symbols)} symbols")
    driver = get_driver()

    try:
        for i, symbol in enumerate(symbols):
            logger.info(f"Scraping {symbol} ({i+1}/{len(symbols)})")
            data = scrape_company_financials(driver, symbol, panel_names)

            for panel_name, rows in data.items():
                if rows:
                    save_to_csv(rows[1:], output_files[panel_name])  # Skip header row
    finally:
        driver.quit()

    logger.info(f"Scraping complete. Files saved to {RAW_DATA_PATH}")
    return output_files


if __name__ == "__main__":
    run_scraper()
