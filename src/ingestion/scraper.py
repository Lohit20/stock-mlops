"""
Live NASDAQ scraper + price fetcher.

What this file does:
- Fetches daily price data for all symbols via yfinance (fast, reliable)
- Scrapes balance sheet + income statement from NASDAQ via Selenium (quarterly)
- Saves raw data to data/raw/ with a timestamp
- Called daily by Airflow
"""

import os
import time
import pandas as pd
from datetime import datetime
from selenium import webdriver
from selenium.common.exceptions import (
    NoSuchElementException,
    TimeoutException,
    WebDriverException,
)
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

RAW_DATA_PATH    = os.getenv("RAW_DATA_PATH", "data/raw")
NASDAQ_BASE_URL  = os.getenv("NASDAQ_BASE_URL", "https://www.nasdaq.com/market-activity/stocks")
SCRAPE_DELAY     = float(os.getenv("SCRAPE_DELAY_SECONDS", "2.0"))   # seconds between symbol requests
PAGE_LOAD_TIMEOUT = int(os.getenv("PAGE_LOAD_TIMEOUT", "30"))


# ── Driver ────────────────────────────────────────────────────────────────────

def get_driver() -> webdriver.Chrome:
    """Create a headless Chrome driver. Uses webdriver-manager if available."""
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    # Mimic a real browser to reduce bot-detection blocks
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )

    try:
        from webdriver_manager.chrome import ChromeDriverManager
        from selenium.webdriver.chrome.service import Service
        service = Service(ChromeDriverManager().install())
        driver  = webdriver.Chrome(service=service, options=options)
    except ImportError:
        driver = webdriver.Chrome(options=options)

    driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
    return driver


# ── Financial statement scraper ───────────────────────────────────────────────

def scrape_company_financials(driver: webdriver.Chrome, symbol: str, panel_names: list) -> dict:
    """
    Scrape financial tables for a single company symbol from NASDAQ.

    Args:
        driver:      Selenium Chrome driver
        symbol:      Stock ticker e.g. 'AAPL'
        panel_names: ['balanceSheetTable', 'incomeStatementTable']

    Returns:
        {panel_name: [header_row, data_row, ...]}
    """
    url        = f"{NASDAQ_BASE_URL}/{symbol}/financials"
    table_data = {}

    for attempt in range(3):
        try:
            driver.get(url)
            time.sleep(SCRAPE_DELAY)  # polite delay after page load

            for panel_name in panel_names:
                script = f'''
                    var table = document.querySelector(
                        'div[data-panel-name="{panel_name}"] table.financials__table'
                    );
                    var data = [];
                    if (table) {{
                        var headers = table.querySelectorAll('thead tr th[scope="col"]');
                        var header_data = Array.from(headers, h => h.textContent.trim());
                        data.push(header_data);
                        var rows = table.querySelectorAll('tbody tr');
                        rows.forEach(row => {{
                            var cells = row.querySelectorAll('th[scope="row"], td');
                            data.push(Array.from(cells, c => c.textContent.trim()));
                        }});
                    }}
                    return data;
                '''
                result = driver.execute_script(script)

                if result:
                    result        = [[symbol] + row for row in result]
                    result[0][0]  = "Company Symbol"
                    table_data[panel_name] = result
                else:
                    logger.warning(f"No table data for {symbol}/{panel_name}")

            break  # success — exit retry loop

        except (TimeoutException, NoSuchElementException, WebDriverException) as exc:
            logger.warning(f"{type(exc).__name__} for {symbol}, attempt {attempt + 1}/3")
            if attempt == 2:
                logger.error(f"Failed to scrape {symbol} after 3 attempts: {exc}")

    return table_data


# ── CSV writer ────────────────────────────────────────────────────────────────

def save_to_csv(rows: list, file_path: str):
    """
    Append scraped rows to a CSV file.

    Args:
        rows: Full result including header row at index 0.
    """
    if not rows:
        return

    header   = rows[0]
    data     = rows[1:]
    df       = pd.DataFrame(data, columns=header)
    write_hdr = not os.path.exists(file_path)
    df.to_csv(file_path, mode="a", header=write_hdr, index=False)


# ── Price fetcher (yfinance) ──────────────────────────────────────────────────

def fetch_prices(symbols: list, period: str = "2y", interval: str = "1d") -> str:
    """
    Download daily OHLCV price data for all symbols using yfinance.

    Much faster than Selenium for price data — no browser needed.
    Downloads up to ~1,500 symbols per batch to stay within API limits.

    Args:
        symbols:  List of ticker strings
        period:   yfinance period string e.g. '2y', '1y', '6mo'
        interval: Bar interval e.g. '1d', '1wk'

    Returns:
        Path to saved CSV file
    """
    import yfinance as yf

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(RAW_DATA_PATH, f"prices_{timestamp}.csv")
    os.makedirs(RAW_DATA_PATH, exist_ok=True)

    batch_size = 100  # yfinance handles ~100 symbols comfortably per call
    all_frames = []

    for i in range(0, len(symbols), batch_size):
        batch = symbols[i : i + batch_size]
        logger.info(f"Fetching prices {i+1}–{min(i+batch_size, len(symbols))} of {len(symbols)}")

        try:
            raw = yf.download(
                tickers=batch,
                period=period,
                interval=interval,
                group_by="ticker",
                auto_adjust=True,
                progress=False,
            )
        except Exception as exc:
            logger.error(f"yfinance batch {i}–{i+batch_size} failed: {exc}")
            continue

        # Reshape multi-ticker result into (timestamp, symbol, close) rows
        if len(batch) == 1:
            sym = batch[0]
            df  = raw[["Close"]].copy()
            df.columns = ["close"]
            df["open"]   = raw["Open"]
            df["high"]   = raw["High"]
            df["low"]    = raw["Low"]
            df["volume"] = raw["Volume"]
            df.insert(0, "symbol", sym)
            df.index.name = "timestamp"
            all_frames.append(df.reset_index())
        else:
            for sym in batch:
                if sym not in raw.columns.get_level_values(0):
                    continue
                sym_df = raw[sym][["Close", "Open", "High", "Low", "Volume"]].copy()
                sym_df.columns = ["close", "open", "high", "low", "volume"]
                sym_df.insert(0, "symbol", sym)
                sym_df.index.name = "timestamp"
                all_frames.append(sym_df.reset_index())

        time.sleep(0.5)  # avoid rate limiting between batches

    if not all_frames:
        logger.error("No price data fetched")
        return output_path

    combined = pd.concat(all_frames, ignore_index=True)
    combined.dropna(subset=["close"], inplace=True)
    combined.to_csv(output_path, index=False)

    logger.info(f"Price data saved → {output_path} ({len(combined):,} rows, {combined['symbol'].nunique()} symbols)")
    return output_path


# ── Main entry points ─────────────────────────────────────────────────────────

def run_price_fetcher(symbols: list = None, period: str = "2y") -> str:
    """
    Fetch and save daily price data. Called by Airflow daily.

    Args:
        symbols: List of tickers. If None, reads from nasdaq_screener.csv.
        period:  How far back to pull data.

    Returns:
        Path to saved prices CSV.
    """
    os.makedirs(RAW_DATA_PATH, exist_ok=True)

    if symbols is None:
        screener_path = os.path.join(RAW_DATA_PATH, "nasdaq_screener.csv")
        if not os.path.exists(screener_path):
            raise FileNotFoundError(f"nasdaq_screener.csv not found at {screener_path}")
        symbols = pd.read_csv(screener_path)["Symbol"].dropna().tolist()

    logger.info(f"Fetching prices for {len(symbols)} symbols (period={period})")
    return fetch_prices(symbols, period=period)


def run_scraper(symbols: list = None) -> dict:
    """
    Scrape financial statements (balance sheet + income statement) via Selenium.
    Called by Airflow daily — runs after market close.

    Args:
        symbols: List of tickers. If None, reads from nasdaq_screener.csv.

    Returns:
        Dict of panel_name → output file path.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(RAW_DATA_PATH, exist_ok=True)

    if symbols is None:
        screener_path = os.path.join(RAW_DATA_PATH, "nasdaq_screener.csv")
        if not os.path.exists(screener_path):
            raise FileNotFoundError(f"nasdaq_screener.csv not found at {screener_path}")
        symbols = pd.read_csv(screener_path)["Symbol"].dropna().tolist()

    if not symbols:
        logger.warning("Symbol list is empty — nothing to scrape")
        return {}

    panel_names  = ["balanceSheetTable", "incomeStatementTable"]
    output_files = {
        panel: os.path.join(RAW_DATA_PATH, f"{panel}_{timestamp}.csv")
        for panel in panel_names
    }

    logger.info(f"Starting financial scrape for {len(symbols)} symbols")
    driver = get_driver()

    try:
        for i, symbol in enumerate(symbols):
            logger.info(f"Scraping {symbol} ({i+1}/{len(symbols)})")
            data = scrape_company_financials(driver, symbol, panel_names)

            for panel_name, rows in data.items():
                if rows:
                    save_to_csv(rows, output_files[panel_name])
    finally:
        driver.quit()

    logger.info(f"Scraping complete → {RAW_DATA_PATH}")
    return output_files


if __name__ == "__main__":
    run_scraper()
