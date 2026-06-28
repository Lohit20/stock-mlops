"""
Data Cleaning + Feature Engineering Pipeline.

Two stages:
  1. run_cleaning()        — raw → data/processed/
  2. run_feature_engineering() — processed → data/features/

Price features (per symbol, rolling):
  Moving averages : SMA 7/14/30, EMA 7/14/30
  Momentum        : 7d / 14d / 30d price change
  Volatility      : rolling 7d / 30d std
  Returns         : daily % and log returns
  RSI             : 14-period relative strength index
  Bollinger Bands : 20-day mid / upper / lower / width

Financial ratios (per symbol, per quarter):
  debt_ratio, current_ratio, profit_margin, roa, roe
"""

import os
import glob
import pandas as pd
import numpy as np
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

RAW_DATA_PATH       = os.getenv("RAW_DATA_PATH",       "data/raw")
PROCESSED_DATA_PATH = os.getenv("PROCESSED_DATA_PATH", "data/processed")
FEATURES_DATA_PATH  = os.getenv("FEATURES_DATA_PATH",  "data/features")


# ── File discovery ─────────────────────────────────────────────────────────────

def _latest_file(directory: str, pattern: str) -> str:
    """Return the most recently created file matching a glob pattern."""
    matches = glob.glob(os.path.join(directory, pattern))
    if not matches:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found in {directory}"
        )
    return max(matches, key=os.path.getctime)


# ── Financial statement cleaning (original notebook logic, preserved) ──────────

def transpose_df(df: pd.DataFrame, dates: list) -> pd.DataFrame:
    """
    Reshape scraped financial data from metric-per-row to company-per-row.

    Raw format (rows = metrics, cols = quarters):
        Company Symbol | Period Ending:  | 2023-Q4 | 2022-Q4 | ...
        AAPL           | Total Assets    | 100M    | 90M     | ...
        AAPL           | Total Liab.     | 40M     | 35M     | ...

    Output format (rows = (date, company), cols = metrics):
        Date    | Company Symbol | Total Assets | Total Liab. | ...
        2023-Q4 | AAPL           | 100M         | 40M         | ...
    """
    transposed = []
    for date in dates:
        d = df.pivot(index="Company Symbol", columns="Period Ending:", values=date)
        d.reset_index(inplace=True)
        d.rename_axis(columns={"Period Ending:": ""}, inplace=True)
        d.reset_index(drop=True, inplace=True)
        d.insert(0, "Date", date)
        d = d.iloc[1:].reset_index(drop=True)
        transposed.append(d)
    result = pd.concat(transposed, axis=0, ignore_index=True)
    return result.rename(columns={result.columns[1]: "Company Symbol"})


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a financial DataFrame:
    - Drop fully-null columns
    - Replace '--' with NaN
    - Strip '$' and ',' then cast to numeric
    - Fill remaining NaN with per-Date group mean (numeric columns only)
    """
    df = df.dropna(axis=1, how="all")
    df = df.replace("--", np.nan)
    subset = df.iloc[:, 2:].replace(r"[\$,]", "", regex=True)
    df.iloc[:, 2:] = subset.apply(pd.to_numeric, errors="coerce")
    # Only fill numeric columns — transform('mean') raises on string cols in pandas ≥ 2.0
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        df[num_cols] = df[num_cols].fillna(
            df.groupby("Date")[num_cols].transform("mean")
        )
    return df


def clean_financial(raw_path: str, out_path: str) -> pd.DataFrame:
    """
    Full clean of a scraped financial CSV (balance sheet or income statement).
    Saves result to out_path and returns the cleaned DataFrame.
    """
    df = pd.read_csv(raw_path)

    # Column 0 = "Company Symbol", column 1 = "Period Ending:", rest = quarter dates
    date_cols = [c for c in df.columns if c not in ("Company Symbol", "Period Ending:")]
    if not date_cols:
        raise ValueError(f"No date columns found in {raw_path}")

    df = transpose_df(df, date_cols)
    df = clean_df(df)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"Saved cleaned financial data → {out_path} ({len(df)} rows)")
    return df


def clean_prices(raw_path: str, out_path: str) -> pd.DataFrame:
    """
    Clean yfinance price CSV.
    Expected columns: timestamp, symbol, close, open, high, low, volume
    """
    df = pd.read_csv(raw_path, parse_dates=["timestamp"])
    df.dropna(subset=["close"], inplace=True)
    df = df[df["close"] > 0]                   # remove bad ticks
    df.sort_values(["symbol", "timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"Saved clean prices → {out_path} ({len(df):,} rows, {df['symbol'].nunique()} symbols)")
    return df


# ── Technical indicator helpers ────────────────────────────────────────────────

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index — 0–100 overbought/oversold oscillator."""
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _bollinger_bands(
    series: pd.Series, window: int = 20, n_std: float = 2.0
) -> tuple:
    """
    Returns (upper, mid, lower, width) bands.
    Width = (upper - lower) / mid — normalised band width.
    """
    mid   = series.rolling(window).mean()
    std   = series.rolling(window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    width = (upper - lower) / mid.replace(0, np.nan)
    return upper, mid, lower, width


# ── Price feature engineering ──────────────────────────────────────────────────

def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical features for a single symbol's price series.
    Expects df sorted by timestamp with a 'close' column.
    """
    df   = df.copy().sort_values("timestamp")
    close = df["close"]

    # Moving averages
    df["sma_7"]  = close.rolling(7).mean()
    df["sma_14"] = close.rolling(14).mean()
    df["sma_30"] = close.rolling(30).mean()
    df["ema_7"]  = close.ewm(span=7,  adjust=False).mean()
    df["ema_14"] = close.ewm(span=14, adjust=False).mean()
    df["ema_30"] = close.ewm(span=30, adjust=False).mean()

    # Momentum (price - price N days ago)
    df["momentum_7"]  = close - close.shift(7)
    df["momentum_14"] = close - close.shift(14)
    df["momentum_30"] = close - close.shift(30)

    # Volatility
    df["volatility_7"]  = close.rolling(7).std()
    df["volatility_30"] = close.rolling(30).std()

    # Returns
    df["returns"]     = close.pct_change()
    df["log_returns"] = np.log(close / close.shift(1))

    # RSI
    df["rsi_14"] = _rsi(close, period=14)

    # Bollinger Bands (20-day)
    bb_upper, bb_mid, bb_lower, bb_width = _bollinger_bands(close)
    df["bb_upper"] = bb_upper
    df["bb_mid"]   = bb_mid
    df["bb_lower"] = bb_lower
    df["bb_width"] = bb_width

    return df


def build_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply add_price_features per symbol via groupby.
    Keeps all original OHLCV columns alongside the engineered features.
    """
    df = df.sort_values(["symbol", "timestamp"])
    return pd.concat(
        [add_price_features(grp) for _, grp in df.groupby("symbol", sort=False)],
        ignore_index=True,
    )


# ── Financial ratio engineering ────────────────────────────────────────────────

def add_financial_ratios(balance: pd.DataFrame, income: pd.DataFrame) -> pd.DataFrame:
    """
    Merge balance sheet + income statement and derive ratio features.

    Ratios:
      debt_ratio    = Total Liabilities / Total Assets
      current_ratio = Total Current Assets / Total Current Liabilities
      profit_margin = Net Income / Total Revenue
      roa           = Net Income / Total Assets
      roe           = Net Income / Total Equity
    """
    df = balance.merge(income, on=["Date", "Company Symbol"], suffixes=("_bs", "_is"))

    def _safe_div(num: str, den: str) -> pd.Series:
        return df[num] / df[den].replace(0, np.nan)

    df["debt_ratio"]    = _safe_div("Total Liabilities", "Total Assets")
    df["current_ratio"] = _safe_div("Total Current Assets", "Total Current Liabilities")
    df["profit_margin"] = _safe_div("Net Income", "Total Revenue")
    df["roa"]           = _safe_div("Net Income", "Total Assets")
    df["roe"]           = _safe_div("Net Income", "Total Equity")

    return df


# ── Pipeline stages ────────────────────────────────────────────────────────────

def run_cleaning() -> dict:
    """
    Stage 1: raw → data/processed/

    Loads the most recent raw files, cleans them, saves to data/processed/.
    Returns paths of all cleaned output files.
    """
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    outputs = {}

    # Prices
    try:
        raw_prices = _latest_file(RAW_DATA_PATH, "prices_*.csv")
        logger.info(f"Cleaning prices: {os.path.basename(raw_prices)}")
        clean_prices(
            raw_prices,
            os.path.join(PROCESSED_DATA_PATH, "clean_prices.csv"),
        )
        outputs["prices"] = os.path.join(PROCESSED_DATA_PATH, "clean_prices.csv")
    except FileNotFoundError as e:
        logger.warning(f"Price file not found — skipping: {e}")

    # Balance sheet
    try:
        raw_bs = _latest_file(RAW_DATA_PATH, "balanceSheetTable_*.csv")
        logger.info(f"Cleaning balance sheet: {os.path.basename(raw_bs)}")
        clean_financial(
            raw_bs,
            os.path.join(PROCESSED_DATA_PATH, "clean_balance_sheet.csv"),
        )
        outputs["balance_sheet"] = os.path.join(PROCESSED_DATA_PATH, "clean_balance_sheet.csv")
    except FileNotFoundError as e:
        logger.warning(f"Balance sheet not found — skipping: {e}")

    # Income statement
    try:
        raw_is = _latest_file(RAW_DATA_PATH, "incomeStatementTable_*.csv")
        logger.info(f"Cleaning income statement: {os.path.basename(raw_is)}")
        clean_financial(
            raw_is,
            os.path.join(PROCESSED_DATA_PATH, "clean_income_statement.csv"),
        )
        outputs["income_statement"] = os.path.join(PROCESSED_DATA_PATH, "clean_income_statement.csv")
    except FileNotFoundError as e:
        logger.warning(f"Income statement not found — skipping: {e}")

    return outputs


def run_feature_engineering() -> dict:
    """
    Stage 2: data/processed/ → data/features/

    Reads cleaned data, engineers all features, saves to data/features/.
    Returns paths of all feature output files.
    """
    os.makedirs(FEATURES_DATA_PATH, exist_ok=True)
    outputs = {}

    # ── Price features ─────────────────────────────────────────────────────────
    price_src = os.path.join(PROCESSED_DATA_PATH, "clean_prices.csv")
    if os.path.exists(price_src):
        logger.info("Building price features...")
        price_df      = pd.read_csv(price_src, parse_dates=["timestamp"])
        price_features = build_price_features(price_df)

        out = os.path.join(FEATURES_DATA_PATH, "price_features.csv")
        price_features.to_csv(out, index=False)
        logger.info(
            f"Price features saved → {out} "
            f"({len(price_features):,} rows, {price_features['symbol'].nunique()} symbols, "
            f"{len(price_features.columns)} columns)"
        )
        outputs["price_features"] = out
    else:
        logger.warning("clean_prices.csv not found — skipping price features")

    # ── Financial ratios ───────────────────────────────────────────────────────
    bs_src = os.path.join(PROCESSED_DATA_PATH, "clean_balance_sheet.csv")
    is_src = os.path.join(PROCESSED_DATA_PATH, "clean_income_statement.csv")

    if os.path.exists(bs_src) and os.path.exists(is_src):
        logger.info("Building financial ratios...")
        balance_df = pd.read_csv(bs_src)
        income_df  = pd.read_csv(is_src)
        ratios     = add_financial_ratios(balance_df, income_df)

        out = os.path.join(FEATURES_DATA_PATH, "financial_ratios.csv")
        ratios.to_csv(out, index=False)
        logger.info(f"Financial ratios saved → {out} ({len(ratios):,} rows)")
        outputs["financial_ratios"] = out
    else:
        logger.warning("Cleaned financial files not found — skipping ratios")

    return outputs


def run_pipeline() -> dict:
    """
    Full pipeline: clean raw data, engineer features, snapshot with DVC.
    Called by Airflow task_feature_engineering.
    """
    logger.info("=== Stage 1: Cleaning ===")
    cleaned = run_cleaning()

    logger.info("=== Stage 2: Feature Engineering ===")
    features = run_feature_engineering()

    logger.info("=== Stage 3: DVC Snapshot ===")
    try:
        from src.versioning.dvc_ops import snapshot_processed_data, snapshot_features
        snapshot_processed_data()
        snapshot_features()
    except Exception as e:
        logger.warning(f"DVC snapshot skipped: {e}")

    return {**cleaned, **features}


if __name__ == "__main__":
    run_pipeline()
