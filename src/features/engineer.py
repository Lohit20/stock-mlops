"""
Feature Engineering.

What this file does:
- Takes raw cleaned data
- Creates new useful features for ML models
- Saves final feature dataset to data/features/

New features we add:
- Price momentum (how fast price is moving)
- Moving averages (smoothed price trends)
- Volatility (how much price jumps around)
- Financial ratios (from balance sheet + income statement)
"""

import os
import pandas as pd
import numpy as np
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

PROCESSED_DATA_PATH = os.getenv("PROCESSED_DATA_PATH", "data/processed")
FEATURES_DATA_PATH  = os.getenv("FEATURES_DATA_PATH",  "data/features")


# ─── Cleaning helpers (from your original notebooks) ──────────────────────────

def transpose_df(df: pd.DataFrame, dates: list) -> pd.DataFrame:
    """
    Reshape financial data from long format to wide format.
    Original notebook logic — kept exactly the same.
    """
    transposed_dfs = []
    for date in dates:
        d1 = df.pivot(index='Company Symbol', columns='Period Ending:', values=date)
        d1.reset_index(inplace=True)
        d1.rename_axis(columns={'Period Ending:': ''}, inplace=True)
        d1.reset_index(drop=True, inplace=True)
        d1.insert(0, "Date", date)
        d1 = d1.iloc[1:].reset_index(drop=True)
        transposed_dfs.append(d1)
    result = pd.concat(transposed_dfs, axis=0, ignore_index=True)
    return result.rename(columns={result.columns[1]: "Company Symbol"})


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean financial dataframe.
    Original notebook logic — kept exactly the same.
    """
    df = df.dropna(axis=1, how='all')
    df = df.replace('--', np.nan)
    subset = df.iloc[:, 2:]
    subset = subset.replace(r'[\$,]', '', regex=True).apply(pd.to_numeric, errors='coerce')
    df.iloc[:, 2:] = subset
    df = df.fillna(df.groupby('Date').transform('mean'))
    return df


# ─── New features for ML models ───────────────────────────────────────────────

def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical price features.

    What each feature means:
    - SMA_7:    Average price over last 7 days (smooths out noise)
    - SMA_30:   Average price over last 30 days (longer trend)
    - EMA_7:    Exponential moving average — recent days weighted more
    - momentum: How much price changed vs 7 days ago
    - volatility: How much price jumps around (standard deviation)
    - returns:  Daily % change in price
    """
    df = df.copy()
    df['sma_7']      = df['price'].rolling(window=7).mean()
    df['sma_30']     = df['price'].rolling(window=30).mean()
    df['ema_7']      = df['price'].ewm(span=7).mean()
    df['momentum']   = df['price'] - df['price'].shift(7)
    df['volatility'] = df['price'].rolling(window=7).std()
    df['returns']    = df['price'].pct_change()
    df['log_returns'] = np.log(df['price'] / df['price'].shift(1))
    return df


def add_financial_ratios(balance: pd.DataFrame, income: pd.DataFrame) -> pd.DataFrame:
    """
    Create financial ratio features from balance sheet + income statement.

    What each ratio means:
    - debt_ratio:     Total Liabilities / Total Assets (how much is borrowed)
    - current_ratio:  Current Assets / Current Liabilities (can pay short-term debts?)
    - profit_margin:  Net Income / Total Revenue (how profitable?)
    - roa:            Net Income / Total Assets (how well are assets used?)
    - roe:            Net Income / Total Equity (return for shareholders)
    """
    df = balance.merge(income, on=['Date', 'Company Symbol'], suffixes=('_bs', '_is'))

    df['debt_ratio']    = df['Total Liabilities'] / df['Total Assets'].replace(0, np.nan)
    df['current_ratio'] = df['Total Current Assets'] / df['Total Current Liabilities'].replace(0, np.nan)
    df['profit_margin'] = df['Net Income'] / df['Total Revenue'].replace(0, np.nan)
    df['roa']           = df['Net Income'] / df['Total Assets'].replace(0, np.nan)
    df['roe']           = df['Net Income'] / df['Total Equity'].replace(0, np.nan)

    return df


# ─── Main pipeline function ───────────────────────────────────────────────────

def run_feature_engineering():
    """
    Full feature engineering pipeline.
    Called by Airflow after data cleaning step.
    """
    os.makedirs(FEATURES_DATA_PATH, exist_ok=True)

    # Load cleaned data
    price_path   = os.path.join(PROCESSED_DATA_PATH, "clean_income_statement.csv")
    balance_path = os.path.join(PROCESSED_DATA_PATH, "clean_balance_sheet.csv")
    income_path  = os.path.join(PROCESSED_DATA_PATH, "clean_income_statement.csv")

    logger.info("Loading cleaned datasets...")
    price_df   = pd.read_csv(price_path, index_col='timestamp', parse_dates=True)
    balance_df = pd.read_csv(balance_path)
    income_df  = pd.read_csv(income_path)

    # Add price features
    logger.info("Adding price features...")
    price_features = add_price_features(price_df)

    # Add financial ratio features
    logger.info("Adding financial ratio features...")
    ratio_features = add_financial_ratios(balance_df, income_df)

    # Save
    price_out = os.path.join(FEATURES_DATA_PATH, "price_features.csv")
    ratio_out = os.path.join(FEATURES_DATA_PATH, "financial_ratios.csv")

    price_features.to_csv(price_out)
    ratio_features.to_csv(ratio_out, index=False)

    logger.info(f"Features saved to {FEATURES_DATA_PATH}")
    return price_out, ratio_out


if __name__ == "__main__":
    run_feature_engineering()
