"""
Bootstrap real data into the pipeline without needing Selenium or ChromeDriver.

What this does:
  1. Creates data/raw/nasdaq_screener.csv  (100 well-known symbols)
  2. Downloads 2 years of daily prices via yfinance  → data/raw/prices_*.csv
  3. Cleans raw data                                 → data/processed/
  4. Engineers features                              → data/features/

Run once to seed the pipeline:
    python scripts/bootstrap_data.py

After this, the full DAG (training, serving, monitoring) can run end-to-end.
Financial statement data (balance sheet / income statement) requires Selenium
and a live NASDAQ connection — that's handled by run_scraper() in the Airflow DAG.
"""

import os
import sys
import pandas as pd
from pathlib import Path
from loguru import logger

# Make sure src/ is importable when running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ── 1. Symbol list ────────────────────────────────────────────────────────────

S_AND_P_100 = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK-B", "UNH", "XOM",
    "LLY", "JPM", "JNJ", "V",    "PG",   "MA",   "AVGO", "HD",   "CVX",  "MRK",
    "ABBV", "KO",   "PEP",  "COST", "ADBE", "WMT",  "MCD",  "CRM",  "CSCO", "BAC",
    "ACN",  "LIN",  "TMO",  "ABT",  "NFLX", "NKE",  "TXN",  "DIS",  "PM",   "CMCSA",
    "DHR",  "WFC",  "NEE",  "RTX",  "BMY",  "ORCL", "INTC", "QCOM", "HON",  "UNP",
    "AMD",  "AMGN", "LOW",  "IBM",  "CAT",  "SPGI", "SBUX", "MDT",  "INTU", "PFE",
    "AXP",  "DE",   "GE",   "GILD", "CVS",  "ISRG", "ADI",  "MU",   "BKNG", "VRTX",
    "REGN", "SYK",  "ZTS",  "MDLZ", "CI",   "TJX",  "MMC",  "SO",   "DUK",  "PLD",
    "BDX",  "EOG",  "AON",  "ITW",  "EL",   "PGR",  "ADP",  "NSC",  "FDX",  "USB",
    "EMR",  "GD",   "MCO",  "EW",   "STZ",  "COP",  "SLB",  "HUM",  "ICE",  "ETN",
]


def create_screener_csv(symbols: list, out_path: str):
    """Write a minimal nasdaq_screener.csv that the pipeline expects."""
    df = pd.DataFrame({"Symbol": symbols, "Name": symbols})
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"Screener CSV written → {out_path} ({len(symbols)} symbols)")


def download_prices(symbols: list, period: str = "2y") -> str:
    """Download price data using the pipeline's own fetcher."""
    from src.ingestion.scraper import fetch_prices
    logger.info(f"Downloading {period} of price data for {len(symbols)} symbols...")
    return fetch_prices(symbols, period=period)


def run_cleaning():
    from src.features.engineer import run_cleaning as _clean
    logger.info("Cleaning raw data...")
    return _clean()


def run_features():
    from src.features.engineer import run_feature_engineering as _features
    logger.info("Engineering features...")
    return _features()


def print_summary(raw_data_path: str, processed_path: str, features_path: str):
    """Print what was created in each directory."""
    print("\n" + "=" * 60)
    print("Bootstrap complete — data directory summary")
    print("=" * 60)

    for label, path in [
        ("data/raw/",       raw_data_path),
        ("data/processed/", processed_path),
        ("data/features/",  features_path),
    ]:
        files = list(Path(path).glob("*.csv"))
        print(f"\n{label}")
        if files:
            for f in sorted(files):
                size_kb = f.stat().st_size // 1024
                try:
                    rows = sum(1 for _ in open(f)) - 1
                    print(f"  ✅ {f.name:<45} {rows:>7,} rows  {size_kb:>5} KB")
                except Exception:
                    print(f"  ✅ {f.name}")
        else:
            print("  (empty)")

    # Show feature column list
    price_feat_path = Path(features_path) / "price_features.csv"
    if price_feat_path.exists():
        cols = pd.read_csv(price_feat_path, nrows=0).columns.tolist()
        print(f"\nPrice feature columns ({len(cols)} total):")
        print(f"  {', '.join(cols)}")

    print("\nNext steps:")
    print("  • docker-compose up   → start MLflow, Airflow, Grafana")
    print("  • python -m src.training.train_lstm  → train first model")
    print("  • uvicorn src.serving.api:app        → start prediction API")
    print("=" * 60)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="Bootstrap pipeline data")
    parser.add_argument("--symbols",  nargs="+", default=None,
                        help="Override symbol list (default: S&P 100)")
    parser.add_argument("--period",   default="2y",
                        help="yfinance period string e.g. '2y', '1y', '6mo'")
    parser.add_argument("--skip-features", action="store_true",
                        help="Download prices only, skip cleaning + features")
    args = parser.parse_args()

    RAW_DATA_PATH       = os.getenv("RAW_DATA_PATH",       "data/raw")
    PROCESSED_DATA_PATH = os.getenv("PROCESSED_DATA_PATH", "data/processed")
    FEATURES_DATA_PATH  = os.getenv("FEATURES_DATA_PATH",  "data/features")

    symbols = args.symbols or S_AND_P_100
    logger.info(f"Bootstrapping with {len(symbols)} symbols, period={args.period}")

    # Step 1 — screener CSV
    screener_path = os.path.join(RAW_DATA_PATH, "nasdaq_screener.csv")
    create_screener_csv(symbols, screener_path)

    # Step 2 — price download
    prices_path = download_prices(symbols, period=args.period)
    logger.info(f"Prices saved → {prices_path}")

    if not args.skip_features:
        # Step 3 — clean
        cleaned = run_cleaning()
        logger.info(f"Cleaned outputs: {list(cleaned.keys())}")

        # Step 4 — features
        features = run_features()
        logger.info(f"Feature outputs: {list(features.keys())}")

    print_summary(RAW_DATA_PATH, PROCESSED_DATA_PATH, FEATURES_DATA_PATH)
