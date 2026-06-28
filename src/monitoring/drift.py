"""
Model & Data Drift Detection using Evidently AI.

What is drift?
- Stock market patterns change over time
- A model trained in 2020 may not work well in 2024
- Drift = when current data looks different from training data

What this file does:
- Compares new incoming data to the original training data
- If they look too different → triggers retraining
- Generates HTML reports you can open in a browser
"""

import os
import pandas as pd
import numpy as np
from loguru import logger
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset
from evidently.metrics import DatasetDriftMetric, ColumnDriftMetric
from dotenv import load_dotenv

load_dotenv()

FEATURES_PATH = os.getenv("FEATURES_DATA_PATH", "data/features")
REPORTS_PATH  = "monitoring/reports"
DRIFT_THRESHOLD = 0.3  # If drift score > 0.3, trigger retraining


def load_reference_data() -> pd.DataFrame:
    """
    Load training data as the reference (baseline).
    Evidently compares new data against this.
    """
    path = os.path.join(FEATURES_PATH, "price_features.csv")
    df = pd.read_csv(path, index_col='timestamp', parse_dates=True)

    # Use first 80% as reference (training period)
    split = int(len(df) * 0.8)
    return df.iloc[:split][['price', 'sma_7', 'sma_30', 'returns', 'volatility']].dropna()


def load_current_data() -> pd.DataFrame:
    """
    Load most recent data as the current (production) data.
    Evidently checks if this has drifted from reference.
    """
    path = os.path.join(FEATURES_PATH, "price_features.csv")
    df = pd.read_csv(path, index_col='timestamp', parse_dates=True)

    # Use last 20% as current data
    split = int(len(df) * 0.8)
    return df.iloc[split:][['price', 'sma_7', 'sma_30', 'returns', 'volatility']].dropna()


def run_data_drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """
    Generate data drift report.

    Evidently compares distributions of each feature.
    If price distribution shifted → market regime changed.
    If volatility distribution shifted → market got more/less volatile.

    Returns:
        Dictionary with drift results and whether retraining is needed
    """
    os.makedirs(REPORTS_PATH, exist_ok=True)

    # Create drift report
    # DataDriftPreset automatically checks all columns for drift
    report = Report(metrics=[
        DatasetDriftMetric(),
        ColumnDriftMetric(column_name="price"),
        ColumnDriftMetric(column_name="returns"),
        ColumnDriftMetric(column_name="volatility"),
    ])

    report.run(reference_data=reference, current_data=current)

    # Save HTML report (open in browser to see visual charts)
    report_path = os.path.join(REPORTS_PATH, "data_drift_report.html")
    report.save_html(report_path)
    logger.info(f"Drift report saved to {report_path}")

    # Extract drift score
    result = report.as_dict()
    drift_detected = result['metrics'][0]['result']['dataset_drift']
    drift_score    = result['metrics'][0]['result']['share_of_drifted_columns']

    logger.info(f"Drift Score: {drift_score:.3f} | Drift Detected: {drift_detected}")

    return {
        "drift_detected":   drift_detected,
        "drift_score":      drift_score,
        "needs_retraining": drift_score > DRIFT_THRESHOLD,
        "report_path":      report_path,
    }


def run_prediction_drift_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str
) -> dict:
    """
    Check if model predictions are drifting from actual values.

    Args:
        y_true: Actual stock prices
        y_pred: Model predicted prices
        model_name: Name of the model being monitored
    """
    os.makedirs(REPORTS_PATH, exist_ok=True)

    # Create DataFrame for Evidently
    df = pd.DataFrame({
        "target":     y_true,
        "prediction": y_pred,
    })

    # Split into reference (first half) and current (second half)
    split = len(df) // 2
    reference = df.iloc[:split]
    current   = df.iloc[split:]

    report = Report(metrics=[RegressionPreset()])
    report.run(reference_data=reference, current_data=current)

    report_path = os.path.join(REPORTS_PATH, f"prediction_drift_{model_name}.html")
    report.save_html(report_path)
    logger.info(f"Prediction drift report: {report_path}")

    return {"report_path": report_path}


def check_and_alert() -> bool:
    """
    Main monitoring function — called daily by Airflow.

    Returns:
        True if retraining is needed, False otherwise
    """
    logger.info("Running drift detection...")

    reference = load_reference_data()
    current   = load_current_data()

    result = run_data_drift_report(reference, current)

    if result["needs_retraining"]:
        logger.warning(
            f"⚠️  DRIFT DETECTED (score={result['drift_score']:.3f}) "
            f"— Triggering retraining pipeline"
        )
        return True

    logger.info(f"✅ No significant drift (score={result['drift_score']:.3f})")
    return False


if __name__ == "__main__":
    needs_retrain = check_and_alert()
    print(f"Retraining needed: {needs_retrain}")
