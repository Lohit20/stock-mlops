"""
Apache Airflow DAG — Full MLOps Pipeline.

What is a DAG?
DAG = Directed Acyclic Graph = a sequence of tasks with dependencies.

Think of it as a recipe:
Step 1 must finish before Step 2 starts,
Steps 5a/5b/5c can run at the same time,
Step 6 waits for ALL of Step 5 to finish.

This DAG runs every day at 6PM (after market close):
1.  scrape_live_data
2.  validate_data
3.  version_data
4.  clean_and_engineer
5a. train_lstm        ┐
5b. train_tft         ├── all run in parallel
5c. train_timesfm     │
5d. train_arm         ┘
6.  compare_models
7.  register_best_model
8.  deploy_to_api
9.  check_drift
10. send_report
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.trigger_rule import TriggerRule

# Default settings applied to every task in the DAG
default_args = {
    "owner":            "mlops",
    "depends_on_past":  False,       # Don't wait for yesterday's run to succeed
    "start_date":       datetime(2024, 1, 1),
    "retries":          2,           # Retry failed tasks twice
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": False,
}

dag = DAG(
    dag_id="stock_forecast_pipeline",
    default_args=default_args,
    description="End-to-end stock price forecasting pipeline",
    schedule_interval="0 18 * * 1-5",  # 6PM Monday–Friday (market days)
    catchup=False,                     # Don't run missed historical runs
    tags=["mlops", "stock", "forecasting"],
)


# ── Task functions ─────────────────────────────────────────────────────────────

def task_scrape():
    from src.ingestion.scraper import run_scraper
    run_scraper()


def task_validate():
    import pandas as pd
    from src.ingestion.validator import validate_balance_sheet, validate_income_statement
    import glob, os

    raw_files = glob.glob("data/raw/balanceSheetTable_*.csv")
    latest    = max(raw_files, key=os.path.getctime)
    df        = pd.read_csv(latest)
    passed, _ = validate_balance_sheet(df)
    if not passed:
        raise ValueError("Data validation failed — stopping pipeline")


def task_version_data():
    """DVC snapshot of new data."""
    import subprocess
    subprocess.run(["dvc", "add", "data/raw"], check=True)
    subprocess.run(["git", "add", "data/raw.dvc"], check=True)
    subprocess.run(["git", "commit", "-m", f"data: snapshot {datetime.now().date()}"], check=True)


def task_feature_engineering():
    from src.features.engineer import run_feature_engineering
    run_feature_engineering()


def task_train_lstm():
    from src.training.train_lstm import train
    train()


def task_train_tft():
    # Placeholder — implemented in Step 3
    print("TFT training — coming in Step 3")


def task_train_timesfm():
    # Placeholder — implemented in Step 3
    print("TimesFM training — coming in Step 3")


def task_train_arm():
    # Placeholder — implemented in Step 3
    print("ARM training — coming in Step 3")


def task_compare_models():
    from src.evaluation.compare_models import run_comparison
    run_comparison()


def task_check_drift(**context):
    from src.monitoring.drift import check_and_alert
    needs_retrain = check_and_alert()
    # Push result to XCom so next tasks can read it
    context['ti'].xcom_push(key='needs_retraining', value=needs_retrain)


# ── Define tasks ───────────────────────────────────────────────────────────────

t1_scrape = PythonOperator(
    task_id="scrape_live_data",
    python_callable=task_scrape,
    dag=dag,
)

t2_validate = PythonOperator(
    task_id="validate_data",
    python_callable=task_validate,
    dag=dag,
)

t3_version = PythonOperator(
    task_id="version_data",
    python_callable=task_version_data,
    dag=dag,
)

t4_features = PythonOperator(
    task_id="clean_and_engineer_features",
    python_callable=task_feature_engineering,
    dag=dag,
)

# All 4 models train in PARALLEL (saves time)
t5a_lstm = PythonOperator(
    task_id="train_lstm",
    python_callable=task_train_lstm,
    dag=dag,
)

t5b_tft = PythonOperator(
    task_id="train_tft",
    python_callable=task_train_tft,
    dag=dag,
)

t5c_timesfm = PythonOperator(
    task_id="train_timesfm",
    python_callable=task_train_timesfm,
    dag=dag,
)

t5d_arm = PythonOperator(
    task_id="train_arm",
    python_callable=task_train_arm,
    dag=dag,
)

# Compare only after ALL 4 models finish
t6_compare = PythonOperator(
    task_id="compare_models",
    python_callable=task_compare_models,
    trigger_rule=TriggerRule.ALL_SUCCESS,
    dag=dag,
)

t7_drift = PythonOperator(
    task_id="check_drift",
    python_callable=task_check_drift,
    provide_context=True,
    dag=dag,
)

t8_report = BashOperator(
    task_id="send_report",
    bash_command="echo 'Pipeline complete. Check MLflow at http://localhost:5000'",
    dag=dag,
)

# ── Task dependencies (defines the order) ─────────────────────────────────────
#
#  t1 → t2 → t3 → t4 → [t5a, t5b, t5c, t5d] → t6 → t7 → t8
#

t1_scrape >> t2_validate >> t3_version >> t4_features
t4_features >> [t5a_lstm, t5b_tft, t5c_timesfm, t5d_arm]
[t5a_lstm, t5b_tft, t5c_timesfm, t5d_arm] >> t6_compare
t6_compare >> t7_drift >> t8_report
