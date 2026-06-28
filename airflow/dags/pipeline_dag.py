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
from loguru import logger
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
    from src.ingestion.scraper import run_scraper, run_price_fetcher
    run_price_fetcher()   # fast — yfinance daily prices for all symbols
    run_scraper()         # Selenium — quarterly financials (balance sheet + income stmt)


def task_validate():
    import os
    from src.ingestion.validator import run_validation
    passed = run_validation(os.getenv("RAW_DATA_PATH", "data/raw"))
    if not passed:
        raise ValueError("Data validation failed — check logs and alerts")


def task_version_data(**context):
    """DVC snapshot of raw data — links git commit to data hash."""
    from src.versioning.dvc_ops import snapshot_raw_data
    info = snapshot_raw_data()
    # Push hash to XCom so training tasks can log it to MLflow
    context["ti"].xcom_push(key="raw_data_hash",   value=info["hash"])
    context["ti"].xcom_push(key="raw_data_commit",  value=info["commit"])


def task_feature_engineering(**context):
    from src.features.engineer import run_pipeline
    outputs = run_pipeline()   # clean → feature engineer → DVC snapshot
    context["ti"].xcom_push(key="feature_outputs", value=outputs)


def _get_data_tags(context: dict) -> dict:
    """Pull DVC hash + commit from XCom (set by task_version_data)."""
    ti = context.get("ti")
    if ti is None:
        return {}
    return {
        "data_hash":   ti.xcom_pull(task_ids="version_data", key="raw_data_hash"),
        "data_commit": ti.xcom_pull(task_ids="version_data", key="raw_data_commit"),
    }


def task_train_lstm(**context):
    from src.training.train_lstm import train
    train(**_get_data_tags(context))


def task_train_tft(**context):
    from src.training.train_tft import train
    train(**_get_data_tags(context))


def task_train_timesfm(**context):
    from src.training.train_timesfm import train
    train(**_get_data_tags(context))


def task_train_arm(**context):
    from src.training.train_arm import train
    train(**_get_data_tags(context))


def task_compare_models(**context):
    from src.evaluation.compare_models import run_comparison
    best = run_comparison()
    # Push winner to XCom so task_register_model can pick it up
    context["ti"].xcom_push(key="best_model_type", value=best.get("model_type"))
    context["ti"].xcom_push(key="best_run_id",     value=best.get("run_id"))
    context["ti"].xcom_push(key="best_val_loss",   value=best.get("val_loss"))


def task_register_model(**context):
    """Step 9 — promote winner through Staging → Production with validation."""
    from src.registry.model_registry import register_best_model
    ti = context["ti"]
    best_info = {
        "model_type": ti.xcom_pull(task_ids="compare_models", key="best_model_type"),
        "run_id":     ti.xcom_pull(task_ids="compare_models", key="best_run_id"),
        "val_loss":   ti.xcom_pull(task_ids="compare_models", key="best_val_loss"),
    }
    result = register_best_model(best_info)
    logger.info(
        f"Registry step complete: {result['model_type']} v{result['version']} "
        f"→ {result['stage']} (promoted={result['promoted']})"
    )
    context["ti"].xcom_push(key="registry_result", value=result)


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
    provide_context=True,
    dag=dag,
)

t4_features = PythonOperator(
    task_id="clean_and_engineer_features",
    python_callable=task_feature_engineering,
    provide_context=True,
    dag=dag,
)

# All 4 models train in PARALLEL (saves time)
t5a_lstm = PythonOperator(
    task_id="train_lstm",
    python_callable=task_train_lstm,
    provide_context=True,
    dag=dag,
)

t5b_tft = PythonOperator(
    task_id="train_tft",
    python_callable=task_train_tft,
    provide_context=True,
    dag=dag,
)

t5c_timesfm = PythonOperator(
    task_id="train_timesfm",
    python_callable=task_train_timesfm,
    provide_context=True,
    dag=dag,
)

t5d_arm = PythonOperator(
    task_id="train_arm",
    python_callable=task_train_arm,
    provide_context=True,
    dag=dag,
)

# Compare only after ALL 4 models finish
t6_compare = PythonOperator(
    task_id="compare_models",
    python_callable=task_compare_models,
    provide_context=True,
    trigger_rule=TriggerRule.ALL_SUCCESS,
    dag=dag,
)

# Step 9 — promote winner through Staging → Production
t6b_register = PythonOperator(
    task_id="register_best_model",
    python_callable=task_register_model,
    provide_context=True,
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
t6_compare >> t6b_register >> t7_drift >> t8_report
