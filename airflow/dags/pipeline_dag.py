"""
Apache Airflow DAG — Full MLOps Pipeline (Step 12).

Pipeline topology
─────────────────
scrape_live_data
  └─► validate_data
        └─► version_data
              └─► clean_and_engineer_features
                    └─► train_lstm ┐
                        train_tft  ├─ (parallel)
                    train_timesfm  │
                        train_arm  ┘
                              └─► compare_models
                                    └─► register_best_model
                                          └─► deploy_to_api
                                                └─► check_drift
                                                      └─► branch_on_drift
                                                            ├─► trigger_retrain ─┐
                                                            └─►  skip_retrain   ─┤
                                                                                  └─► send_report

Schedule: 6 PM Monday–Friday (after NYSE close).

Key design decisions
────────────────────
- Training tasks run in parallel; compare_models waits for ALL_SUCCESS.
- deploy_to_api calls POST /admin/reload — non-fatal if API is down.
- branch_on_drift uses BranchPythonOperator; send_report joins via
  NONE_FAILED_MIN_ONE_SUCCESS so it always runs regardless of branch.
- trigger_retrain re-queues this same DAG (wait_for_completion=False).
- All tasks share on_failure_callback → Slack alert.
- Training tasks have execution_timeout guards (TFT gets 4 h, others less).
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.trigger_rule import TriggerRule

from src.pipelines.tasks import (
    task_scrape,
    task_validate,
    task_version_data,
    task_feature_engineering,
    task_train_lstm,
    task_train_tft,
    task_train_timesfm,
    task_train_arm,
    task_compare_models,
    task_register_model,
    task_deploy_to_api,
    task_check_drift,
    task_branch_on_drift,
    task_send_report,
    on_failure,
)

# ── DAG ───────────────────────────────────────────────────────────────────────

default_args = {
    "owner":                    "mlops",
    "depends_on_past":          False,
    "start_date":               datetime(2024, 1, 1),
    "retries":                  2,
    "retry_delay":              timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "email_on_failure":         False,
    "on_failure_callback":      on_failure,
}

dag = DAG(
    dag_id="stock_forecast_pipeline",
    default_args=default_args,
    description="End-to-end stock price forecasting MLOps pipeline",
    schedule_interval="0 18 * * 1-5",
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "stock", "forecasting"],
)

# ── Task objects ───────────────────────────────────────────────────────────────

t1_scrape = PythonOperator(
    task_id="scrape_live_data",
    python_callable=task_scrape,
    execution_timeout=timedelta(hours=2),
    dag=dag,
)

t2_validate = PythonOperator(
    task_id="validate_data",
    python_callable=task_validate,
    execution_timeout=timedelta(minutes=15),
    dag=dag,
)

t3_version = PythonOperator(
    task_id="version_data",
    python_callable=task_version_data,
    provide_context=True,
    execution_timeout=timedelta(minutes=10),
    dag=dag,
)

t4_features = PythonOperator(
    task_id="clean_and_engineer_features",
    python_callable=task_feature_engineering,
    provide_context=True,
    execution_timeout=timedelta(hours=1),
    dag=dag,
)

t5a_lstm = PythonOperator(
    task_id="train_lstm",
    python_callable=task_train_lstm,
    provide_context=True,
    execution_timeout=timedelta(hours=3),
    dag=dag,
)

t5b_tft = PythonOperator(
    task_id="train_tft",
    python_callable=task_train_tft,
    provide_context=True,
    execution_timeout=timedelta(hours=4),
    dag=dag,
)

t5c_timesfm = PythonOperator(
    task_id="train_timesfm",
    python_callable=task_train_timesfm,
    provide_context=True,
    execution_timeout=timedelta(hours=2),
    dag=dag,
)

t5d_arm = PythonOperator(
    task_id="train_arm",
    python_callable=task_train_arm,
    provide_context=True,
    execution_timeout=timedelta(hours=1),
    dag=dag,
)

t6_compare = PythonOperator(
    task_id="compare_models",
    python_callable=task_compare_models,
    provide_context=True,
    trigger_rule=TriggerRule.ALL_SUCCESS,
    execution_timeout=timedelta(hours=1),
    dag=dag,
)

t7_register = PythonOperator(
    task_id="register_best_model",
    python_callable=task_register_model,
    provide_context=True,
    execution_timeout=timedelta(minutes=30),
    dag=dag,
)

t8_deploy = PythonOperator(
    task_id="deploy_to_api",
    python_callable=task_deploy_to_api,
    provide_context=True,
    execution_timeout=timedelta(minutes=10),
    retries=3,
    dag=dag,
)

t9_drift = PythonOperator(
    task_id="check_drift",
    python_callable=task_check_drift,
    provide_context=True,
    execution_timeout=timedelta(minutes=30),
    dag=dag,
)

t10_branch = BranchPythonOperator(
    task_id="branch_on_drift",
    python_callable=task_branch_on_drift,
    provide_context=True,
    dag=dag,
)

t11_retrain = TriggerDagRunOperator(
    task_id="trigger_retrain",
    trigger_dag_id="stock_forecast_pipeline",
    wait_for_completion=False,
    dag=dag,
)

t11_skip = EmptyOperator(
    task_id="skip_retrain",
    dag=dag,
)

t12_report = PythonOperator(
    task_id="send_report",
    python_callable=task_send_report,
    provide_context=True,
    trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    execution_timeout=timedelta(minutes=10),
    dag=dag,
)

# ── Dependencies ───────────────────────────────────────────────────────────────

t1_scrape >> t2_validate >> t3_version >> t4_features

t4_features >> [t5a_lstm, t5b_tft, t5c_timesfm, t5d_arm]

[t5a_lstm, t5b_tft, t5c_timesfm, t5d_arm] >> t6_compare

t6_compare >> t7_register >> t8_deploy >> t9_drift >> t10_branch

t10_branch >> [t11_retrain, t11_skip]

[t11_retrain, t11_skip] >> t12_report
