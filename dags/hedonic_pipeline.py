"""Weekly hedonic pipeline: winner exports -> dataset -> model -> indices -> dashboard.

Every Wednesday 15:00 Moscow (baza-winner refreshes the exports after 14:00):

  winner_fetch    both exports -> s3://<bucket>/cian_winner/new/<name>_<date>.csv.gz
  sync_raw        s3 cian_winner/{old,new}/ -> server inbox (missing files only)
  ingest_build    raw -> bronze -> silver -> gold sold flats (+ S3 flats_pipeline/...)
  dataset         sold flats -> cleaned training dataset
  features        dataset -> features (iep variant)
  hedonic_index   boosting with month dummies + linear model, basket indices, area / room indices
  evaluate        out-of-time test (last 3 months) -> quality shown in the bundle manifest
  export_bundle   web app bundle -> bundles/<RUN_DATE>/
  publish         modeling outputs, models/<RUN_DATE>/model.cbm, bundles/<RUN_DATE>/ -> S3
  deploy          restart the new dashboard (app_v2, port 8011): it serves every bundle (old and new
                  model side by side); the current dashboard (app, port 8010) is not touched

Install: symlink this file and hedonic_common.py into the Airflow dags folder.
"""

from datetime import timedelta

import pendulum
from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator

from hedonic_common import DEFAULT_ARGS, MOSCOW, PROJECT_DIR, pipeline

STEPS = [
    ("winner_fetch", "winner-fetch", 1),
    ("sync_raw", "sync-raw", 1),
    ("ingest_build", "run", 4),
    ("dataset", "dataset", 2),
    ("features", "features", 3),
    ("hedonic_index", "hedonic-index", 4),
    ("evaluate", "evaluate", 3),
    ("export_bundle", "export-bundle", 2),
    ("publish", "publish", 1),
]

DEPLOY = f"""
set -e
cd {PROJECT_DIR}
docker compose up -d --build app_v2
docker compose restart app_v2
for i in $(seq 1 60); do
  docker compose ps app_v2 | grep -q "(healthy)" && break; sleep 5
done
docker compose ps app_v2 | grep -q "(healthy)"
curl -fsS http://127.0.0.1:8011/models
"""

with DAG(
    dag_id="hedonic_pipeline",
    description="Weekly winner exports -> dataset -> hedonic model -> indices -> dashboard",
    default_args=DEFAULT_ARGS,
    start_date=pendulum.datetime(2026, 11, 4, tz=MOSCOW),
    schedule="0 15 * * 3",  # Wednesday 15:00 Moscow
    catchup=False,
    max_active_runs=1,
    tags=["hedonic", "winner", "model"],
) as dag:
    tasks = [
        BashOperator(task_id=task_id, bash_command=pipeline(cmd), execution_timeout=timedelta(hours=hours),
                     **({"retries": 2, "retry_delay": timedelta(minutes=15)} if task_id == "winner_fetch" else {}))
        for task_id, cmd, hours in STEPS
    ]
    tasks.append(BashOperator(task_id="deploy", bash_command=DEPLOY, execution_timeout=timedelta(minutes=15)))
    for a, b in zip(tasks, tasks[1:]):
        a >> b
