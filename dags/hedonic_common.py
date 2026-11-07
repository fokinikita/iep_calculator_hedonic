"""Shared bits for the hedonic pipeline DAG.

Every step runs in the project's Docker image (`docker compose run --rm pipeline <command>`), not in
Airflow's interpreter, so the scheduler's environment stays untouched. Logs: the Airflow task log
(stdout of the container), DATA_DIR/logs/pipeline_<day>.log on the server and s3 logs/<RUN_DATE>/.
"""

from datetime import timedelta

PROJECT_DIR = "/root/projects/iep_calculator_hedonic"
MOSCOW = "Europe/Moscow"  # Airflow runs with default_timezone = utc; Russia has no DST since 2014

# Version of the run (models, bundle and intermediates are stored under it): the run's date in Moscow,
# or `{"run_date": "YYYY-MM-DD"}` in the trigger conf (re-run an older version).
RUN_DATE = (
    "{{ dag_run.conf.get('run_date') if dag_run.conf and dag_run.conf.get('run_date') "
    "else (dag_run.run_after + macros.timedelta(hours=3)).strftime('%Y-%m-%d') }}"
)

DEFAULT_ARGS = {
    "owner": "hedonic",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}


def pipeline(command: str) -> str:
    """Shell command running one pipeline step in the project's container."""
    return (
        f"cd {PROJECT_DIR} && docker compose --profile pipeline run --rm -T "
        f"-e RUN_DATE={RUN_DATE} pipeline {command}"
    )
