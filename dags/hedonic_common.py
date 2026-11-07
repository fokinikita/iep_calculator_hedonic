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


def _env(name: str) -> str:
    """A value from the project's .env.pipeline (parsed by hand: Airflow has no python-dotenv)."""
    try:
        with open(f"{PROJECT_DIR}/.env.pipeline", encoding="utf-8") as fh:
            for line in fh:
                if line.strip().startswith(name + "="):
                    return line.split("=", 1)[1].strip()
    except OSError:
        pass
    return ""


def telegram(text: str) -> None:
    """Message to TELEGRAM_CHAT_ID (through TELEGRAM_PROXY if set). Never raises."""
    import urllib.parse
    import urllib.request

    token, chat, proxy = _env("TELEGRAM_BOT_TOKEN"), _env("TELEGRAM_CHAT_ID"), _env("TELEGRAM_PROXY")
    if not token or not chat:
        return
    handlers = [urllib.request.ProxyHandler({"http": proxy, "https": proxy})] if proxy else []
    data = urllib.parse.urlencode({"chat_id": chat, "text": text, "parse_mode": "HTML"}).encode()
    try:
        urllib.request.build_opener(*handlers).open(
            f"https://api.telegram.org/bot{token}/sendMessage", data=data, timeout=20
        ).read()
    except Exception:
        pass


def notify_failure(context) -> None:
    ti = context.get("task_instance")
    reason = str(context.get("exception") or "task failed")[:300]
    telegram(f"❌ <b>hedonic_pipeline failed</b>\n{getattr(ti, 'task_id', '?')}\n<code>{reason}</code>")


def notify_success(context) -> None:
    telegram(f"✅ <b>hedonic_pipeline</b>: new model and dashboard deployed ({context.get('ds')})")


DEFAULT_ARGS["on_failure_callback"] = notify_failure
