from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    # The precomputed bundle written by `python -m flats_pipeline export-app`.
    APP_DATA_DIR: Path = PROJECT_ROOT / "data"
    # Yandex Maps JS API key (map + address search on the page).
    YANDEX_MAPS_API_JS_KEY: SecretStr = SecretStr("")
    # Threads for the model / nearest-sales search (one request = one small prediction).
    APP_THREADS: int = 2


settings = Settings()
