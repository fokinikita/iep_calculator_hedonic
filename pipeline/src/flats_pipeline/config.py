from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Where all local pipeline data lives (bronze/silver/gold, manifest, logs).
    DATA_DIR: Path = Path("D:/data_winner/flats_pipeline")
    # Folders scanned for raw `*msk_habit.offer*.csv.gz` exports (old archive, manual downloads).
    INBOX_DIRS: list[Path] = [Path("D:/data_winner/rars")]

    # Optional direct download links. Leave empty to only pick up files from INBOX_DIRS.
    PRICES_URL: str = ""
    OFFERS_URL: str = ""
    # Extra HTTP headers for the download, as JSON, e.g. {"Cookie": "session=..."}.
    DOWNLOAD_HEADERS: dict[str, str] = {}

    # A removed lot that has not come back for more than this many days counts as sold.
    SOLD_AFTER_DAYS: int = 14

    # Hard memory cap for the pipeline process, GB (0 = no cap).
    MAX_MEMORY_GB: float = 2.5
    # Cap for the modelling commands (features, CatBoost training), GB.
    MODEL_MAX_MEMORY_GB: float = 6.0
    POLARS_MAX_THREADS: int = 2
    CATBOOST_THREADS: int = 4

    # Where `export-app` writes the web app bundle (the app repo iep_calculator_hedonic, its data/).
    APP_BUNDLE_DIR: Path = PROJECT_ROOT.parent.parent / "iep_calculator_hedonic" / "data"


    # S3: the hedonic bucket, only from .env.pipeline (HEDONIC_S3_*).
    UPLOAD_TO_S3: bool = True
    S3_ENDPOINT_URL: str = Field("", validation_alias="HEDONIC_S3_ENDPOINT_URL")
    S3_ACCESS_KEY: SecretStr = Field(SecretStr(""), validation_alias="HEDONIC_S3_ACCESS_KEY")
    S3_SECRET_KEY: SecretStr = Field(SecretStr(""), validation_alias="HEDONIC_S3_SECRET_KEY")
    S3_BUCKET_NAME: str = Field("", validation_alias="HEDONIC_S3_BUCKET")
    S3_REGION: str = Field("ru-1", validation_alias="HEDONIC_S3_REGION")
    # Prefix of the pipeline's own outputs (bronze/silver/gold, modeling runs) in the bucket.
    S3_PREFIX: str = "flats_pipeline"

    # --- automatic weekly run (auto.py) ---
    # Raw baza-winner exports in the bucket: <WINNER_PREFIX>{old,new}/<name>_<date>.csv.gz
    WINNER_PREFIX: str = "cian_winner/"
    WINNER_URLS: list[str] = [
        "https://export.baza-winner.ru/art_realtor/for-analytics.msk_habit.offer.csv.gz",
        "https://export.baza-winner.ru/art_realtor/for-analytics.msk_habit.offer_pub_list.csv.gz",
    ]
    # Version of this run (models, bundle, intermediates are stored under it); default: today, Moscow.
    RUN_DATE: str = ""
    # Dated web app bundles: <BUNDLES_DIR>/<RUN_DATE>/ (the dashboard serves every bundle in it).
    BUNDLES_DIR: Path = PROJECT_ROOT.parent / "bundles"

    @property
    def run_date(self) -> str:
        if self.RUN_DATE:
            return self.RUN_DATE
        from datetime import datetime
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo("Europe/Moscow")).date().isoformat()

    @property
    def download_dir(self) -> Path:
        return self.DATA_DIR / "inbox"

    @property
    def bronze_dir(self) -> Path:
        return self.DATA_DIR / "bronze"

    @property
    def silver_dir(self) -> Path:
        return self.DATA_DIR / "silver"

    @property
    def gold_dir(self) -> Path:
        return self.DATA_DIR / "gold"

    @property
    def manifest_path(self) -> Path:
        return self.DATA_DIR / "manifest.json"

    @property
    def modeling_dir(self) -> Path:
        """Training dataset, features, models and the hedonic index."""
        return self.DATA_DIR / "modeling"

    @property
    def logs_dir(self) -> Path:
        return self.DATA_DIR / "logs"


settings = Settings()
