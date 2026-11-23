from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/settings.py -> repo root; the worker and API read `<repo>/.env` regardless of cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    # Precedence: init kwargs > environment variables > .env file. Unknown .env keys
    # (POSTGRES_USER, REPO_ROOT, ...) belong to compose/scripts and are ignored.
    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "sqlite+aiosqlite:///./mlops.db"
    data_root: Path = Path("./var")
    backends_root: Path = REPO_ROOT / "backend" / "workers" / "backends"
    recipes_dir: Path = Path("./recipes")
    catalog_dir: Path = REPO_ROOT / "catalog"
    web_origin: str = "http://localhost:3000"
    mlflow_tracking_uri: str | None = None
    disk_warn_gb: float = 20
    disk_min_gb: float = 10
    cancel_grace_s: float = 60
    poll_interval_s: float = 2

    @property
    def incoming_dir(self) -> Path:
        return self.data_root / "data" / "incoming"

    @property
    def datasets_dir(self) -> Path:
        return self.data_root / "data" / "datasets"

    @property
    def prepared_dir(self) -> Path:
        return self.data_root / "data" / "prepared"

    @property
    def runs_dir(self) -> Path:
        return self.data_root / "runs"


@lru_cache
def get_settings() -> Settings:
    return Settings()
