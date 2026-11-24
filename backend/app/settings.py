import os
from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/settings.py -> repo root; the worker and API read `<repo>/.env` regardless of cwd.
# In the API image the code lives at /app/app, so this resolves to `/`: there is no /.env and
# compose sets every path (DATA_ROOT, BACKENDS_ROOT, RECIPES_DIR, CATALOG_DIR) explicitly.
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
    ingest_workers: int = max(1, (os.cpu_count() or 2) // 2)
    cancel_grace_s: float = 60
    poll_interval_s: float = 2
    hf_token: SecretStr | None = None  # env HF_TOKEN; handed to backend subprocesses, never logged

    import_root: Path | None = None  # filled by the validator below; curated datasets on the server; default <data_root>/import

    @model_validator(mode="after")
    def _default_import_root(self) -> "Settings":
        if self.import_root is None:
            self.import_root = self.data_root / "import"
        return self

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
    def hf_home(self) -> Path:
        return self.data_root / "models" / "hf"

    @property
    def runs_dir(self) -> Path:
        return self.data_root / "runs"


@lru_cache
def get_settings() -> Settings:
    return Settings()
