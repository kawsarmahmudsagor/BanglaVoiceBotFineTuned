from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "sqlite+aiosqlite:///./mlops.db"
    data_root: Path = Path("./var")
    backends_root: Path = Path("./backends")
    recipes_dir: Path = Path("./recipes")
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
