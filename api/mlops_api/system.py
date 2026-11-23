import shutil
from dataclasses import dataclass

from .settings import Settings

GB = 1024**3


@dataclass(frozen=True)
class DiskStatus:
    free_gb: float
    warn: bool
    block: bool


def disk_status(settings: Settings) -> DiskStatus:
    """Free space on DATA_ROOT. Creates the directory if it does not exist yet."""
    settings.data_root.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(settings.data_root).free / GB
    return DiskStatus(
        free_gb=free_gb,
        warn=free_gb < settings.disk_warn_gb,
        block=free_gb < settings.disk_min_gb,
    )
