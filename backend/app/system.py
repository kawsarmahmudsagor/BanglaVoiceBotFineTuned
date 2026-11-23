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


@dataclass(frozen=True)
class GpuInfo:
    name: str
    vram_used_mb: int
    vram_total_mb: int
    util_pct: int
    temp_c: int


def read_gpu() -> GpuInfo | None:
    """First NVIDIA GPU via NVML; None when there is no GPU, driver or pynvml."""
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(h)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            return GpuInfo(
                name=name.decode() if isinstance(name, bytes) else str(name),
                vram_used_mb=int(mem.used // 2**20),
                vram_total_mb=int(mem.total // 2**20),
                util_pct=int(pynvml.nvmlDeviceGetUtilizationRates(h).gpu),
                temp_c=int(pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)),
            )
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return None
