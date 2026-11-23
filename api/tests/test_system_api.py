import sys

from mlops_api import system
from mlops_api.models import GpuLock
from runs_helpers import make_run, set_state


async def test_system_without_gpu(client, monkeypatch):
    monkeypatch.setattr("mlops_api.system.read_gpu", lambda: None)
    body = (await client.get("/api/system")).json()
    assert body["gpu"] is None and "free_gb" in body["disk"]
    assert body["lock"] == {"holder_kind": None, "holder_id": None, "acquired_at": None}
    assert body["queue"] == {"queued": 0, "running": 0}


async def test_system_with_gpu_lock_and_queue(client, session, session_factory, monkeypatch):
    gpu = system.GpuInfo(name="X", vram_used_mb=1, vram_total_mb=2, util_pct=3, temp_c=4)
    monkeypatch.setattr("mlops_api.system.read_gpu", lambda: gpu)
    a = await make_run(client, session)
    b = await make_run(client, session)
    await make_run(client, session)
    await set_state(session_factory, a, state="training")
    await set_state(session_factory, b, state="done")
    lock = await session.get(GpuLock, 1)
    lock.holder_kind, lock.holder_id = "run", a
    await session.commit()
    body = (await client.get("/api/system")).json()
    assert body["gpu"] == {"name": "X", "vram_used_mb": 1, "vram_total_mb": 2, "util_pct": 3, "temp_c": 4}
    assert body["lock"]["holder_id"] == a and body["lock"]["acquired_at"] is None
    assert body["queue"] == {"queued": 1, "running": 1}


def test_read_gpu_returns_none_on_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", None)  # import raises
    assert system.read_gpu() is None
