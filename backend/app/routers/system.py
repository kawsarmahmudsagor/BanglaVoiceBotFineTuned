from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Depends, Request
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from .. import system
from ..db import get_session
from ..models import GpuLock, Run
from .datasets import iso

router = APIRouter(prefix="/api")

RUNNING_STATES = ("preparing", "training", "evaluating", "exporting")


async def _count(session: AsyncSession, states: tuple[str, ...]) -> int:
    q = select(func.count()).select_from(Run).where(Run.state.in_(states))
    return (await session.execute(q)).scalar_one()


@router.get("/system")
async def get_system(request: Request, session: AsyncSession = Depends(get_session)) -> dict:
    gpu = system.read_gpu()
    disk = system.disk_status(request.app.state.settings)
    lock = await session.get(GpuLock, 1)
    return {
        "gpu": asdict(gpu) if gpu else None,
        "lock": {
            "holder_kind": lock.holder_kind if lock else None,
            "holder_id": lock.holder_id if lock else None,
            "acquired_at": iso(lock.acquired_at) if lock else None,
        },
        "disk": {"free_gb": disk.free_gb, "warn": disk.warn, "block": disk.block},
        "queue": {"queued": await _count(session, ("queued",)), "running": await _count(session, RUNNING_STATES)},
    }
