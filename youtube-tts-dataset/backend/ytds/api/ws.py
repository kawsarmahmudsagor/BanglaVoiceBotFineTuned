"""WebSocket: streams a dataset's bus events to the UI."""
from __future__ import annotations

import asyncio
import contextlib

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()


@router.websocket("/ws/datasets/{dataset_id}")
async def dataset_events(websocket: WebSocket, dataset_id: str) -> None:
    bus = websocket.app.state.bus
    queue = bus.subscribe(dataset_id)  # before accept: events published after the handshake are buffered
    sender: asyncio.Task | None = None

    async def pump() -> None:
        while True:
            await websocket.send_json(await queue.get())

    try:
        await websocket.accept()
        sender = asyncio.create_task(pump())
        while True:
            await websocket.receive_text()  # only to notice the disconnect
    except WebSocketDisconnect:
        pass
    finally:
        bus.unsubscribe(dataset_id, queue)
        if sender is not None:
            sender.cancel()
            with contextlib.suppress(BaseException):
                await sender
