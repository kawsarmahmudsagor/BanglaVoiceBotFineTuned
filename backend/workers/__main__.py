"""Worker entry point: `python -m workers`."""

import asyncio
import logging
import signal
import sys

from app.db import make_engine, make_sessionmaker
from app.settings import get_settings

from .executor import NullForwarder, execute_run
from .ingest_job import cpu_tick
from .jobs import claim_next_run, recover
from .lock import ensure_lock_row
from .tracking import make_forwarder

log = logging.getLogger("workers")


async def gpu_tick(sf, settings, forwarder_factory) -> bool:
    """Claim and execute one queued run. Returns whether it did work."""
    async with sf() as s:
        run = await claim_next_run(s)
    if run is None:
        return False
    log.info("claimed run %s", run.id)
    try:
        forwarder = forwarder_factory(settings)
    except Exception:
        # Tracking is best-effort; the lock is already held, so never let this block training.
        log.exception("forwarder factory failed for run %s; continuing without tracking", run.id)
        forwarder = NullForwarder()
    state = await execute_run(sf, run.id, settings, forwarder)
    log.info("run %s finished: %s", run.id, state)
    return True


async def loop_forever(tick, interval_s: float, name: str) -> None:
    """Call `tick` repeatedly; sleep when idle or after a failed tick. Only cancellation ends it."""
    while True:
        try:
            worked = await tick()
        except Exception:
            log.exception("%s tick failed", name)
            worked = False
        if not worked:
            await asyncio.sleep(interval_s)


async def main(settings, forwarder_factory=make_forwarder, install_signals: bool = True) -> None:
    engine = make_engine(settings.database_url)
    sf = make_sessionmaker(engine)
    try:
        async with sf() as s:
            await ensure_lock_row(s)
        async with sf() as s:
            recovered = await recover(s, settings.cancel_grace_s)
        if recovered:
            log.warning("recovered orphaned runs: %s", ", ".join(recovered))

        if install_signals and sys.platform != "win32":
            loop = asyncio.get_running_loop()
            current = asyncio.current_task()
            stopping = False

            def _stop() -> None:
                nonlocal stopping
                if not stopping:  # a second signal must not interrupt the cleanup writes
                    stopping = True
                    current.cancel()

            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):  # SIGHUP: SSH session dropped
                loop.add_signal_handler(sig, _stop)

        log.info("worker started")
        loops = [
            asyncio.create_task(
                loop_forever(lambda: gpu_tick(sf, settings, forwarder_factory), settings.poll_interval_s, "gpu")
            ),
            asyncio.create_task(loop_forever(lambda: cpu_tick(sf, settings), settings.poll_interval_s, "cpu")),
        ]
        try:
            await asyncio.gather(*loops)
        finally:
            # gather() cancels its children but does not wait for them (and they must not be cancelled
            # a second time mid-cleanup): wait so an in-flight run can record
            # "worker interrupted" and release the lock before the engine is disposed.
            await asyncio.gather(*loops, return_exceptions=True)
    finally:
        await engine.dispose()


def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        asyncio.run(main(get_settings()))
    except (KeyboardInterrupt, asyncio.CancelledError):
        log.info("worker stopped")


if __name__ == "__main__":
    run()
