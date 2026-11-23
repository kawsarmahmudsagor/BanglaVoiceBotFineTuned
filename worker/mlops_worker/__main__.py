"""Worker entry point: `python -m mlops_worker`."""

import asyncio
import logging
import signal
import sys

from mlops_api.db import make_engine, make_sessionmaker
from mlops_api.settings import get_settings

from .executor import Forwarder, NullForwarder, execute_run
from .ingest_job import cpu_tick
from .jobs import claim_next_run, recover
from .lock import ensure_lock_row

log = logging.getLogger("mlops_worker")


def default_forwarder_factory(settings) -> Forwarder:
    return NullForwarder()


async def gpu_tick(sf, settings, forwarder_factory) -> bool:
    """Claim and execute one queued run. Returns whether it did work."""
    async with sf() as s:
        run = await claim_next_run(s)
    if run is None:
        return False
    log.info("claimed run %s", run.id)
    state = await execute_run(sf, run.id, settings, forwarder_factory(settings))
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


async def main(settings, forwarder_factory=default_forwarder_factory, install_signals: bool = True) -> None:
    engine = make_engine(settings.database_url)
    sf = make_sessionmaker(engine)
    try:
        async with sf() as s:
            await ensure_lock_row(s)
        async with sf() as s:
            recovered = await recover(s)
        if recovered:
            log.warning("recovered orphaned runs: %s", ", ".join(recovered))

        if install_signals and sys.platform != "win32":
            loop = asyncio.get_running_loop()
            current = asyncio.current_task()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, current.cancel)

        log.info("worker started")
        await asyncio.gather(
            loop_forever(lambda: gpu_tick(sf, settings, forwarder_factory), settings.poll_interval_s, "gpu"),
            loop_forever(lambda: cpu_tick(sf, settings), settings.poll_interval_s, "cpu"),
        )
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
