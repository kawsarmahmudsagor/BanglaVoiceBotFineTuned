"""CPU queue: dataset ingests run here, independent of the GPU lock."""

import logging

from app.datasets.versions import run_ingest

from .jobs import claim_next_ingest

log = logging.getLogger(__name__)


async def cpu_tick(sf, settings) -> bool:
    """Claim and run one pending ingest. Returns whether it did work."""
    async with sf() as s:
        ingest = await claim_next_ingest(s)
    if ingest is None:
        return False
    log.info("claimed ingest %s (%s)", ingest.id, ingest.dataset_name)
    async with sf() as s:
        await run_ingest(s, ingest.id, settings)
    return True
