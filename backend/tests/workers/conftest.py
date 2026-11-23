import datetime as dt
import itertools

from sqlalchemy import select

from app.models import Ingest, Recipe, RecipeVersion, Run, Upload

_seq = itertools.count(1)
_BASE = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)


async def make_run(session, state="queued", created_at=None, run_id=None) -> Run:
    n = next(_seq)
    recipe = Recipe(name=f"recipe-{n}")
    session.add(recipe)
    await session.flush()
    rv = RecipeVersion(recipe_id=recipe.id, number=1, spec={"name": f"recipe-{n}"}, yaml="name: x\n")
    session.add(rv)
    await session.flush()
    run = Run(
        id=run_id or f"r-{n:08x}",
        recipe_version_id=rv.id,
        state=state,
        run_dir=f"/runs/{n}",
        created_at=created_at or _BASE + dt.timedelta(minutes=n),
    )
    session.add(run)
    await session.commit()
    return run


async def get_run(session, run_id) -> Run:
    return (await session.execute(select(Run).where(Run.id == run_id).execution_options(populate_existing=True))).scalar_one()


async def make_ingest(session, state="pending", created_at=None) -> Ingest:
    n = next(_seq)
    up = Upload(id=f"u-{n}", filename="a.zip", length=1, state="complete")
    session.add(up)
    await session.flush()
    ing = Ingest(
        upload_id=up.id,
        dataset_name=f"ds-{n}",
        task="asr",
        state=state,
        created_at=created_at or _BASE + dt.timedelta(minutes=n),
    )
    session.add(ing)
    await session.commit()
    return ing


async def get_ingest(session, ingest_id) -> Ingest:
    return (await session.execute(select(Ingest).where(Ingest.id == ingest_id).execution_options(populate_existing=True))).scalar_one()
