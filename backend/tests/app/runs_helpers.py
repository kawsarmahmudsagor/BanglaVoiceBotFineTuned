from sqlalchemy import select

from app.models import Dataset, DatasetVersion, Run
from app.core.contract import (
    EvalResults, ExportArtifact, ExportManifest, RunDir, write_eval_results, write_export_manifest,
)


def spec(name="r1", train="d1@v1", eval_=None, task="llm-sft", commercial=True):
    return {
        "name": name, "task": task, "backend": "fake", "base_model": "m",
        "base_model_meta": {"license": "mit", "commercial_ok": commercial},
        "datasets": {"train": train, **({"eval": eval_} if eval_ else {})},
        "method": {"type": "none"}, "hparams": {"steps": 5}, "export": ["dummy"],
    }


async def seed_dataset(session, name="d1", task="llm-sft", commercial=True, hash_="h1"):
    ds = Dataset(name=name, task=task)
    session.add(ds)
    await session.flush()
    session.add(DatasetVersion(dataset_id=ds.id, number=1, hash=hash_, path="p", n_samples=1,
                               provenance={"commercial_ok": commercial}))
    await session.commit()


async def make_run(client, session, name="r1", ds_commercial=True, **kw) -> str:
    if (await session.execute(select(Dataset).where(Dataset.name == "d1"))).scalar_one_or_none() is None:
        await seed_dataset(session, commercial=ds_commercial)
    r = await client.post("/api/recipes", json={"spec": spec(name=name, **kw)})
    assert r.status_code == 201, r.text
    r = await client.post("/api/runs", json={"recipe": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]


async def set_state(session_factory, rid, **fields):
    async with session_factory() as s:
        run = await s.get(Run, rid)
        for k, v in fields.items():
            setattr(run, k, v)
        await s.commit()


def finish_files(run_dir, metrics=None):
    rd = RunDir(run_dir)
    rd.ensure()
    write_eval_results(rd, EvalResults(metrics=metrics or {"wer": 0.2}))
    (rd.export_dir / "model.bin").write_bytes(b"weights")
    write_export_manifest(rd, ExportManifest(artifacts=[
        ExportArtifact(format="dummy", path="model.bin", size_bytes=7)]))
    return rd
