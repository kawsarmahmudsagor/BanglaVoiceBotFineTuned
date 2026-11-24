from app.core.backend_kit import mark_complete, write_init_meta
from app.core.contract import RunDir
from app.models import ModelVersion

from runs_helpers import make_run, set_state


def _ckpt(run_dir, name, step, complete=True):
    d = RunDir(run_dir).checkpoints_dir / name
    d.mkdir(parents=True)
    write_init_meta(d, backend="fake", base_model="m", method="none", format="fake", step=step)
    if complete:
        mark_complete(d)


Q = "/api/init-sources?task=llm-sft&backend=fake&base_model=m"


async def test_init_sources_filters_by_backend_and_model(client, session, settings):
    a = await make_run(client, session, name="ra")
    _ckpt(settings.runs_dir / a, "final", 5)
    _ckpt(settings.runs_dir / a, "step-2", 2, complete=False)
    data = (await client.get(Q)).json()
    assert [r["id"] for r in data["runs"]] == [a]
    assert data["runs"][0]["recipe"] == "ra" and data["runs"][0]["state"] == "queued"
    assert data["runs"][0]["finished_at"] is None
    assert [c["name"] for c in data["runs"][0]["checkpoints"]] == ["final"]
    assert data["model_versions"] == []
    for q in ("task=llm-sft&backend=other&base_model=m", "task=llm-sft&backend=fake&base_model=z",
              "task=asr&backend=fake&base_model=m"):
        assert (await client.get(f"/api/init-sources?{q}")).json()["runs"] == []
    assert (await client.get("/api/init-sources?task=llm-sft")).status_code == 422


async def test_init_sources_excludes_runs_without_complete_checkpoint(client, session, settings):
    a = await make_run(client, session, name="ra")
    b = await make_run(client, session, name="rb")
    await make_run(client, session, name="rc")
    _ckpt(settings.runs_dir / a, "step-1", 1, complete=False)
    _ckpt(settings.runs_dir / b, "final", 3)
    assert [r["id"] for r in (await client.get(Q)).json()["runs"]] == [b]


async def test_init_sources_lists_model_versions(client, session, settings):
    a = await make_run(client, session, name="ra")
    b = await make_run(client, session, name="rb")
    c = await make_run(client, session, name="rc")
    _ckpt(settings.runs_dir / a, "final", 5)
    _ckpt(settings.runs_dir / c, "final", 5, complete=False)
    session.add_all([
        ModelVersion(name="m3", number=1, run_id=c, task="llm-sft", base_model="m"),
        ModelVersion(name="m1", number=2, run_id=a, task="llm-sft", base_model="m", metrics={"wer": 0.1}),
        ModelVersion(name="m2", number=1, run_id=b, task="asr", base_model="m"),
    ])
    await session.commit()
    mvs = (await client.get(Q)).json()["model_versions"]
    assert mvs == [{"ref": "m1@v2", "run_id": a, "metrics": {"wer": 0.1}}]


async def test_init_sources_only_offers_checkpoints_resolvable_by_step(client, session, settings):
    a = await make_run(client, session, name="ra")
    _ckpt(settings.runs_dir / a, "final", 9)
    _ckpt(settings.runs_dir / a, "checkpoint-300", 300)
    _ckpt(settings.runs_dir / a, "flow-epoch-2", 5000)
    _ckpt(settings.runs_dir / a, "llm-final", 400)
    d = RunDir(settings.runs_dir / a).checkpoints_dir / "llm-epoch-3-step-200"
    d.mkdir(parents=True)
    write_init_meta(d, backend="fake", base_model="m", method="none", format="cosyvoice-component", step=200)
    mark_complete(d)
    names = [c["name"] for c in (await client.get(Q)).json()["runs"][0]["checkpoints"]]
    assert names == ["final", "checkpoint-300"]
