import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from app.core.splits import SplitConfig, SplitRatios
from app.core.contract import DatasetRef, RunConfig, RunDir, read_prepared_refs, write_prepared_refs
from app.core.prepare import (
    BuildResult,
    PrepareError,
    PrepSpec,
    Skip,
    ensure_prepared,
    iter_rows,
    load_splits,
    prepared_dir,
)

SPEC = PrepSpec(family="f", version=1, params={"k": 1})


def _ok_build(rows, out):
    (out / "x.txt").write_text("ok")
    return BuildResult(counts={k: len(v) for k, v in rows.items()}, files=["x.txt"])


def _ds(tmp_path, name, table, h):
    root = tmp_path / name
    root.mkdir()
    pq.write_table(table, root / "manifest.parquet")
    return DatasetRef(name=name, version=1, hash=h, root=str(root),
                      manifest=str(root / "manifest.parquet"), splits=["train", "val"], split=SplitConfig())


@pytest.fixture
def audio_ds(tmp_path):
    t = pa.table({
        "sample_id": ["a", "b", "c", "d"],
        "file": ["a.wav", "b.wav", "c.wav", "d.wav"],
        "text": ["x", "y", "z", "w"],
        "duration_s": [1.0, 2.0, 3.0, 4.0],
        "sample_rate": [16000] * 4,
        "split": ["train", "train", "val", "test"],
    })
    return _ds(tmp_path, "audio", t, "audiohash0001")


@pytest.fixture
def llm_ds(tmp_path):
    msgs = json.dumps([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}])
    t = pa.table({
        "sample_id": ["a", "b", "c"],
        "messages_json": [msgs, msgs, msgs],
        "tools_json": [None, json.dumps([{"name": "t"}]), None],
        "split": ["train", "val", "test"],
    })
    return _ds(tmp_path, "llm", t, "llmhash0001")


@pytest.fixture
def cfg(tmp_path, audio_ds):
    return RunConfig(run_id="r", attempt=1, resume=False, task="asr", backend="b", base_model="m",
                     method={}, hparams={}, export=[], seed=1, datasets={"train": audio_ds},
                     prepared_root=str(tmp_path / "prepared"))


def test_spec_hash_is_stable_and_order_independent():
    a = PrepSpec(family="f", version=1, params={"b": 1, "a": "বাংলা"})
    b = PrepSpec(family="f", version=1, params={"a": "বাংলা", "b": 1})
    assert a.hash() == b.hash() and len(a.hash()) == 16


def test_iter_rows_filters_split_and_resolves_audio(audio_ds):
    rows = list(iter_rows(audio_ds, "train"))
    assert len(rows) == 2
    assert {r["split"] for r in rows} == {"train"}
    assert Path(rows[0]["audio_path"]) == Path(audio_ds.root) / "files" / rows[0]["file"]


def test_iter_rows_decodes_messages(llm_ds):
    row = next(iter_rows(llm_ds, "train"))
    assert row["messages"][0]["role"] in {"system", "user"}
    assert row["tools"] is None
    val = next(iter_rows(llm_ds, "val"))
    assert val["tools"] == [{"name": "t"}]


def test_load_splits(audio_ds):
    got = load_splits(audio_ds, ["train", "val"])
    assert {k: len(v) for k, v in got.items()} == {"train": 2, "val": 1}


def test_miss_builds_then_hit_reuses(cfg, audio_ds):
    calls = []

    def build(rows, out):
        calls.append(1)
        return _ok_build(rows, out)

    first = ensure_prepared(cfg, audio_ds, SPEC, build)
    second = ensure_prepared(cfg, audio_ds, SPEC, build)
    assert calls == [1] and first.cache_hit is False and second.cache_hit is True
    assert Path(second.path) == prepared_dir(cfg, audio_ds, SPEC)
    assert (Path(second.path) / "prepared.json").exists()
    assert second.counts == {"train": 2, "val": 1}


def test_build_exception_leaves_no_target_and_no_tmp(cfg, audio_ds):
    def boom(rows, out):
        (out / "half").write_text("x")
        raise ValueError("bad")

    with pytest.raises(ValueError):
        ensure_prepared(cfg, audio_ds, SPEC, boom)
    parent = prepared_dir(cfg, audio_ds, SPEC).parent
    assert not prepared_dir(cfg, audio_ds, SPEC).exists()
    assert not list(parent.glob("*.tmp-*"))


def test_stale_tmp_dirs_are_removed(cfg, audio_ds):
    target = prepared_dir(cfg, audio_ds, SPEC)
    stale = target.with_name(target.name + ".tmp-99999")
    stale.mkdir(parents=True)
    ensure_prepared(cfg, audio_ds, SPEC, _ok_build)
    assert not stale.exists()


def test_skip_share_above_threshold_fails_and_caches_nothing(cfg, audio_ds):
    def skip_all(rows, out):
        return BuildResult(counts={}, skipped=[Skip(sample_id=r["sample_id"], split=s, reason="too long")
                                               for s, rs in rows.items() for r in rs])

    with pytest.raises(PrepareError, match=r"100\.0% of rows skipped .*too long"):
        ensure_prepared(cfg, audio_ds, SPEC, skip_all)
    assert not prepared_dir(cfg, audio_ds, SPEC).exists()


def test_skip_share_within_threshold_is_cached(cfg, audio_ds):
    def skip_one(rows, out):
        return BuildResult(counts={"train": 1, "val": 1},
                           skipped=[Skip(sample_id="a", split="train", reason="r")])

    info = ensure_prepared(cfg, audio_ds, SPEC, skip_one, max_skip_share=0.5)
    assert len(info.skipped) == 1


def test_concurrent_builds_share_one_result(cfg, audio_ds, tmp_path):
    counter = tmp_path / "counter.txt"
    script = tmp_path / "worker.py"
    script.write_text(textwrap.dedent("""
        import sys, time, json
        from pathlib import Path
        from app.core.contract import RunConfig, DatasetRef
        from app.core.prepare import ensure_prepared, PrepSpec, BuildResult
        cfg = RunConfig.model_validate_json(Path(sys.argv[1]).read_text())
        counter = Path(sys.argv[2])
        def build(rows, out):
            with counter.open("a") as f:
                f.write("built" + chr(10))
            time.sleep(1)
            return BuildResult(counts={k: len(v) for k, v in rows.items()})
        info = ensure_prepared(cfg, cfg.datasets["train"], PrepSpec(family="f", version=1, params={"k": 1}), build)
        print(info.cache_hit)
    """))
    cfg_file = tmp_path / "cfg.json"
    cfg_file.write_text(cfg.model_dump_json())
    procs = [subprocess.Popen([sys.executable, str(script), str(cfg_file), str(counter)],
                              stdout=subprocess.PIPE, text=True) for _ in range(2)]
    hits = []
    for p in procs:
        out, _ = p.communicate(timeout=60)
        assert p.returncode == 0
        hits.append(out.strip() == "True")
    assert counter.read_text().count("built") == 1
    assert sorted(hits) == [False, True]


def test_prepared_refs_roundtrip(tmp_path):
    rd = RunDir(tmp_path / "run")
    assert read_prepared_refs(rd) == []
    refs = [{"role": "train", "spec_hash": "abc", "path": "/p", "cache_hit": False}]
    write_prepared_refs(rd, refs)
    assert rd.prepared_refs_path == rd.path / "prepared_refs.json"
    assert read_prepared_refs(rd) == refs
    rd.prepared_refs_path.write_text("{not json")
    assert read_prepared_refs(rd) == []
    rd.prepared_refs_path.write_text('{"a": 1}')
    assert read_prepared_refs(rd) == []


def test_iter_rows_applies_split_config(audio_ds):
    # dataset gives train/val/test; use_dataset_splits=False reassigns everything by hash
    ignore = audio_ds.model_copy(update={"split": SplitConfig(use_dataset_splits=False,
                                                              ratios=SplitRatios(train=0.5, val=0.5, test=0.0))})
    got = load_splits(ignore, ["train", "val"])
    assert sum(len(v) for v in got.values()) == 4
    assert got["val"] and {r["split"] for r in got["val"]} == {"val"}
    assert got["train"] and {r["split"] for r in got["train"]} == {"train"}
    # same config twice (cached) is stable
    assert [r["sample_id"] for r in iter_rows(ignore, "train")] == [r["sample_id"] for r in got["train"]]


def test_empty_assigned_split_is_prepare_error(llm_ds):
    ds = llm_ds.model_copy(update={"split": SplitConfig(use_dataset_splits=False)})  # 3 rows cannot fill 70/20/10
    with pytest.raises(PrepareError, match="split '.*' is empty: add data or change datasets.split.ratios"):
        list(iter_rows(ds, "train"))


def test_prepared_dir_differs_when_only_ratios_differ(cfg, audio_ds):
    other = audio_ds.model_copy(update={"split": SplitConfig(ratios=SplitRatios(train=0.8, val=0.1, test=0.1))})
    assert prepared_dir(cfg, audio_ds, SPEC) != prepared_dir(cfg, other, SPEC)
    assert prepared_dir(cfg, audio_ds, SPEC) != prepared_dir(cfg, audio_ds.model_copy(update={"split": None}), SPEC)
    assert prepared_dir(cfg, audio_ds, SPEC).name == ensure_prepared(cfg, audio_ds, SPEC, _ok_build).spec_hash


def test_eval_only_ref_yields_all_rows(audio_ds):
    ref = audio_ds.model_copy(update={"split": None, "splits": ["train", "val", "test"]})
    assert {s: len(list(iter_rows(ref, s))) for s in ref.splits} == {"train": 4, "val": 4, "test": 4}
