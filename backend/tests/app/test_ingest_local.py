import io
import os
import shutil
import stat
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.datasets.ingest import ImportPathError, ingest, ingest_dir, resolve_import_path
from app.datasets.versions import run_ingest




class Audio:
    @staticmethod
    def sine(seconds, freq=220.0, sr=16000) -> bytes:
        t = np.arange(int(seconds * sr)) / sr
        buf = io.BytesIO()
        sf.write(buf, (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32), sr, format="WAV", subtype="PCM_16")
        return buf.getvalue()


PROV = {"source": "test", "license": "cc-by", "consent": "yes", "commercial_ok": False, "notes": ""}
META = "file,text,split\na.wav,এক,train\nb.wav,দুই,val\n"


def body(path, name="local-ds", task="asr"):
    return {"path": path, "name": name, "task": task, "eval_only": False, "provenance": PROV}


def make_dataset(root: Path, rel="ds1") -> Path:
    d = root / rel
    d.mkdir(parents=True)
    (d / "a.wav").write_bytes(Audio.sine(2.0))
    (d / "b.wav").write_bytes(Audio.sine(2.0, freq=300))
    (d / "metadata.csv").write_text(META, encoding="utf-8", newline="")
    return d


def snapshot(d: Path):
    return {p.relative_to(d).as_posix(): (p.stat().st_mode, p.read_bytes())
            for p in sorted(d.rglob("*")) if p.is_file()}


@pytest.fixture
def import_root(settings):
    settings.import_root.mkdir(parents=True)
    return settings.import_root


async def run(client, session_factory, settings, b):
    r = await client.post("/api/datasets/ingest-local", json=b)
    assert r.status_code == 202, r.text
    iid = r.json()["ingest_id"]
    async with session_factory() as s:
        await run_ingest(s, iid, settings)
    return (await client.get(f"/api/ingests/{iid}")).json()


def version_files(settings, name="local-ds"):
    vdirs = [p for p in (settings.datasets_dir / name).iterdir() if p.is_dir() and p.name != ".staging"]
    assert len(vdirs) == 1
    return sorted((vdirs[0] / "files").rglob("*.wav"))


async def test_ingest_local_ok(client, session_factory, settings, import_root):
    make_dataset(import_root)
    ing = await run(client, session_factory, settings, body("ds1"))
    assert ing["state"] == "done", ing
    v = (await client.get("/api/datasets/local-ds/versions/1")).json()
    assert v["split_source"] == "dataset" and v["n_samples"] == 2


async def test_ingest_local_absolute_path(client, session_factory, settings, import_root):
    d = make_dataset(import_root)
    assert (await run(client, session_factory, settings, body(str(d))))["state"] == "done"


async def test_ingest_local_rejects_outside_root(client, import_root, tmp_path):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    assert (await client.post("/api/datasets/ingest-local", json=body(str(outside)))).status_code == 403
    assert (await client.post("/api/datasets/ingest-local", json=body(str(tmp_path / "nope")))).status_code == 403


async def test_ingest_local_rejects_dotdot(client, import_root, tmp_path):
    (tmp_path / "secret").mkdir()
    assert (await client.post("/api/datasets/ingest-local", json=body("../secret"))).status_code == 403
    assert (await client.post("/api/datasets/ingest-local", json=body("../missing"))).status_code == 403


async def test_ingest_local_rejects_symlink_escape(client, import_root, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        os.symlink(outside, import_root / "link", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("cannot create symlinks here")
    assert (await client.post("/api/datasets/ingest-local", json=body("link"))).status_code == 403


async def test_ingest_local_missing_is_404_and_file_is_403(client, import_root):
    assert (await client.post("/api/datasets/ingest-local", json=body("nope"))).status_code == 404
    (import_root / "f.txt").write_text("x")
    assert (await client.post("/api/datasets/ingest-local", json=body("f.txt"))).status_code == 403


async def test_ingest_local_task_mismatch_409(client, session_factory, settings, import_root):
    make_dataset(import_root)
    await run(client, session_factory, settings, body("ds1"))
    assert (await client.post("/api/datasets/ingest-local", json=body("ds1", task="tts"))).status_code == 409


async def test_ingest_local_hardlinks_same_fs(client, session_factory, settings, import_root):
    d = make_dataset(import_root)
    before = snapshot(d)
    assert (await run(client, session_factory, settings, body("ds1")))["state"] == "done"
    files = version_files(settings)
    assert len(files) == 2
    if sys.platform != "win32":
        for f in files:
            assert os.stat(f).st_ino == os.stat(d / f.name).st_ino
    else:
        for f in files:
            assert f.read_bytes() == (d / f.name).read_bytes()
    after = snapshot(d)
    assert set(after) == set(before)
    assert {k: v[1] for k, v in after.items()} == {k: v[1] for k, v in before.items()}  # content unchanged
    if sys.platform != "win32":
        # hard-linked source files share the version's read-only mode, by design
        for name in ("a.wav", "b.wav"):
            assert stat.S_IMODE(after[name][0]) == 0o444
        assert stat.S_IMODE(after["metadata.csv"][0]) == stat.S_IMODE(before["metadata.csv"][0])
    assert sorted(p.name for p in d.iterdir()) == ["a.wav", "b.wav", "metadata.csv"]


async def test_ingest_local_files_read_only(client, session_factory, settings, import_root):
    make_dataset(import_root)
    await run(client, session_factory, settings, body("ds1"))
    files = version_files(settings)
    assert files
    for f in files:
        assert not f.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)


async def test_ingest_local_copy_fallback(client, session_factory, settings, import_root, monkeypatch):
    d = make_dataset(import_root)

    def boom(*a, **k):
        raise OSError(18, "Invalid cross-device link")

    monkeypatch.setattr(os, "link", boom)
    assert (await run(client, session_factory, settings, body("ds1")))["state"] == "done"
    for f in version_files(settings):
        assert f.read_bytes() == (d / f.name).read_bytes()
        assert not os.path.samestat(os.stat(f), os.stat(d / f.name)) if hasattr(os.path, "samestat") else True


async def test_import_dirs_lists_dataset_dirs(client, import_root):
    make_dataset(import_root, "a")
    make_dataset(import_root, "grp/b")
    (import_root / "grp" / "b" / "curation.json").write_text('{"task": "tts"}')
    make_dataset(import_root, "x/y/z")  # three levels deep: not listed
    (import_root / "empty").mkdir()
    items = (await client.get("/api/datasets/import-dirs")).json()["items"]
    assert items == [
        {"path": "a", "task_hint": None, "has_curation": False},
        {"path": "grp/b", "task_hint": "tts", "has_curation": True},
    ]


def test_ingest_dir_same_hash_as_zip_upload(tmp_path):
    d = make_dataset(tmp_path, "src")
    zp = tmp_path / "up"
    shutil.make_archive(str(zp), "zip", d)
    a = ingest_dir(d, "asr", tmp_path / "out_dir")
    b = ingest(tmp_path / "up.zip", "asr", tmp_path / "out_zip")
    assert a.report.ok and b.report.ok
    assert a.version_hash == b.version_hash and a.n_samples == 2


def test_ingest_dir_disk_check_counts_copies_only(tmp_path, monkeypatch):
    d = make_dataset(tmp_path, "src")
    monkeypatch.setattr(os, "link", lambda *a, **k: (_ for _ in ()).throw(OSError(18, "x")))
    import app.datasets.ingest as mod

    real = os.stat

    class Fake:  # pretend the source is on another device
        def __init__(self, st):
            self._st = st
            self.st_dev = st.st_dev + 1

        def __getattr__(self, n):
            return getattr(self._st, n)

    monkeypatch.setattr(mod.os, "stat", lambda p, *a, **k: Fake(real(p, *a, **k)) if Path(p) == d else real(p, *a, **k))
    r = ingest_dir(d, "asr", tmp_path / "out", disk_min_gb=10**9)
    assert not r.report.ok and r.report.errors[0].code == "insufficient_disk"


def test_resolve_import_path_unit(tmp_path):
    root = tmp_path / "root"
    (root / "d").mkdir(parents=True)
    assert resolve_import_path("d", root) == (root / "d").resolve()
    with pytest.raises(ImportPathError):
        resolve_import_path("..", root)


async def test_ingest_local_nul_byte_path_is_403(client, import_root):
    r = await client.post("/api/datasets/ingest-local", json=body("ds\x001"))
    assert r.status_code == 403
    assert "ds" not in r.json()["detail"]


def test_resolve_generic_message(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    for raw in ("../secret", str(tmp_path / "nope"), "a\x00b"):
        with pytest.raises(ImportPathError) as e:
            resolve_import_path(raw, root)
        assert str(e.value) == "path is not allowed"


async def test_import_dirs_skips_unreadable(client, import_root, monkeypatch):
    make_dataset(import_root, "a")
    (import_root / "locked").mkdir()
    real = Path.iterdir

    def iterdir(self):
        if self.name == "locked":
            raise PermissionError(13, "denied")
        return real(self)

    monkeypatch.setattr(Path, "iterdir", iterdir)
    r = await client.get("/api/datasets/import-dirs")
    assert r.status_code == 200
    assert [i["path"] for i in r.json()["items"]] == ["a"]


def test_ingest_dir_rejects_file_symlink_escape(tmp_path):
    d = make_dataset(tmp_path, "src")
    outside = tmp_path / "outside.wav"
    outside.write_bytes(Audio.sine(2.0, freq=500))
    (d / "b.wav").unlink()
    try:
        os.symlink(outside, d / "b.wav")
    except (OSError, NotImplementedError):
        pytest.skip("cannot create symlinks here")
    r = ingest_dir(d, "asr", tmp_path / "out")
    assert not r.report.ok and r.version_hash is None
    assert any(e.code == "unsafe_path" and "b.wav" in e.message for e in r.report.errors)
