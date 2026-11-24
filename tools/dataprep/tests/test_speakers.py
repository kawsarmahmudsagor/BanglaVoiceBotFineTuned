from __future__ import annotations

import numpy as np
import pandas as pd
import soundfile as sf
from test_select import make_wd, prof

from dataprep.__main__ import main
from dataprep.select import select
from dataprep.steps import speakers as sp
from dataprep.workdir import StepFile, WorkDir


def _blobs(sizes, dim=192, noise=0.02, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.linalg.qr(rng.normal(size=(dim, len(sizes))))[0].T  # orthogonal unit centers
    x = np.concatenate([c + noise * rng.normal(size=(n, dim)) for c, n in zip(centers, sizes)])
    truth = np.repeat(np.arange(len(sizes)), sizes)
    return x / np.linalg.norm(x, axis=1, keepdims=True), truth


def _same_partition(a, b):
    return len(set(zip(a, b))) == len(set(a)) == len(set(b))


def test_cluster_finds_three_gaussian_clusters():
    x, truth = _blobs([20, 15, 10])
    labels = sp.cluster(x, 0.35)
    assert len(set(labels)) == 3 and _same_partition(labels, truth)


def test_two_stage_matches_direct_on_small_case():
    x, truth = _blobs([60, 40, 30])
    labels = sp.cluster_two_stage(x, 0.35, max_direct=50)  # forces k-means centroids (k = 13)
    assert _same_partition(labels, truth)


def test_name_clusters_by_size_then_smallest_id():
    ids = ["e", "a", "b", "c", "d", "f"]
    labels = np.array([5, 7, 7, 9, 9, 2])  # sizes: 7->2 (a,b), 9->2 (c,d), 5->1 (e), 2->1 (f)
    assert sp.name_clusters(ids, labels) == ["bai-3", "bai-1", "bai-1", "bai-2", "bai-2", "bai-4"]
    assert sp.name_clusters(ids, labels, prefix="voice")[1] == "voice-1"


def _write_emb(wd, ids, x, errors):
    df = pd.DataFrame({"id": ids, "emb": [r.astype("float32").tolist() for r in x], "error": errors})
    StepFile.write(wd.step_path("speaker_emb"), df.astype({"error": "object"}), step="speaker_emb", version=1, params={"model": sp.MODEL})


def test_cluster_step_writes_named_speakers(tmp_path):
    x, _ = _blobs([6, 3, 1])
    ids = [f"r{i}" for i in range(10)]
    wd = WorkDir(tmp_path)
    _write_emb(wd, ids + ["bad"], np.vstack([x, x[:1]]), [None] * 10 + ["boom"])
    assert sp.cluster_step(wd, threshold=0.35) == 3
    df = StepFile.read(wd.step_path("speakers"), step="speakers", version=1, params={"threshold": 0.35})
    assert list(df.columns) == ["id", "speaker", "error"] and "bad" not in set(df["id"])
    by = df.set_index("id")["speaker"]
    assert set(by[[f"r{i}" for i in range(6)]]) == {"bai-1"}
    assert set(by[["r6", "r7", "r8"]]) == {"bai-2"} and by["r9"] == "bai-3"


def test_min_speaker_minutes_drops_small_cluster(tmp_path):
    x, _ = _blobs([6, 4])
    wd = make_wd(tmp_path)  # r0..r9, dur 5.0 s each
    _write_emb(wd, [f"r{i}" for i in range(10)], x, [None] * 10)
    sp.cluster_step(wd)
    res = select(wd, prof(min_speaker_minutes=0.45))  # bai-1 = 30 s kept, bai-2 = 20 s dropped (floor 27 s)
    assert set(res.rows["speaker"]) == {"bai-1"} and len(res.rows) == 6


def _wav(path, sr=16000):
    sf.write(path, (0.3 * np.sin(np.arange(sr) / 10)).astype("float32"), sr)


def _point_clips_at_wav(wd, tmp_path):
    p = tmp_path / "a.wav"
    _wav(p)
    clips = pd.read_parquet(wd.clips_path)
    clips["src_path"] = str(p)
    StepFile.write(wd.clips_path, clips, step="import", version=1, params={})


def test_speakers_command_end_to_end_and_resume(tmp_path, monkeypatch):
    wd = make_wd(tmp_path / "work", n=6)
    _point_clips_at_wav(wd, tmp_path)
    x, _ = _blobs([4, 2])
    calls: list[int] = []

    def fake(y):
        calls.append(len(y))
        return x[len(calls) - 1]

    monkeypatch.setattr(sp, "_embedder", fake)
    args = ["speakers", "--work", str(wd.root)]
    assert main(args) == 0 and len(calls) == 6
    df = pd.read_parquet(wd.step_path("speakers"))
    assert df["speaker"].nunique() == 2 and df["error"].isna().all()
    assert main([*args, "--prefix", "voice"]) == 0 and len(calls) == 6  # embeddings resumed, clustering redone
    assert set(pd.read_parquet(wd.step_path("speakers"))["speaker"]) == {"voice-1", "voice-2"}
    StepFile.read(wd.step_path("speakers"), step="speakers", version=1, params={"threshold": 0.35})


def test_missing_extra_fails_fast(tmp_path, monkeypatch):
    wd = make_wd(tmp_path)
    monkeypatch.setattr(sp, "_embedder", None)

    def boom():
        raise ImportError("no speechbrain")

    monkeypatch.setattr(sp, "make_embedder", boom)
    assert main(["speakers", "--work", str(wd.root)]) == 1
    assert not wd.step_path("speaker_emb").exists()


def test_only_profile_limits_embedded_clips(tmp_path, monkeypatch):
    wd = make_wd(tmp_path / "work", {0: {"dur": 0.2}, 1: {"dur": 0.2}}, n=4)
    _point_clips_at_wav(wd, tmp_path)
    rng = np.random.default_rng(1)
    seen: list[int] = []
    monkeypatch.setattr(sp, "_embedder", lambda y: seen.append(1) or rng.normal(size=192))
    assert main(["speakers", "--work", str(wd.root), "--only-profile", "tts-strict"]) == 0
    emb = pd.read_parquet(wd.step_path("speaker_emb"))
    assert set(emb["id"]) == {"r2", "r3"} and len(seen) == 2
