import csv
import json

import numpy as np
import pytest
import soundfile as sf

from ytds.exporter import UnreviewedTranslated, export_dataset, exportable
from ytds.runner import vocals_path
from ytds.store import Store

SR = 44100


@pytest.fixture
def env(tmp_path):
    store = Store(tmp_path / "ytds.db")
    ds = store.create_dataset("ds", "Speaker One")
    store.add_video(ds, "vid1", "http://x/vid1")
    store.update_video(ds, "vid1", title="T", channel="C", upload_date="20240101", license="cc")
    rng = np.random.default_rng(0)
    t = np.arange(SR * 30) / SR
    y = (0.3 * np.sin(2 * np.pi * 220 * t) + 0.02 * rng.standard_normal(len(t))).astype("float32")
    p = vocals_path(tmp_path, ds, "vid1")
    p.parent.mkdir(parents=True)
    sf.write(p, y, SR)
    return store, ds, tmp_path


def _add(store, ds, specs):
    clips = []
    for i, (status, src, text) in enumerate(specs):
        clips.append({"id": f"vid1_{i:04d}", "video_id": "vid1", "idx": i, "start_s": 1.0 + i * 5,
                      "end_s": 4.0 + i * 5, "status": status, "text_final": text, "text_source": src,
                      "no_punct": 0})
    store.replace_clips(ds, "vid1", clips)


def _run(env, out="out", **kw):
    store, ds, tmp = env
    calls = []
    s = export_dataset(store, ds, tmp / out, lambda d, t: calls.append((d, t)), data_dir=tmp, **kw)
    return s, calls, tmp / out


def test_export_layout(env):
    store, ds, _ = env
    _add(store, ds, [("auto", "stt", "এক"), ("approved", "stt", "দুই"), ("auto", "stt", "তিন")])
    s, calls, out = _run(env)
    with open(out / "metadata.csv", encoding="utf-8", newline="") as fh:
        assert fh.readline().rstrip("\n") == "file,text,speaker,split"
        fh.seek(0)
        rows = list(csv.DictReader(fh))
    assert len(rows) == 3 and {r["speaker"] for r in rows} == {"Speaker One"}
    for r in rows:
        info = sf.info(out / r["file"])
        assert (info.samplerate, info.channels, info.subtype) == (24000, 1, "PCM_16")
    assert calls == [(1, 3), (2, 3), (3, 3)]
    assert not out.with_suffix(".tmp").exists()


def test_refuses_unreviewed_translated(env):
    store, ds, _ = env
    _add(store, ds, [("auto", "stt", "এক"), ("needs_review", "translated", "দুই")])
    with pytest.raises(UnreviewedTranslated) as e:
        _run(env)
    assert e.value.count == 1
    assert [c["id"] for c in exportable(store, ds)[0]] == ["vid1_0000"]
    s, _, out = _run(env, skip_unreviewed_translated=True)
    assert s["clips"] == 1 and s["unreviewed_translated_skipped"] == 1
    with open(out / "metadata.csv", encoding="utf-8", newline="") as fh:
        assert [r["text"] for r in csv.DictReader(fh)] == ["এক"]
    curation = json.loads((out / "curation.json").read_text(encoding="utf-8"))
    assert curation["counts"]["unreviewed_translated_skipped"] == 1


def test_flagged_unapproved_excluded(env):
    store, ds, _ = env
    _add(store, ds, [("auto", "stt", "এক"), ("flagged", "stt", "দুই"), ("rejected", "stt", "তিন"),
                     ("dropped", "stt", "চার"), ("approved", "stt", "")])
    clips, n = exportable(store, ds)
    assert [c["id"] for c in clips] == ["vid1_0000"] and n == 0
    s, _, _ = _run(env)
    assert s["clips"] == 1 and s["empty_text_excluded"] == 1


def test_curation_has_provenance_and_commercial_false(env):
    store, ds, _ = env
    _add(store, ds, [("auto", "stt", "এক"), ("approved", "translated", "দুই"), ("dropped", "stt", "x")])
    store.update_clip(ds, "vid1_0002", drop_reason="dnsmos")
    store.update_clip(ds, "vid1_0001", no_punct=1)
    _, _, out = _run(env)
    cur = json.loads((out / "curation.json").read_text(encoding="utf-8"))
    assert cur["commercial_use"] is False and cur["tool"] == "youtube-tts-dataset"
    assert cur["git_commit"] and cur["exported_at"] and cur["thresholds"]["min_bn_ratio"] == 0.5
    assert cur["counts"]["total_clips"] == 2 and cur["counts"]["no_punct"] == 1
    assert cur["counts"]["text_source"] == {"stt": 1, "translated": 1}
    v = cur["videos"][0]
    assert v["url"] == "http://x/vid1" and v["license"] == "cc" and v["clips_kept"] == 2
    assert v["seconds_dropped"] == {"dnsmos": 3.0}
    assert cur["validation_dropped"] == 0


def test_validation_failure_drops_row_and_wav(env, monkeypatch):
    from app.datasets import validate_audio as va

    store, ds, _ = env
    _add(store, ds, [("auto", "stt", "এক"), ("auto", "stt", "দুই"), ("auto", "stt", "তিন")])
    real = va.validate_audio

    def corrupting(root, task, **kw):
        (root / "wavs" / "vid1_0001.wav").write_bytes(b"not audio")
        return real(root, task, **kw)

    monkeypatch.setattr(va, "validate_audio", corrupting)
    s, _, out = _run(env)
    with open(out / "metadata.csv", encoding="utf-8", newline="") as fh:
        files = [r["file"] for r in csv.DictReader(fh)]
    assert files == ["wavs/vid1_0000.wav", "wavs/vid1_0002.wav"]
    assert not (out / "wavs" / "vid1_0001.wav").exists()
    assert s["validation_dropped"] == 1
    cur = json.loads((out / "curation.json").read_text(encoding="utf-8"))
    assert cur["validation_dropped"] == 1 and cur["counts"]["total_clips"] == 2


def test_whole_dataset_validation_error_raises(env, monkeypatch):
    from app.datasets import validate_audio as va
    from app.datasets.report import Issue, Report

    store, ds, _ = env
    _add(store, ds, [("auto", "stt", "এক")])
    rep = Report()
    rep.add_error(Issue("missing_column", "metadata.csv has no 'speaker' column"))
    monkeypatch.setattr(va, "validate_audio", lambda root, task, **kw: (rep, []))
    with pytest.raises(RuntimeError):
        _run(env)


def test_zero_rows_aborts_and_keeps_previous_export(env):
    store, ds, tmp = env
    _add(store, ds, [("auto", "stt", "এক")])
    _, _, out = _run(env)
    before = (out / "metadata.csv").read_bytes()
    vocals_path(tmp, ds, "vid1").unlink()
    with pytest.raises(Exception):
        _run(env)
    assert (out / "metadata.csv").read_bytes() == before
    assert not (tmp / "out.tmp").exists()


def test_curation_models_lists_recorded_model_ids(env):
    from ytds.config import Settings

    store, ds, tmp = env
    _add(store, ds, [("auto", "stt", "এক")])
    store.update_video(ds, "vid1", model_a="qwen3-asr-bn-sayedshaun", model_b="whisper-bn-ft1")
    store.add_video(ds, "vid2", "http://x/vid2")  # nothing exported from it
    store.update_video(ds, "vid2", model_a="qwen3-asr-bn-ft1", model_b="whisper-bn-mozilla")
    _, _, out = _run(env, settings=Settings(data_dir=tmp, model_b_path="B"))
    models = json.loads((out / "curation.json").read_text(encoding="utf-8"))["models"]
    assert "fallback" not in models
    assert models["model_a"] == [{"id": "qwen3-asr-bn-sayedshaun", "source": "SayedShaun/Qwen3-ASR-0.6B-Bengali"}]
    assert models["model_b"] == [{"id": "whisper-bn-ft1", "source": "B"}]
    assert models["separation"] == "htdemucs" and models["vad"] == "silero-vad"
