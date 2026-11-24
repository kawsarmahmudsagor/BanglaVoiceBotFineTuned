"""CosyVoice converter (light: fake extractor, no torch / onnxruntime / network)."""
import json
from pathlib import Path

import numpy as np
import pytest

from app.core.prepare import PrepareError
from app.core.text import bn_normalize

from mlops_backend.prepare import cosyvoice as m
from mlops_backend.prepare.cosyvoice import (
    INSTRUCT, UPSTREAM_COMMIT, build_cosyvoice, check_cosyvoice_refs, cosyvoice_spec, write_kaldi_lists,
)


LONG = lambda p: 2.0  # noqa: E731 - probe: every clip 2 s (the tiny fixtures are 0.5 s)


def fake_extractor(path):
    return np.arange(4, dtype="float32"), [1, 2, 3]


def crow(sid, text="আমি  যাব", path="/x/a.wav", speaker="a", dur=1.0):
    return {"sample_id": sid, "audio_path": path, "text": text, "speaker": speaker, "duration_s": dur}


def test_kaldi_lists_sorted_and_normalized(tmp_path):
    rows = [crow("b2", text="আমি  যাব", path="/x/b2.wav", speaker="y"),
            crow("a1", text=" তুমি   এসো ", path="/x/a1.wav", speaker="x"),
            crow("a3", path="/x/a3.wav", speaker="x")]
    write_kaldi_lists(rows, tmp_path)
    rd = lambda n: (tmp_path / n).read_text(encoding="utf-8").splitlines()  # noqa: E731
    assert rd("wav.scp") == ["a1 /x/a1.wav", "a3 /x/a3.wav", "b2 /x/b2.wav"]
    assert rd("text")[0] == "a1 " + bn_normalize(" তুমি   এসো ", mode="tts")
    assert rd("utt2spk") == ["a1 x", "a3 x", "b2 y"]
    assert rd("spk2utt") == ["x a1 a3", "y b2"]
    assert rd("instruct")[0] == f"a1 {INSTRUCT}"


def test_upstream_commit_matches_pin():
    pin = (Path(m.__file__).parents[2] / "UPSTREAM_COMMIT").read_text().splitlines()[0].strip()
    assert UPSTREAM_COMMIT == pin


def test_spec_records_commit_revision_and_storage():
    spec = cosyvoice_spec("rev123")
    assert spec.family == "cosyvoice-parquet"
    assert spec.params["upstream_commit"] == UPSTREAM_COMMIT
    assert spec.params["model_revision"] == "rev123" and spec.params["storage"] == "source-bytes"


def _read_parquet(p):
    import pyarrow.parquet as pq

    return pq.read_table(p).to_pylist()


def test_build_references_source_and_writes_parquet(tiny_tts_ds, tmp_path):
    import soundfile as sf

    _, rows = tiny_tts_ds(n=4)
    out = tmp_path / "out"
    res = build_cosyvoice(extractor=fake_extractor, probe=LONG, shard_size=2)({"train": rows[:3], "val": rows[3:]}, out)
    assert res.counts == {"train": 3, "val": 1} and not res.skipped
    assert not (out / "wavs").exists()
    scp = (out / "train" / "wav.scp").read_text(encoding="utf-8").splitlines()
    assert scp == [f"{r['sample_id']} {Path(r['audio_path']).resolve()}" for r in rows[:3]]
    shards = (out / "train" / "data.list").read_text(encoding="utf-8").splitlines()
    assert len(shards) == 2 and all(Path(s).is_absolute() and s.endswith(".tar") for s in shards)
    recs = [r for s in shards for r in _read_parquet(s)]
    assert [r["utt"] for r in recs] == ["t0", "t1", "t2"]
    assert set(recs[0]) >= {"utt", "audio_data", "wav", "text", "spk", "utt_embedding", "spk_embedding",
                            "speech_token", "instruct"}
    assert recs[0]["speech_token"] == [1, 2, 3] and len(recs[0]["utt_embedding"]) == 4
    assert recs[0]["audio_data"] == Path(rows[0]["audio_path"]).read_bytes()  # source bytes as-is
    assert recs[0]["text"] == bn_normalize(rows[0]["text"], mode="tts") and recs[0]["instruct"] == INSTRUCT
    for n in ("utt2data.list", "spk2data.list"):
        assert (out / "train" / n).is_file()
    # 24 kHz mono references, one per speaker
    for spk in ("a", "b"):
        info = sf.info(str(out / "refs" / f"{spk}.wav"))
        assert info.samplerate == m.REF_SAMPLE_RATE and info.channels == 1
        assert (out / "refs" / f"{spk}.txt").read_text(encoding="utf-8").startswith("আমি যাব")
    assert "train/wav.scp" in res.files and "train/data.list" in res.files


def test_data_list_uses_final_dir_not_tmp(tiny_tts_ds, tmp_path):
    _, rows = tiny_tts_ds(n=2)
    out = tmp_path / "ds" / "spec.tmp-123"
    build_cosyvoice(extractor=fake_extractor, probe=LONG)({"train": rows}, out)
    lines = (out / "train" / "data.list").read_text(encoding="utf-8").splitlines()
    assert lines and all("tmp-" not in ln and str(tmp_path / "ds" / "spec") in ln for ln in lines)


def test_undecodable_audio_skipped(tiny_tts_ds, tmp_path):
    _, rows = tiny_tts_ds(n=3)
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio")
    rows[1] = {**rows[1], "audio_path": str(bad)}
    real = m.make_probe(lambda p, sr: (_ for _ in ()).throw(ValueError("x")))
    probe = lambda p: real(p) and 2.0  # noqa: E731 - real decodability probe, but long enough clips
    res = build_cosyvoice(extractor=fake_extractor, probe=probe)({"train": rows}, tmp_path / "o")
    assert res.counts["train"] == 2
    assert [s.reason for s in res.skipped] == ["undecodable audio: ValueError"]


def test_empty_text_and_long_audio_skipped(tiny_tts_ds, tmp_path):
    _, rows = tiny_tts_ds(n=3)
    rows[0] = {**rows[0], "text": "   "}
    probe = lambda p: 31.0 if str(p).endswith("t1.wav") else 2.0  # noqa: E731
    res = build_cosyvoice(extractor=fake_extractor, probe=probe)({"train": rows}, tmp_path / "o")
    assert sorted(s.reason for s in res.skipped) == ["duration 31.0 s > 30 s", "empty text"]
    assert res.counts["train"] == 1


def test_empty_split_raises(tmp_path):
    with pytest.raises(PrepareError):
        build_cosyvoice(extractor=fake_extractor, probe=lambda p: 99.0)({"train": [crow("a")]}, tmp_path / "o")


def test_stale_source_detected_on_cache_hit(tiny_tts_ds, tmp_path):
    _, rows = tiny_tts_ds(n=2)
    out = tmp_path / "out"
    res = build_cosyvoice(extractor=fake_extractor, probe=LONG)({"train": rows}, out)

    class Info:
        path = str(out)
        files = res.files

    check_cosyvoice_refs(Info)
    Path(rows[-1]["audio_path"]).unlink()
    with pytest.raises(RuntimeError, match="missing audio"):
        check_cosyvoice_refs(Info)


@pytest.mark.heavy
def test_build_writes_tokens_embeddings_and_parquet(tiny_tts_ds, tmp_path):
    """VM only: real CAM++ / speech-tokenizer ONNX from the base model snapshot."""
    import pyarrow.parquet as pq
    import torch

    from mlops_backend.cosyvoice import download_model

    model_dir = download_model("FunAudioLLM/Fun-CosyVoice3-0.5B-2512")
    _, rows = tiny_tts_ds(n=4, seconds=1.5)  # converter skips clips under 1 s
    out = tmp_path / "out"
    res = build_cosyvoice(model_dir=model_dir)({"train": rows}, out)
    assert res.counts == {"train": 4}
    d = out / "train"
    shards = [Path(p) for p in (d / "data.list").read_text().splitlines()]
    assert shards
    tab = pq.read_table(sorted(d.glob("parquet_*.tar"))[0])
    assert {"utt", "audio_data", "text", "spk", "utt_embedding", "spk_embedding", "speech_token"} <= set(tab.column_names)
    utt2emb = torch.load(d / "utt2embedding.pt")
    utt2tok = torch.load(d / "utt2speech_token.pt")
    spk2emb = torch.load(d / "spk2embedding.pt")
    assert len(utt2emb) == 4 and len(utt2tok) == 4 and set(spk2emb) == {"a", "b"}
    assert np.asarray(utt2emb["t0"]).shape == (192,)
    rows_pq = {r["utt"]: r for r in tab.to_pylist()}
    first = sorted(rows_pq)[0]
    assert rows_pq[first]["speech_token"] == utt2tok[first]
    assert len(utt2tok["t0"]) > 0 and max(utt2tok["t0"]) < 6561 + 200


def test_prepare_cosyvoice_writes_refs_and_params(run_dir, monkeypatch, tmp_path):
    import mlops_backend.cosyvoice as fam
    from app.core.contract import read_prepared_refs

    rd = run_dir("tts", "FunAudioLLM/Fun-CosyVoice3-0.5B-2512")
    cfg = __import__("app.core.contract", fromlist=["read_config"]).read_config(rd)
    monkeypatch.setattr(fam, "build_cosyvoice",
                        lambda model_dir, rep=None: build_cosyvoice(extractor=fake_extractor, probe=LONG))
    model_dir = tmp_path / "snap" / "abc123"
    model_dir.mkdir(parents=True)
    assert fam.prepare_cosyvoice(rd, cfg, None, download=lambda repo: model_dir) == 0
    refs = read_prepared_refs(rd)
    assert [r["role"] for r in refs] == ["train", "eval"]
    assert refs[0]["spec"]["params"]["model_revision"] == "abc123" and refs[0]["counts"]["train"] == 2


def test_short_clip_reason(tiny_tts_ds, tmp_path):
    _, rows = tiny_tts_ds(n=3)
    probe = lambda p: 0.5 if str(p).endswith("t0.wav") else 2.0  # noqa: E731
    res = build_cosyvoice(extractor=fake_extractor, probe=probe)({"train": rows}, tmp_path / "o")
    assert [s.reason for s in res.skipped] == ["duration 0.5 s < 1 s"] and res.counts["train"] == 2


def test_extraction_failure_skips_row_not_build(tiny_tts_ds, tmp_path):
    _, rows = tiny_tts_ds(n=3)

    def ext(path):
        if path.endswith("t1.wav"):
            raise IndexError("empty fbank")
        return fake_extractor(path)

    out = tmp_path / "o"
    res = build_cosyvoice(extractor=ext, probe=LONG)({"train": rows}, out)
    assert res.counts["train"] == 2
    assert [(s.sample_id, s.reason) for s in res.skipped] == [("t1", "undecodable audio: IndexError")]
    assert "t1" not in (out / "train" / "wav.scp").read_text(encoding="utf-8")


def test_spk_embedding_is_mean_and_dtypes(tiny_tts_ds, tmp_path):
    import pyarrow as pa

    _, rows = tiny_tts_ds(n=4)  # speakers a, b, a, b
    vals = {"t0": 1.0, "t1": 10.0, "t2": 3.0, "t3": 20.0}
    ext = lambda path: (np.full(4, vals[Path(path).stem], dtype="float32"), [5, 6])  # noqa: E731
    out = tmp_path / "o"
    build_cosyvoice(extractor=ext, probe=LONG)({"train": rows}, out)
    tab = __import__("pyarrow.parquet", fromlist=["x"]).read_table(next((out / "train").glob("parquet_*.tar")))
    assert tab.schema.field("speech_token").type == pa.list_(pa.int64())
    assert tab.schema.field("utt_embedding").type == pa.list_(pa.float32())
    assert tab.schema.field("spk_embedding").type == pa.list_(pa.float32())
    got = {r["utt"]: r["spk_embedding"][0] for r in tab.to_pylist()}
    assert got == {"t0": 2.0, "t2": 2.0, "t1": 15.0, "t3": 15.0}


def test_fallback_source_is_transcoded_to_flac_bytes():
    import io

    import soundfile as sf

    loader = lambda p, sr: np.zeros(sr, dtype="float32")  # noqa: E731
    data = m._audio_bytes({"audio_path": "/x/a.m4a", "transcode": True}, loader)
    info = sf.info(io.BytesIO(data))
    assert info.format == "FLAC" and info.samplerate == 16000 and info.channels == 1
