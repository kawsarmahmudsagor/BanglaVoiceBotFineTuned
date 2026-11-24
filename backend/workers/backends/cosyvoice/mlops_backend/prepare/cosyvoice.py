"""CosyVoice 3 converter. Per split `<split>/`: Kaldi lists (wav.scp text utt2spk spk2utt instruct),
utt2embedding.pt / spk2embedding.pt (CAM++), utt2speech_token.pt (speech_tokenizer_v3), parquet shards
(`parquet_<i>.tar`, upstream tools/make_parquet_list.py layout) and `data.list`; one `refs/<speaker>.wav|txt`.

Disk budget (ruling R24): no wavs/ copy. wav.scp lists the absolute SOURCE paths and each parquet shard embeds the
source file's bytes as-is in `audio_data` (upstream's processor decodes them with torchaudio.load(BytesIO)).
Only a source soundfile cannot open but the fallback decoder can (m4a, webm...) is transcoded to 16 kHz FLAC bytes.
Tokens and embeddings are computed from an in-memory 16 kHz decode, as upstream tools/extract_*.py do.

Light module: numpy, pyarrow, soundfile, torch, onnxruntime and whisper load inside functions only.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

from app.core.backend_kit.audio_refs import check_audio_refs, probed_rows, progress_ticker
from app.core.prepare import BuildResult, PrepareError, PrepSpec, Skip
from app.core.text import BN_NORMALIZE_VERSION, bn_normalize
from app.core.tts_refs import DEFAULT_SPEAKER, pick_reference

UPSTREAM_COMMIT = "074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc"  # = ../UPSTREAM_COMMIT (checked by a test)
SAMPLE_RATE = 16000        # tokenizer / embedder input rate
REF_SAMPLE_RATE = 24000    # reference clips: the frontend loads the prompt wav at 24 kHz (speech_feat) and 16 kHz
MAX_SECONDS = 30           # the speech tokenizer does not support longer audio
SHARD_SIZE = 1000          # upstream --num_utts_per_parquet default (run.sh uses 1000)
EXTRACT_THREADS = 8        # upstream --num_thread default
INSTRUCT = "You are a helpful assistant.<|endofprompt|>"  # upstream CosyVoice3 recipe: prefix of every text


class Fallback:
    """Probe result for a file soundfile cannot open but the fallback decoder can."""

    def __init__(self, seconds: float):
        self.seconds = seconds


def make_probe(loader: Callable | None = None) -> Callable:
    """probe(path) -> seconds (soundfile header) | Fallback (decoded by the fallback decoder) | raises."""
    def probe(path):
        import soundfile as sf

        try:
            return float(sf.info(str(path)).duration)
        except Exception:  # noqa: BLE001 - not readable by soundfile: try the fallback decoder
            load = loader
            if load is None:
                from app.core.backend_kit.audio import load_audio as load
            return Fallback(len(load(path, SAMPLE_RATE)) / SAMPLE_RATE)

    return probe


def cosyvoice_spec(model_revision: str) -> PrepSpec:
    return PrepSpec(family="cosyvoice-parquet", version=1, params={
        "upstream_commit": UPSTREAM_COMMIT,
        "model_revision": model_revision,
        "sample_rate": SAMPLE_RATE,
        "ref_sample_rate": REF_SAMPLE_RATE,
        "bn_normalize_version": BN_NORMALIZE_VERSION,
        "max_seconds": MAX_SECONDS,
        "shard_size": SHARD_SIZE,
        "instruct": INSTRUCT,
        "storage": "source-bytes",
    })


def _safe(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|\s]', "_", name) or DEFAULT_SPEAKER


def _one_line(s: str) -> str:
    return " ".join(s.split())


def _speaker(r: dict) -> str:
    return _safe(str(r.get("speaker") or DEFAULT_SPEAKER))


def _tts_text(r: dict) -> str:
    return r.get("tts_text") or bn_normalize(r["text"], mode="tts")


def write_kaldi_lists(rows, out: Path) -> None:
    """wav.scp, text, utt2spk, spk2utt (+ instruct), sorted by sample_id; text is bn_normalize(mode="tts")."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: str(r["sample_id"]))
    spk2utt: dict[str, list[str]] = {}
    with (out / "wav.scp").open("w", encoding="utf-8") as scp, (out / "text").open("w", encoding="utf-8") as txt, \
            (out / "utt2spk").open("w", encoding="utf-8") as u2s, \
            (out / "instruct").open("w", encoding="utf-8") as ins:
        for r in rows:
            sid, spk = str(r["sample_id"]), _speaker(r)
            scp.write(f"{sid} {r['audio_path']}\n")
            txt.write(f"{sid} {_one_line(_tts_text(r))}\n")
            u2s.write(f"{sid} {spk}\n")
            ins.write(f"{sid} {INSTRUCT}\n")
            spk2utt.setdefault(spk, []).append(sid)
    with (out / "spk2utt").open("w", encoding="utf-8") as f:
        for spk in sorted(spk2utt):
            f.write(f"{spk} {' '.join(spk2utt[spk])}\n")


def make_extractor(model_dir: Path) -> Callable[[str], tuple]:
    """extract(path) -> (utt embedding float32[192], speech tokens list[int]); follows upstream
    tools/extract_embedding.py (kaldi fbank 80 bins, dither 0, mean-normalised, CAM++ onnx) and
    tools/extract_speech_token.py (whisper log-mel 128 bins, speech_tokenizer_v3 onnx). Thread-safe."""
    import numpy as np
    import onnxruntime
    import torch
    import torchaudio.compliance.kaldi as kaldi
    import whisper

    from app.core.backend_kit.audio import load_audio

    model_dir = Path(model_dir)
    opt = onnxruntime.SessionOptions()
    opt.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    opt.intra_op_num_threads = 1
    avail = onnxruntime.get_available_providers()
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
    tok = onnxruntime.InferenceSession(str(model_dir / "speech_tokenizer_v3.onnx"), sess_options=opt,
                                       providers=providers)
    emb = onnxruntime.InferenceSession(str(model_dir / "campplus.onnx"), sess_options=opt, providers=providers)

    def extract(path):
        wav = torch.from_numpy(load_audio(path, SAMPLE_RATE))[None]
        feat = kaldi.fbank(wav, num_mel_bins=80, dither=0, sample_frequency=SAMPLE_RATE)
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = emb.run(None, {emb.get_inputs()[0].name: feat.unsqueeze(0).numpy()})[0].flatten()
        mel = whisper.log_mel_spectrogram(wav, n_mels=128)
        ins = tok.get_inputs()
        tokens = tok.run(None, {ins[0].name: mel.detach().cpu().numpy(),
                                ins[1].name: np.array([mel.shape[2]], dtype=np.int32)})[0].flatten().tolist()
        return embedding.astype("float32"), tokens

    return extract


def _final_dir(out: Path) -> Path:
    """ensure_prepared builds in `<target>.tmp-<pid>` and renames to `<target>`; data.list must hold the final paths."""
    return out.with_name(re.sub(r"\.tmp-\d+$", "", out.name))


def _audio_bytes(rec: dict, loader: Callable | None) -> bytes:
    if not rec.get("transcode"):
        return Path(rec["audio_path"]).read_bytes()
    import io

    import soundfile as sf

    load = loader
    if load is None:
        from app.core.backend_kit.audio import load_audio as load
    buf = io.BytesIO()
    sf.write(buf, load(rec["audio_path"], SAMPLE_RATE), SAMPLE_RATE, format="FLAC", subtype="PCM_16")
    return buf.getvalue()


def _row(row: dict, res, sid: str) -> tuple[str | None, dict | None]:
    if isinstance(res, Exception):
        return f"undecodable audio: {type(res).__name__}", None
    transcode = isinstance(res, Fallback)
    seconds = res.seconds if transcode else float(res or row.get("duration_s") or 0)
    if seconds > MAX_SECONDS:
        return f"duration {seconds:.1f} s > {MAX_SECONDS} s", None
    text = bn_normalize(row["text"], mode="tts")
    if not text:
        return "empty text", None
    return None, {"sample_id": sid, "speaker": str(row.get("speaker") or DEFAULT_SPEAKER),
                  "audio_path": str(Path(row["audio_path"]).resolve()), "text": row["text"], "tts_text": text,
                  "duration_s": round(seconds, 3), "transcode": transcode}


def _extract_all(recs: list[dict], extractor: Callable, tick: Callable[[int], None]):
    """(utt -> embedding, utt -> tokens), decoding a chunk at a time with a thread pool."""
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor

    emb: dict[str, object] = {}
    tokens: dict[str, object] = {}
    step = EXTRACT_THREADS * 16
    with ThreadPoolExecutor(EXTRACT_THREADS) as pool:
        for lo in range(0, len(recs), step):
            chunk = recs[lo:lo + step]
            for r, (e, t) in zip(chunk, pool.map(lambda r: extractor(r["audio_path"]), chunk)):
                if len(t) == 0:
                    raise PrepareError(f"speech tokenizer returned no tokens for {r['sample_id']}")
                emb[r["sample_id"]] = np.asarray(e, dtype="float32")
                tokens[r["sample_id"]] = np.asarray(t, dtype="int64")
            tick(len(chunk))
    return emb, tokens


def _write_split(out: Path, split: str, recs: list[dict], extractor: Callable, shard_size: int, rep,
                 loader: Callable | None) -> list[str]:
    """Everything for one split; returns its files relative to `out`."""
    import json

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    d = out / split
    recs = sorted(recs, key=lambda r: r["sample_id"])
    write_kaldi_lists(recs, d)
    files = [f"{split}/{n}" for n in ("wav.scp", "text", "utt2spk", "spk2utt", "instruct")]
    emb, tokens = _extract_all(recs, extractor, _extract_ticker(rep, split, len(recs)))
    by_spk: dict[str, list] = {}
    for r in recs:
        by_spk.setdefault(_speaker(r), []).append(emb[r["sample_id"]])
    spk_emb = {s: np.mean(v, axis=0).astype("float32") for s, v in by_spk.items()}
    try:
        import torch
    except ImportError:  # light environment; the trainer reads the parquet shards, not these
        torch = None
    if torch is not None:
        for name, obj in (("utt2embedding.pt", emb), ("spk2embedding.pt", spk_emb),
                          ("utt2speech_token.pt", tokens)):
            torch.save(obj, d / name)
            files.append(f"{split}/{name}")
    final = _final_dir(out)
    shards, utt2pq, spk2pq = [], [], []
    for i, lo in enumerate(range(0, len(recs), shard_size)):
        chunk = recs[lo:lo + shard_size]
        name = f"parquet_{i:09d}.tar"
        spks = [_speaker(r) for r in chunk]
        table = pa.table({
            "utt": [r["sample_id"] for r in chunk],
            "audio_data": pa.array([_audio_bytes(r, loader) for r in chunk], type=pa.binary()),
            "wav": [r["audio_path"] for r in chunk],
            "text": [_one_line(_tts_text(r)) for r in chunk],
            "spk": spks,
            "utt_embedding": pa.array([emb[r["sample_id"]] for r in chunk], type=pa.list_(pa.float32())),
            "spk_embedding": pa.array([spk_emb[s] for s in spks], type=pa.list_(pa.float32())),
            "speech_token": pa.array([tokens[r["sample_id"]] for r in chunk], type=pa.list_(pa.int64())),
            "instruct": [INSTRUCT] * len(chunk),
        })
        pq.write_table(table, d / name)
        shard = str(final / split / name)
        shards.append(shard)
        utt2pq.append(str(final / split / f"utt2parquet_{i:09d}.json"))
        spk2pq.append(str(final / split / f"spk2parquet_{i:09d}.json"))
        (d / f"utt2parquet_{i:09d}.json").write_text(
            json.dumps({r["sample_id"]: shard for r in chunk}, ensure_ascii=False, indent=2), encoding="utf-8")
        (d / f"spk2parquet_{i:09d}.json").write_text(
            json.dumps({s: shard for s in sorted(set(spks))}, ensure_ascii=False, indent=2), encoding="utf-8")
        files += [f"{split}/{name}", f"{split}/utt2parquet_{i:09d}.json", f"{split}/spk2parquet_{i:09d}.json"]
    for fname, lines in (("data.list", shards), ("utt2data.list", utt2pq), ("spk2data.list", spk2pq)):
        (d / fname).write_text("".join(ln + "\n" for ln in lines), encoding="utf-8")
        files.append(f"{split}/{fname}")
    return files


def _extract_ticker(rep, split: str, total: int) -> Callable[[int], None]:
    from app.core.backend_kit.audio_refs import PROGRESS_EVERY
    from app.core.contract import Phase

    state = {"done": 0, "reported": 0}

    def tick(n: int) -> None:
        state["done"] += n
        if rep is not None and (state["done"] - state["reported"] >= PROGRESS_EVERY or state["done"] == total):
            state["reported"] = state["done"]
            rep.status(Phase.prepare, state["done"], total, f"{split}: tokens/embeddings {state['done']}/{total}")

    return tick


def _write_refs(out: Path, recs: list[dict], loader: Callable | None) -> None:
    from app.core.backend_kit.audio import write_wav

    load = loader
    if load is None:
        from app.core.backend_kit.audio import load_audio as load
    for spk, r in pick_reference(recs).items():
        name = _safe(spk)
        write_wav(out / "refs" / f"{name}.wav", load(r["audio_path"], REF_SAMPLE_RATE), REF_SAMPLE_RATE)
        (out / "refs" / f"{name}.txt").write_text(_tts_text(r), encoding="utf-8")


def build_cosyvoice(model_dir: Path | None = None, rep=None, probe: Callable | None = None,
                    extractor: Callable | None = None, loader: Callable | None = None,
                    shard_size: int = SHARD_SIZE) -> Callable[[dict[str, list[dict]], Path], BuildResult]:
    probe = probe or make_probe(loader)

    def build(rows: dict[str, list[dict]], out: Path) -> BuildResult:
        out.mkdir(parents=True, exist_ok=True)
        tick = progress_ticker(rep, sum(len(v) for v in rows.values()))
        counts: dict[str, int] = {}
        skipped: list[Skip] = []
        kept: dict[str, list[dict]] = {}
        for split, items in rows.items():
            recs: list[dict] = []
            for row, res in probed_rows(items, probe, tick):
                sid = str(row.get("sample_id", ""))
                reason, rec = _row(row, res, sid)
                if reason:
                    skipped.append(Skip(sample_id=sid, split=split, reason=reason))
                else:
                    recs.append(rec)
            if not recs:
                raise PrepareError(f"split {split} is empty after prepare ({len(skipped)} rows skipped)")
            kept[split] = recs
            counts[split] = len(recs)
        ext = extractor or make_extractor(model_dir)
        files: list[str] = []
        for split, recs in kept.items():
            files += _write_split(out, split, recs, ext, shard_size, rep, loader)
        _write_refs(out, [r for recs in kept.values() for r in recs], loader)
        return BuildResult(counts=counts, skipped=skipped, files=files)

    return build


def check_cosyvoice_refs(info) -> None:
    """Cache-hit safety: the first and last wav.scp entry of each split must still point at existing audio."""
    for name in info.files:
        if not name.endswith("/wav.scp"):
            continue
        entries = _first_last(Path(info.path) / name)
        check_audio_refs([ln.split(" ", 1)[1] for ln in entries], str(info.path))


def _first_last(path: Path) -> list[str]:
    """First and last non-empty line of a text file (the last read by seeking from the end)."""
    with path.open("rb") as f:
        first = f.readline()
        if not first.strip():
            return []
        pos, tail = f.seek(0, 2), b""
        while pos > 0 and tail.count(b"\n") < 2:
            step = min(4096, pos)
            pos -= step
            f.seek(pos)
            tail = f.read(step) + tail
    last = [ln for ln in tail.splitlines() if ln.strip()][-1]
    return [first.decode("utf-8").rstrip("\r\n"), last.decode("utf-8")]
