"""Gradio UI: test Bangla STT quality (optional WER/CER vs reference) and latency: Whisper, Qwen3-ASR or NeMo.

Mirrors the LocalVoiceAgent STT pipeline: RealtimeSTT's `faster_whisper` engine = faster-whisper on a
CTranslate2 model, language forced to Bangla, silero VAD, beam 5. A HF-format checkpoint (what the fine-tune
produces, or mozilla-ai/whisper-large-v3-turbo-bn) is converted to CT2 once and cached; a CT2 dir
(the `ct2` export) is used as is. Qwen3-ASR checkpoints (repo id, training checkpoint or `safetensors` export)
run with transformers, bf16, greedy, through qwen_engine.py (also a RealtimeSTT engine for the live tab). NeMo
checkpoints (FastConformer / Conformer: repo id with a .nemo file, a .nemo file, or the `nemo` export dir) run with
NeMo, fp32, greedy RNN-T or CTC, through nemo_engine.py (also a live-tab engine).

Run: ./run.sh
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import gradio as gr
import numpy as np

from app.core.backend_kit.audio import load_audio
from app.core.text import bn_normalize
from nemo_engine import ENGINE as NEMO_ENGINE, NemoAsr, is_nemo, register as register_nemo
from qwen_engine import ENGINE as QWEN_ENGINE, QwenAsr, is_qwen, register
from registry import base_model_of, label_for, registry
from runlog import LOG_PATH, log
import whisper_chunking
from live_stream import TurnStreamer

register()  # module level: the recorder's spawned child re-imports this module and needs the engine too
register_nemo()


def _raise_whisper_token_cap() -> None:
    """faster-whisper hardcodes WhisperModel.max_length = 448 and passes it to CTranslate2, which then stops after
    224 generated tokens (half of max_length). Bangla costs ~16 tokens per second of speech, so every window longer
    than ~14 s was cut mid-sentence (often mid-character). Raise max_length so the cap becomes 440 new tokens,
    which with the 4-token prompt still fits the decoder's 448 positions. WHISPER_MAX_LENGTH=448 restores the stock
    behaviour (what LocalVoiceAgent does today). Module level, like register(): the live recorder's child process
    re-imports this module and builds its own WhisperModel."""
    import faster_whisper

    cap = int(os.environ.get("WHISPER_MAX_LENGTH", "880"))
    cls = faster_whisper.WhisperModel
    if getattr(cls, "_playground_cap", None) == cap:
        return
    init = getattr(cls, "_playground_orig_init", cls.__init__)

    def __init__(self, *a, **kw):
        init(self, *a, **kw)
        self.max_length = cap

    cls._playground_orig_init, cls._playground_cap, cls.__init__ = init, cap, __init__


_raise_whisper_token_cap()
whisper_chunking.install()  # 10 s windows, overlap at forced splits, retry of cut windows (see whisper_chunking.py)
WHISPER_WINDOW_S = whisper_chunking.WINDOW_S

DEFAULT_CKPT = os.environ.get("STT_CHECKPOINT", "openai/whisper-large-v3-turbo")
CT2_CACHE = Path(os.environ.get("CT2_CACHE", Path.home() / ".cache" / "stt-playground" / "ct2"))
DEVICE = os.environ.get("STT_DEVICE", "cuda")
LANG = "bn"

_state: dict = {"model": None, "label": None, "qwen": False, "family": None, "ckpt": None, "compute_type": None,
                "decoder": None}
NEMO_DECODERS = ["rnnt", "ctc"]


def family_of(ckpt: str) -> str:
    """Whisper, Qwen3-ASR or NeMo. NeMo first: its repos have no config.json for the Qwen check."""
    if is_nemo(ckpt):
        return "NeMo"
    return "Qwen3-ASR" if is_qwen(ckpt) else "Whisper"
_history: list[list] = []


def _with_tokenizer(ckpt: str) -> str:
    """A Whisper training checkpoint dir (checkpoint-N) holds no tokenizer, and the CT2 converter needs one.
    Stage a dir with links to the checkpoint's files plus the base model's tokenizer; other inputs pass through."""
    p = Path(ckpt)
    if not p.is_dir() or (p / "tokenizer.json").exists() or (p / "vocab.json").exists():
        return ckpt
    base = base_model_of(p) or "openai/whisper-large-v3-turbo"
    stage = CT2_CACHE.parent / "staged" / ckpt.strip("/").replace("/", "--")
    stage.mkdir(parents=True, exist_ok=True)
    if not (stage / "tokenizer.json").exists():  # tokenizer first: saving it never writes through a link
        from transformers import WhisperTokenizerFast

        WhisperTokenizerFast.from_pretrained(base).save_pretrained(str(stage))
    for f in p.iterdir():
        if f.is_file() and f.suffix not in (".pt", ".pth", ".bin") and not (stage / f.name).exists():
            (stage / f.name).symlink_to(f.resolve())
    return str(stage)


def resolve_ct2(ckpt: str, compute_type: str) -> tuple[str, str]:
    """(ct2 dir, note). A dir holding model.bin is used directly; anything else is converted and cached."""
    p = Path(ckpt)
    if p.is_dir() and (p / "model.bin").exists():
        return str(p), "CT2 dir used as is"
    out = CT2_CACHE / f"{ckpt.strip('/').replace('/', '--')}-{compute_type}"
    if (out / "model.bin").exists():
        return str(out), f"cached CT2 conversion {out}"
    out.parent.mkdir(parents=True, exist_ok=True)
    ckpt = _with_tokenizer(ckpt)
    script = Path(sys.executable).parent / "ct2-transformers-converter"
    head = [str(script)] if script.exists() else [sys.executable, "-m", "ctranslate2.converters.transformers"]
    subprocess.run(head + ["--model", ckpt, "--output_dir", str(out), "--force", "--quantization", compute_type,
                           "--copy_files", "preprocessor_config.json"], check=True)
    if not (out / "tokenizer.json").exists():  # some repos ship only vocab.json/merges.txt
        from transformers import WhisperTokenizerFast

        WhisperTokenizerFast.from_pretrained(ckpt).backend_tokenizer.save(str(out / "tokenizer.json"))
    return str(out), f"converted to CT2 ({compute_type}) at {out}"


def load_model(ckpt: str, compute_type: str, decoder: str = "rnnt") -> str:
    ckpt = ckpt.strip()
    if not ckpt:
        return "Enter a checkpoint (HF repo id, HF-format dir, CT2 dir, Qwen3-ASR dir or .nemo)."
    t0 = time.perf_counter()
    _state.update(model=None, qwen=False, family=None, ckpt=None, label=None, decoder=None)
    family = family_of(ckpt)
    if family == "NeMo":
        return _load_nemo(ckpt, decoder)
    if family == "Qwen3-ASR":
        return _load_qwen(ckpt)
    try:
        from faster_whisper import WhisperModel

        path, note = resolve_ct2(ckpt, compute_type)
        convert_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        model = WhisperModel(path, device=DEVICE, compute_type=compute_type)
        load_s = time.perf_counter() - t1
        import numpy as np

        t2 = time.perf_counter()
        list(model.transcribe(np.zeros(16000, dtype="float32"), language=LANG)[0])  # warm-up (CUDA init)
        warm_s = time.perf_counter() - t2
    except Exception as e:  # noqa: BLE001 - show any load error in the UI
        log("load", ok=False, model=label_for(ckpt), checkpoint=ckpt, family="Whisper", device=DEVICE,
            compute_type=compute_type, error=f"{type(e).__name__}: {e}")
        return f"Load failed: {type(e).__name__}: {e}"
    _state.update(model=model, label=label_for(ckpt), family="Whisper", ckpt=ckpt, compute_type=compute_type)
    log("load", ok=True, model=_state["label"], checkpoint=ckpt, family="Whisper",
        engine="faster-whisper (CTranslate2)", ct2_path=path, device=DEVICE, compute_type=compute_type,
        convert_s=round(convert_s, 3), load_s=round(load_s, 3), warmup_s=round(warm_s, 3))
    return (f"Loaded **{ckpt}** on {DEVICE}/{compute_type}. {note}. "
            f"convert {convert_s:.1f}s, load {load_s:.1f}s, warm-up {warm_s:.1f}s.")


def _load_qwen(ckpt: str) -> str:
    t0 = time.perf_counter()
    try:
        model = QwenAsr(ckpt, DEVICE)
        load_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        model.transcribe(np.zeros(16000, dtype="float32"))  # warm-up (CUDA init)
        warm_s = time.perf_counter() - t1
    except Exception as e:  # noqa: BLE001 - show any load error in the UI
        log("load", ok=False, model=label_for(ckpt), checkpoint=ckpt, family="Qwen3-ASR", device=DEVICE,
            compute_type="bfloat16", error=f"{type(e).__name__}: {e}")
        return f"Load failed: {type(e).__name__}: {e}"
    _state.update(model=model, label=label_for(ckpt), qwen=True, family="Qwen3-ASR", ckpt=ckpt,
                  compute_type="bfloat16")
    log("load", ok=True, model=_state["label"], checkpoint=ckpt, family="Qwen3-ASR", engine="transformers",
        device=DEVICE, compute_type="bfloat16", load_s=round(load_s, 3), warmup_s=round(warm_s, 3))
    return (f"Loaded **{ckpt}** (Qwen3-ASR, transformers) on {DEVICE}/bf16, greedy; compute type, beam and VAD "
            f"do not apply. load {load_s:.1f}s, warm-up {warm_s:.1f}s.")


def _load_nemo(ckpt: str, decoder: str) -> str:
    t0 = time.perf_counter()
    try:
        model = NemoAsr(ckpt, DEVICE, decoder)
        load_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        model.transcribe(np.zeros(16000, dtype="float32"))  # warm-up (CUDA init)
        warm_s = time.perf_counter() - t1
    except Exception as e:  # noqa: BLE001 - show any load error in the UI
        log("load", ok=False, model=label_for(ckpt), checkpoint=ckpt, family="NeMo", device=DEVICE,
            compute_type="float32", decoder=decoder, error=f"{type(e).__name__}: {e}")
        return f"Load failed: {type(e).__name__}: {e}"
    _state.update(model=model, label=label_for(ckpt), family="NeMo", ckpt=ckpt, compute_type="float32",
                  decoder=model.decoder)
    log("load", ok=True, model=_state["label"], checkpoint=ckpt, family="NeMo", engine=f"nemo ({model.kind})",
        device=DEVICE, compute_type="float32", decoder=model.decoder, load_s=round(load_s, 3),
        warmup_s=round(warm_s, 3))
    note = "" if model.decoder == decoder else f" ({decoder} not available: this model has only {model.decoder})"
    return (f"Loaded **{ckpt}** (NeMo {model.kind}) on {DEVICE}/fp32, greedy **{model.decoder.upper()}**{note}; "
            f"compute type, beam and VAD do not apply. load {load_s:.1f}s, warm-up {warm_s:.1f}s.")


def _edit_distance(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def _score(hyp: str, ref: str) -> tuple[float, float] | None:
    """(WER, CER) after bn_normalize (asr mode), or None for an empty reference."""
    h, r = bn_normalize(hyp, mode="asr"), bn_normalize(ref, mode="asr")
    if not r.split():
        return None
    wer = _edit_distance(r.split(), h.split()) / len(r.split())
    cer = _edit_distance(list(r), list(h)) / max(len(r), 1)
    return wer, cer


def transcribe(audio_path, reference, runs, beam, vad):
    model = _state["model"]
    if model is None:
        return "Load a checkpoint first.", "", _history
    if not audio_path:
        return "Record or upload audio.", "", _history
    try:  # decode once, like RealtimeSTT feeds a numpy array; also avoids faster-whisper's PyAV decoder
        wav = load_audio(audio_path, 16000)
    except Exception as e:  # noqa: BLE001
        return f"Cannot decode audio: {type(e).__name__}: {e}", "", _history
    lat, text, audio_s = [], "", len(wav) / 16000
    for _ in range(max(1, int(runs))):
        t0 = time.perf_counter()
        if _state["family"] != "Whisper":  # Qwen3-ASR / NeMo: whole clip, greedy
            text = model.transcribe(wav)
        else:
            if vad:  # same path as the live recorder: batched pipeline, windowed by whisper_chunking
                from faster_whisper import BatchedInferencePipeline

                segs, _info = BatchedInferencePipeline(model=model).transcribe(
                    wav, language=LANG, task="transcribe", beam_size=int(beam), vad_filter=True, batch_size=16)
            else:
                segs, _info = model.transcribe(wav, language=LANG, task="transcribe", beam_size=int(beam),
                                              vad_filter=False, condition_on_previous_text=False)
            text = " ".join(s.text.strip() for s in segs).strip()  # generator: decoding happens here
        lat.append(time.perf_counter() - t0)
    mean = sum(lat) / len(lat)
    rtf = mean / audio_s if audio_s else 0.0
    lines = [f"audio {audio_s:.2f}s",
             f"latency {mean:.3f} s mean, {min(lat):.3f} s best (n={len(lat)})",
             f"RTF {rtf:.3f}  " + ("(Qwen3-ASR, greedy)" if _state["qwen"]
                                     else f"(NeMo, greedy {_state['decoder'].upper()})" if _state["family"] == "NeMo"
                                     else f"(beam {int(beam)}, VAD {'on, ' + f'{WHISPER_WINDOW_S:g} s windows' if vad else 'off'})")]
    score = _score(text, reference) if reference.strip() else None
    if score:
        lines.append(f"WER {score[0]:.1%} | CER {score[1]:.1%} (bn_normalize, asr mode)")
    qwen = _state["family"] != "Whisper"  # greedy, whole clip: no beam / VAD / window settings
    log("transcribe", mode="file", model=_state["label"], checkpoint=_state["ckpt"],
        family=_state["family"], device=DEVICE, compute_type=_state["compute_type"], decoder=_state["decoder"],
        decoding="greedy" if qwen else "beam", beam_size=None if qwen else int(beam),
        vad_filter=None if qwen else bool(vad), window_s=None if qwen or not vad else WHISPER_WINDOW_S,
        whisper_max_length=None if qwen else int(os.environ.get("WHISPER_MAX_LENGTH", "880")),
        language=LANG, audio_file=str(audio_path),
        audio_s=round(audio_s, 3), runs=len(lat), latency_s=round(mean, 4), latency_best_s=round(min(lat), 4),
        latency_runs_s=[round(x, 4) for x in lat], rtf=round(rtf, 4), text=text, reference=reference.strip() or None,
        wer=round(score[0], 4) if score else None, cer=round(score[1], 4) if score else None)
    lines.append(f"logged to {LOG_PATH}")
    _history.insert(0, [_state["label"], f"{audio_s:.2f}", f"{mean * 1000:.0f}", f"{rtf:.3f}", text[:80]])
    del _history[50:]
    return text, "\n".join(lines), _history


# ---- live mode: the reference project's pipeline. RealtimeSTT AudioToTextRecorder (faster_whisper engine), fed
# from the browser mic via feed_audio (AUDIO_IO=browser), turn end = post_speech_silence_duration (STT_SILENCE_SEC),
# stage marks from the same callbacks as LocalVoiceAgent's stt.timing_callbacks. ----
SILENCE_SEC = float(os.environ.get("STT_SILENCE_SEC", "0.6"))  # LocalVoiceAgent default
TURN_AUDIO_DIR = LOG_PATH.parent / "audio"  # one WAV per live turn, referenced by the log's audio_file
STREAM_DEFAULT = os.environ.get("STT_STREAM", "0") == "1"  # live tab starts in streaming mode
STREAM_PAUSE_S = float(os.environ.get("STT_STREAM_PAUSE_S", "0.3"))


class Live:
    def __init__(self):
        self.rec = None
        self.thread = None
        self.running = False
        self.marks: dict[str, float] = {}
        self.turns: list[dict] = []
        self.state = "stopped"
        self.lock = threading.Lock()
        self.resampler = None
        self.resampler_sr = None
        self.label = None
        self.cfg: dict = {}
        self.stream = STREAM_DEFAULT  # transcribe chunks while speaking (live_stream.py); read per turn
        self.chunk_pause = STREAM_PAUSE_S
        self.streamer = None
        self.turn_gen = 0

    def _mark(self, k):
        with self.lock:
            self.marks[k] = time.perf_counter()

    def _callbacks(self):
        def on_start(_audio=None):
            self._mark("stt_start")
            return None  # truthy would abort the transcription

        def on_turn_stop():
            with self.lock:
                if "endpoint" not in self.marks:  # user resumed speaking: not the end of the turn
                    self.marks.pop("speech_end", None)

        def on_recording_start():
            self.state = "speaking..."
            self.turn_gen += 1
            self.streamer = None
            if self.stream and self.rec is not None:
                gen = self.turn_gen
                pause = min(float(self.chunk_pause), 0.75 * float(self.cfg.get("post_speech_silence_s", SILENCE_SEC)))
                self.streamer = TurnStreamer(self.rec, pause, lambda: self.turn_gen == gen)

        return {
            "on_turn_detection_start": lambda: self._mark("speech_end"),
            "on_turn_detection_stop": on_turn_stop,
            "on_recording_stop": lambda: self._mark("endpoint"),
            "on_transcription_start": on_start,
            "on_recording_start": on_recording_start,
        }

    def start(self, ckpt, compute_type, silence, beam, stream=None, chunk_pause=None, decoder="rnnt") -> str:
        self.stop()
        from RealtimeSTT import AudioToTextRecorder

        self.set_stream(stream, chunk_pause)

        ckpt = ckpt.strip()
        family = family_of(ckpt)
        qwen = family != "Whisper"  # Qwen3-ASR / NeMo: greedy, no CT2 conversion, no beam
        engine = {"NeMo": NEMO_ENGINE, "Qwen3-ASR": QWEN_ENGINE}.get(family, "faster_whisper")
        ctype = {"NeMo": "float32", "Qwen3-ASR": "bfloat16"}.get(family, compute_type)
        path = ckpt if qwen else resolve_ct2(ckpt, compute_type)[0]
        if family == "NeMo":  # read by the engine in the recorder's spawned process (it inherits the environment)
            os.environ["STT_NEMO_DECODER"] = decoder
        self.state = "loading model in recorder process..."
        self.rec = AudioToTextRecorder(
            model=path, transcription_engine=engine, device=DEVICE,
            compute_type=ctype, language=LANG, use_microphone=False, spinner=False, enable_realtime_transcription=False,
            post_speech_silence_duration=float(silence), beam_size=int(beam), **self._callbacks())
        self.label = label_for(ckpt)
        self.cfg = {"model": self.label, "checkpoint": ckpt, "family": family,
                    "engine": engine, "device": DEVICE, "decoder": decoder if family == "NeMo" else None,
                    "compute_type": ctype, "decoding": "greedy" if qwen else "beam",
                    "beam_size": None if qwen else int(beam), "language": LANG,
                    "post_speech_silence_s": float(silence),
                    "window_s": None if qwen else WHISPER_WINDOW_S,
                    "whisper_max_length": None if qwen else int(os.environ.get("WHISPER_MAX_LENGTH", "880"))}
        log("live_start", **self.cfg)
        self.resampler = None
        self.running = True
        self.state = "listening..."
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        how = {"NeMo": f"NeMo, fp32, greedy {decoder.upper()}", "Qwen3-ASR": "Qwen3-ASR, bf16, greedy"}.get(
            family, f"{compute_type}, beam {int(beam)}")
        return f"Live recorder ready: {ckpt} ({how}), turn-end silence {float(silence):.2f}s. {self._mode_text()}"

    def set_stream(self, stream=None, chunk_pause=None) -> str:
        """Toggle streaming decode; applies from the next turn, no restart."""
        if stream is not None:
            self.stream = bool(stream)
        if chunk_pause is not None:
            self.chunk_pause = float(chunk_pause)
        if not self.stream and self.streamer:  # switched off mid-turn: this turn is decoded whole by text()
            self.streamer.cancel()
            self.streamer = None
        if self.rec is not None:
            log("live_mode", model=self.label, streaming=self.stream, chunk_pause_s=self.chunk_pause)
        return self._mode_text()

    def _mode_text(self) -> str:
        return (f"Mode: **streaming** (chunks decoded while speaking at pauses >= {self.chunk_pause:.2f}s, "
                f"rest flushed at turn end)." if self.stream else "Mode: **turn-end decode** (RealtimeSTT default).")

    def _turn(self) -> tuple[str, np.ndarray | None, dict, float | None]:
        """(text, turn audio, extra log fields, stt start) for one turn. Turn-end mode: the recorder's text(), which
        decodes the whole turn after the endpoint. Streaming mode: wait_audio() (same endpointing, no decode), then
        the TurnStreamer flushes whatever it has not decoded yet."""
        rec = self.rec
        rec.interrupt_stop_event.clear()  # what rec.text() does: wait_audio(), then transcribe()
        rec.was_interrupted.clear()
        rec.wait_audio()  # the loop waits here for the next turn, so the mode is decided after it (per turn)
        streamer, self.streamer = self.streamer, None
        if rec.is_shut_down or rec.interrupt_stop_event.is_set():
            if streamer:
                streamer.cancel()
            return "", None, {}, None
        if streamer is None:  # turn-end mode (or streaming switched on mid-turn): decode the whole turn
            text = rec.transcribe()
            return text or "", getattr(rec, "last_transcription_bytes", None), {"streaming": False}, None
        st = time.perf_counter()
        audio = getattr(rec, "audio", None)
        if audio is None or not len(audio):
            streamer.cancel()
            return "", None, {}, st
        text, extra = streamer.finish(audio)
        return text, audio, extra, st

    def _loop(self):
        while self.running:
            try:
                text, audio, extra, st0 = self._turn()
            except Exception:  # noqa: BLE001 - recorder shut down
                if not self.running:
                    break
                traceback.print_exc()
                continue
            done = time.perf_counter()
            with self.lock:
                m, self.marks = self.marks, {}
            if not text or "speech_end" not in m:
                self.state = "listening..."
                continue
            se, ep = m["speech_end"], m.get("endpoint", done)
            st = st0 or m.get("stt_start", ep)
            mode = "stream" if extra.get("streaming") else "turn-end"
            self.turns.insert(0, {"mode": mode, "chunks": len(extra.get("chunks", [])) or 1,
                                  "silence_ms": (ep - se) * 1000, "gap_ms": (st - ep) * 1000,
                                  "stt_ms": (done - st) * 1000, "total_ms": (done - se) * 1000, "text": text.strip()})
            wav_path, audio_s = self._save_turn_audio(audio)
            cfg = {**self.cfg, "streaming": False, **extra}
            log("transcribe", mode="live", **cfg, silence_wait_s=round(ep - se, 4),
                handoff_s=round(st - ep, 4), latency_s=round(done - st, 4),
                speech_end_to_text_s=round(done - se, 4), audio_s=audio_s, audio_file=wav_path,
                text=text.strip())
            self.state = "listening..."

    def _save_turn_audio(self, audio) -> tuple[str | None, float | None]:
        """Save the turn's audio (16 kHz float32: what the recorder sent to the engine in turn-end mode, the whole
        recorded turn in streaming mode) as a WAV next to the log, so a turn can be replayed in the file tab."""
        if audio is None or not len(audio):
            return None, None
        import soundfile as sf

        out = TURN_AUDIO_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-{int(time.time() * 1000) % 1000:03d}.wav"
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            sf.write(out, np.asarray(audio, dtype=np.float32), 16000)
        except Exception as e:  # noqa: BLE001 - never break the live loop over a log file
            print(f"cannot save turn audio: {e}")
            return None, round(len(audio) / 16000, 3)
        return str(out), round(len(audio) / 16000, 3)

    def feed(self, chunk):
        """Browser mic chunk -> 16 kHz int16 -> recorder. Resampled here with a stateful stream resampler:
        RealtimeSTT's feed_audio resamples each chunk on its own (scipy FFT), which garbles 48 kHz mic audio."""
        if not self.running or chunk is None:
            return
        sr, data = chunk
        x = np.asarray(data)
        x = x.astype(np.float32) / 32768.0 if x.dtype.kind in "iu" else x.astype(np.float32)
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != 16000:
            if self.resampler is None or self.resampler_sr != sr:
                import soxr

                self.resampler, self.resampler_sr = soxr.ResampleStream(sr, 16000, 1, dtype="float32"), sr
            x = self.resampler.resample_chunk(x)
        if len(x):
            self.rec.feed_audio((np.clip(x, -1.0, 1.0) * 32767).astype(np.int16).tobytes(), original_sample_rate=16000)

    def stop(self) -> str:
        self.running = False
        rec, self.rec = self.rec, None
        if rec is not None:
            log("live_stop", model=self.label, turns=len(self.turns))
            try:
                rec.shutdown()
            except Exception:  # noqa: BLE001
                pass
        self.state = "stopped"
        return "Live recorder stopped."

    def view(self):
        rows = [[t["mode"], t["chunks"], f"{t['silence_ms']:.0f}", f"{t['gap_ms']:.0f}", f"{t['stt_ms']:.0f}", f"{t['total_ms']:.0f}", t["text"]]
                for t in self.turns[:50]]
        head = f"**{self.state}** · {self._mode_text()}"
        if self.turns:
            t = self.turns[0]
            head += (f"\n\nLast turn: speech end \u2192 text = **{t['total_ms']:.0f} ms** "
                     f"(silence wait {t['silence_ms']:.0f} + hand-off {t['gap_ms']:.0f} + STT {t['stt_ms']:.0f})")
        return head, (self.turns[0]["text"] if self.turns else ""), rows


live = Live()


def build():
    with gr.Blocks(title="Bangla STT playground") as demo:
        gr.Markdown("## Bangla STT playground: quality & latency\nWhisper via faster-whisper (CT2), the engine "
                    "RealtimeSTT uses; Qwen3-ASR via transformers (bf16, greedy); NeMo FastConformer / Conformer "
                    "(fp32, greedy RNN-T or CTC).")
        reg = registry()
        with gr.Row():
            preset = gr.Dropdown(list(reg), value=label_for(DEFAULT_CKPT, reg) if DEFAULT_CKPT in reg.values() else None,
                                 label="Model (base + fine-tuned runs; selecting loads it)", scale=4)
            refresh = gr.Button("\u21bb Rescan runs", scale=0, min_width=120)
        with gr.Row():
            ckpt = gr.Textbox(value=DEFAULT_CKPT, label="Checkpoint (HF repo id, HF-format dir, CT2 dir, Qwen3-ASR dir, or .nemo / nemo export dir)", scale=3)
            ctype = gr.Dropdown(["float16", "int8_float16", "int8", "float32"], value="float16", label="Compute type (Whisper)", scale=1)
            dec = gr.Dropdown(NEMO_DECODERS, value=os.environ.get("STT_NEMO_DECODER", "rnnt"),
                              label="Decoder (NeMo hybrid; reloads)", scale=1)
            load_btn = gr.Button("Load", variant="primary", scale=1)
        status = gr.Markdown("Not loaded.")
        gr.Markdown(f"Every load, transcription and live turn is appended to `{LOG_PATH}` (JSONL, seconds).")
        with gr.Tab("Live (auto turn-end)"):
            gr.Markdown("Same pipeline as LocalVoiceAgent: RealtimeSTT recorder (faster-whisper, or the Qwen3-ASR engine "
                        "for a Qwen checkpoint). Press **Start**, allow the mic "
                        "(localhost or HTTPS), speak. When you pause for the silence threshold the turn ends and is transcribed "
                        "automatically. Changing settings needs a restart (Start again).")
            with gr.Row():
                lsil = gr.Slider(0.2, 2.0, value=SILENCE_SEC, step=0.05, label="Turn-end silence (s)  [reference STT_SILENCE_SEC=0.6]")
                lbeam = gr.Slider(1, 10, value=5, step=1, label="Beam size")
                lstart = gr.Button("Start", variant="primary")
                lstop = gr.Button("Stop")
            with gr.Row():
                lstream = gr.Checkbox(value=STREAM_DEFAULT, label="Stream: transcribe chunks while speaking, flush the "
                                      "rest at turn end (applies from the next turn)")
                lpause = gr.Slider(0.15, 1.0, value=STREAM_PAUSE_S, step=0.05,
                                   label="Chunk pause (s): commit a chunk at a pause this long (capped at 75% of turn-end silence)")
            lmsg = gr.Markdown("")
            mic = gr.Audio(sources=["microphone"], streaming=True, type="numpy", label="Microphone")
            lstatus = gr.Markdown("**stopped**")
            ltext = gr.Textbox(label="Last transcript", lines=3, interactive=False)
            lturns = gr.Dataframe(headers=["mode", "chunks", "silence wait ms", "hand-off ms", "STT ms", "speech end \u2192 text ms", "text"],
                                  label="Turns (speech end = VAD hears silence, as in the reference's TurnTimer)", interactive=False)
            mic.stream(live.feed, mic, None, stream_every=0.2, time_limit=None, concurrency_limit=1, show_progress="hidden")
            lstart.click(live.start, [ckpt, ctype, lsil, lbeam, lstream, lpause, dec], lmsg)
            lstream.change(live.set_stream, [lstream, lpause], lmsg)
            lpause.release(live.set_stream, [lstream, lpause], lmsg)
            lstop.click(live.stop, None, lmsg)
            gr.Timer(0.3).tick(live.view, None, [lstatus, ltext, lturns])
        with gr.Tab("Record / upload"):
            with gr.Row():
                with gr.Column():
                    audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Audio")
                    ref = gr.Textbox(label="Reference transcript (optional, for WER/CER)", lines=2)
                    with gr.Row():
                        runs = gr.Slider(1, 10, value=3, step=1, label="Timed runs (averaged)")
                        beam = gr.Slider(1, 10, value=5, step=1, label="Beam size")
                    vad = gr.Checkbox(value=True, label="VAD filter (silero, as in RealtimeSTT)")
                    go = gr.Button("Transcribe", variant="primary")
                with gr.Column():
                    out = gr.Textbox(label="Transcript", lines=5, interactive=False)
                    metrics = gr.Textbox(label="Metrics", lines=4, interactive=False)
            hist = gr.Dataframe(headers=["model", "audio s", "latency ms", "RTF", "text"], value=_history,
                                label="History", interactive=False)
        def pick(name, compute_type, decoder):
            path = registry().get(name)
            return (gr.skip(), gr.skip()) if not path else (path, load_model(path, compute_type, decoder))

        def switch_decoder(path, compute_type, decoder):
            return load_model(path, compute_type, decoder) if family_of(path) == "NeMo" else gr.skip()

        preset.input(pick, [preset, ctype, dec], [ckpt, status])
        refresh.click(lambda: gr.update(choices=list(registry())), None, preset)
        load_btn.click(load_model, [ckpt, ctype, dec], status)
        dec.input(switch_decoder, [ckpt, ctype, dec], status)
        go.click(transcribe, [audio, ref, runs, beam, vad], [out, metrics, hist])
        demo.load(load_model, [ckpt, ctype, dec], status)
    return demo


if __name__ == "__main__":
    build().launch(server_name=os.environ.get("HOST", "0.0.0.0"), server_port=int(os.environ.get("PORT", "7860")))
