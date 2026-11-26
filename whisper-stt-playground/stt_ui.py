"""Gradio UI: test Whisper large-v3-turbo Bangla STT quality (optional WER/CER vs reference) and latency.

Mirrors the LocalVoiceAgent STT pipeline: RealtimeSTT's `faster_whisper` engine = faster-whisper on a
CTranslate2 model, language forced to Bangla, silero VAD, beam 5. A HF-format checkpoint (what the fine-tune
produces, or mozilla-ai/whisper-large-v3-turbo-bn) is converted to CT2 once and cached; a CT2 dir
(the `ct2` export) is used as is.

Run: ./run.sh
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import gradio as gr
import numpy as np

from app.core.backend_kit.audio import load_audio
from app.core.text import bn_normalize

PRESETS = {
    "Whisper large-v3-turbo (base, to be fine-tuned)": "openai/whisper-large-v3-turbo",
    "Whisper large-v3-turbo Bengali (mozilla-ai)": "mozilla-ai/whisper-large-v3-turbo-bn",
}
DEFAULT_CKPT = os.environ.get("STT_CHECKPOINT", "openai/whisper-large-v3-turbo")
CT2_CACHE = Path(os.environ.get("CT2_CACHE", Path.home() / ".cache" / "whisper-stt-playground" / "ct2"))
DEVICE = os.environ.get("STT_DEVICE", "cuda")
LANG = "bn"

_state: dict = {"model": None, "label": None}
_history: list[list] = []


def resolve_ct2(ckpt: str, compute_type: str) -> tuple[str, str]:
    """(ct2 dir, note). A dir holding model.bin is used directly; anything else is converted and cached."""
    p = Path(ckpt)
    if p.is_dir() and (p / "model.bin").exists():
        return str(p), "CT2 dir used as is"
    out = CT2_CACHE / f"{ckpt.strip('/').replace('/', '--')}-{compute_type}"
    if (out / "model.bin").exists():
        return str(out), f"cached CT2 conversion {out}"
    out.parent.mkdir(parents=True, exist_ok=True)
    script = Path(sys.executable).parent / "ct2-transformers-converter"
    head = [str(script)] if script.exists() else [sys.executable, "-m", "ctranslate2.converters.transformers"]
    subprocess.run(head + ["--model", ckpt, "--output_dir", str(out), "--force", "--quantization", compute_type,
                           "--copy_files", "preprocessor_config.json"], check=True)
    if not (out / "tokenizer.json").exists():  # some repos ship only vocab.json/merges.txt
        from transformers import WhisperTokenizerFast

        WhisperTokenizerFast.from_pretrained(ckpt).backend_tokenizer.save(str(out / "tokenizer.json"))
    return str(out), f"converted to CT2 ({compute_type}) at {out}"


def load_model(ckpt: str, compute_type: str) -> str:
    ckpt = ckpt.strip()
    if not ckpt:
        return "Enter a checkpoint (HF repo id, HF-format dir, or CT2 dir)."
    t0 = time.perf_counter()
    try:
        from faster_whisper import WhisperModel

        _state["model"] = None
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
        return f"Load failed: {type(e).__name__}: {e}"
    _state.update(model=model, label=ckpt.rstrip("/").split("/")[-1])
    return (f"Loaded **{ckpt}** on {DEVICE}/{compute_type}. {note}. "
            f"convert {convert_s:.1f}s, load {load_s:.1f}s, warm-up {warm_s:.1f}s.")


def _edit_distance(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def _score(hyp: str, ref: str) -> str:
    h, r = bn_normalize(hyp, mode="asr"), bn_normalize(ref, mode="asr")
    if not r.split():
        return ""
    wer = _edit_distance(r.split(), h.split()) / len(r.split())
    cer = _edit_distance(list(r), list(h)) / max(len(r), 1)
    return f"WER {wer:.1%} | CER {cer:.1%} (bn_normalize, asr mode)"


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
        segs, _info = model.transcribe(wav, language=LANG, task="transcribe", beam_size=int(beam),
                                      vad_filter=bool(vad), condition_on_previous_text=False)
        text = " ".join(s.text.strip() for s in segs).strip()  # generator: decoding happens here
        lat.append(time.perf_counter() - t0)
    mean = sum(lat) / len(lat)
    rtf = mean / audio_s if audio_s else 0.0
    lines = [f"audio {audio_s:.2f}s",
             f"latency {mean * 1000:.0f} ms mean, {min(lat) * 1000:.0f} ms best (n={len(lat)})",
             f"RTF {rtf:.3f}  (beam {int(beam)}, VAD {'on' if vad else 'off'})"]
    if reference.strip():
        lines.append(_score(text, reference))
    _history.insert(0, [_state["label"], f"{audio_s:.2f}", f"{mean * 1000:.0f}", f"{rtf:.3f}", text[:80]])
    del _history[50:]
    return text, "\n".join(lines), _history


# ---- live mode: the reference project's pipeline. RealtimeSTT AudioToTextRecorder (faster_whisper engine), fed
# from the browser mic via feed_audio (AUDIO_IO=browser), turn end = post_speech_silence_duration (STT_SILENCE_SEC),
# stage marks from the same callbacks as LocalVoiceAgent's stt.timing_callbacks. ----
SILENCE_SEC = float(os.environ.get("STT_SILENCE_SEC", "0.6"))  # LocalVoiceAgent default


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

        return {
            "on_turn_detection_start": lambda: self._mark("speech_end"),
            "on_turn_detection_stop": on_turn_stop,
            "on_recording_stop": lambda: self._mark("endpoint"),
            "on_transcription_start": on_start,
            "on_recording_start": lambda: setattr(self, "state", "speaking..."),
        }

    def start(self, ckpt, compute_type, silence, beam) -> str:
        self.stop()
        from RealtimeSTT import AudioToTextRecorder

        path, note = resolve_ct2(ckpt.strip(), compute_type)
        self.state = "loading model in recorder process..."
        self.rec = AudioToTextRecorder(
            model=path, transcription_engine="faster_whisper", device=DEVICE, compute_type=compute_type,
            language=LANG, use_microphone=False, spinner=False, enable_realtime_transcription=False,
            post_speech_silence_duration=float(silence), beam_size=int(beam), **self._callbacks())
        self.label = ckpt.rstrip("/").split("/")[-1]
        self.resampler = None
        self.running = True
        self.state = "listening..."
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return f"Live recorder ready: {ckpt} ({compute_type}), turn-end silence {float(silence):.2f}s, beam {int(beam)}."

    def _loop(self):
        while self.running:
            try:
                text = self.rec.text()
            except Exception:  # noqa: BLE001 - recorder shut down
                break
            done = time.perf_counter()
            with self.lock:
                m, self.marks = self.marks, {}
            if not text or "speech_end" not in m:
                self.state = "listening..."
                continue
            se, ep, st = m["speech_end"], m.get("endpoint", done), m.get("stt_start", m.get("endpoint", done))
            self.turns.insert(0, {"silence_ms": (ep - se) * 1000, "gap_ms": (st - ep) * 1000,
                                  "stt_ms": (done - st) * 1000, "total_ms": (done - se) * 1000, "text": text.strip()})
            self.state = "listening..."

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
            try:
                rec.shutdown()
            except Exception:  # noqa: BLE001
                pass
        self.state = "stopped"
        return "Live recorder stopped."

    def view(self):
        rows = [[f"{t['silence_ms']:.0f}", f"{t['gap_ms']:.0f}", f"{t['stt_ms']:.0f}", f"{t['total_ms']:.0f}", t["text"]]
                for t in self.turns[:50]]
        head = f"**{self.state}**"
        if self.turns:
            t = self.turns[0]
            head += (f"\n\nLast turn: speech end \u2192 text = **{t['total_ms']:.0f} ms** "
                     f"(silence wait {t['silence_ms']:.0f} + hand-off {t['gap_ms']:.0f} + STT {t['stt_ms']:.0f})")
        return head, (self.turns[0]["text"] if self.turns else ""), rows


live = Live()


def build():
    with gr.Blocks(title="Whisper Bangla STT playground") as demo:
        gr.Markdown("## Whisper large-v3-turbo Bangla STT: quality & latency\nfaster-whisper (CT2), same engine as RealtimeSTT.")
        with gr.Row():
            preset = gr.Dropdown(list(PRESETS), value=None, label="Preset", scale=2)
            ckpt = gr.Textbox(value=DEFAULT_CKPT, label="Checkpoint (HF repo id, HF-format dir, or CT2 dir)", scale=3)
            ctype = gr.Dropdown(["float16", "int8_float16", "int8", "float32"], value="float16", label="Compute type", scale=1)
            load_btn = gr.Button("Load", variant="primary", scale=1)
        status = gr.Markdown("Not loaded.")
        with gr.Tab("Live (auto turn-end)"):
            gr.Markdown("Same pipeline as LocalVoiceAgent: RealtimeSTT recorder (faster-whisper). Press **Start**, allow the mic "
                        "(localhost or HTTPS), speak. When you pause for the silence threshold the turn ends and is transcribed "
                        "automatically. Changing settings needs a restart (Start again).")
            with gr.Row():
                lsil = gr.Slider(0.2, 2.0, value=SILENCE_SEC, step=0.05, label="Turn-end silence (s)  [reference STT_SILENCE_SEC=0.6]")
                lbeam = gr.Slider(1, 10, value=5, step=1, label="Beam size")
                lstart = gr.Button("Start", variant="primary")
                lstop = gr.Button("Stop")
            lmsg = gr.Markdown("")
            mic = gr.Audio(sources=["microphone"], streaming=True, type="numpy", label="Microphone")
            lstatus = gr.Markdown("**stopped**")
            ltext = gr.Textbox(label="Last transcript", lines=3, interactive=False)
            lturns = gr.Dataframe(headers=["silence wait ms", "hand-off ms", "STT ms", "speech end \u2192 text ms", "text"],
                                  label="Turns (speech end = VAD hears silence, as in the reference's TurnTimer)", interactive=False)
            mic.stream(live.feed, mic, None, stream_every=0.2, time_limit=None, concurrency_limit=1, show_progress="hidden")
            lstart.click(live.start, [ckpt, ctype, lsil, lbeam], lmsg)
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
        preset.change(lambda p: PRESETS.get(p, gr.skip()), preset, ckpt)
        load_btn.click(load_model, [ckpt, ctype], status)
        go.click(transcribe, [audio, ref, runs, beam, vad], [out, metrics, hist])
        demo.load(load_model, [ckpt, ctype], status)
    return demo


if __name__ == "__main__":
    build().launch(server_name=os.environ.get("HOST", "0.0.0.0"), server_port=int(os.environ.get("PORT", "7860")))
