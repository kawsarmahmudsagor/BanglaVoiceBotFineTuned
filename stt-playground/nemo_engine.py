"""NeMo ASR for the playground (FastConformer / Conformer, RNN-T, CTC or hybrid): one model wrapper for the file tab
and a RealtimeSTT engine for the live tab.

Loading is the nemo training backend's own `restore_model` (backend/workers/backends/nemo/mlops_backend/asr.py), so
the playground loads a checkpoint exactly as training and eval do: a `.nemo` file, a dir holding one (the platform's
`nemo` export or a run's `final` checkpoint), or a Hugging Face repo holding a `.nemo` file. AI4Bharat's multilingual
IndicConformer is converted to Bangla-only on load there.

Decoding is greedy. A hybrid RNN-T/CTC model decodes with RNN-T by default; STT_NEMO_DECODER=ctc (or the UI's
decoder choice, which sets that variable before the live recorder spawns its engine process) switches to the CTC
head. The model runs the encoder and decoder directly on the clip (no NeMo `transcribe()` dataloader), so latency
is the model, not the data pipeline.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ENGINE = "playground_nemo"
SAMPLE_RATE = 16000
_BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend/workers/backends/nemo"

# Base models whose Hugging Face repo holds a .nemo file (checked by name: they have no config.json for AutoConfig).
NEMO_REPOS = {
    "SayedShaun/stt_bn_fastconformer_hybrid_large_v2",
    "ai4bharat/indicconformer_stt_bn_hybrid_ctc_rnnt_large",
    "bengaliAI/BanglaConformer",
}


def decoder_choice() -> str:
    d = os.environ.get("STT_NEMO_DECODER", "rnnt").strip().lower()
    return d if d in ("rnnt", "ctc") else "rnnt"


def nemo_file(ckpt: str) -> str | None:
    """The .nemo file `ckpt` points at (a .nemo path, or a dir holding exactly one), else None."""
    p = Path(ckpt.strip())
    if p.is_file() and p.suffix == ".nemo":
        return str(p)
    if p.is_dir():
        found = sorted(p.glob("*.nemo"))
        if found:
            return str(found[0])
    return None


def is_nemo(ckpt: str) -> bool:
    ckpt = ckpt.strip().rstrip("/")
    return nemo_file(ckpt) is not None or ckpt in NEMO_REPOS


def _restore(ckpt: str):
    """The training backend's restore_model (imported from its source dir, like qwen_engine does for the prompt)."""
    if str(_BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(_BACKEND_DIR))
    from mlops_backend.asr import quiet_numba, quiet_predictions, restore_model

    quiet_numba()
    model = restore_model(nemo_file(ckpt) or ckpt.strip())
    quiet_predictions(model)
    return model


def _text(h) -> str:
    """A decoder output item: a Hypothesis, a string, or a one-item n-best list."""
    if isinstance(h, (list, tuple)):
        h = h[0] if h else ""
    return h if isinstance(h, str) else str(getattr(h, "text", h) or "")


class NemoAsr:
    """A NeMo ASR model, transcribing one 16 kHz float32 clip at a time with greedy decoding."""

    def __init__(self, ckpt: str, device: str = "cuda", decoder: str | None = None):
        import torch

        self.model = _restore(ckpt).to(device).eval()
        self.device = torch.device(device)
        self.hybrid = hasattr(self.model, "ctc_decoder") and hasattr(self.model, "decoding")
        self.ctc_only = not hasattr(self.model, "joint")  # EncDecCTCModel(BPE): no RNN-T decoder
        want = decoder or decoder_choice()
        self.decoder = "ctc" if self.ctc_only or (self.hybrid and want == "ctc") else "rnnt"
        self.model.preprocessor.featurizer.dither = 0.0  # deterministic features, as in NeMo's transcribe()
        self.model.preprocessor.featurizer.pad_to = 0
        if self.hybrid:
            self.model.change_decoding_strategy(decoder_type=self.decoder, verbose=False)
        self.kind = type(self.model).__name__

    def transcribe(self, wav: np.ndarray) -> str:
        import torch

        wav = np.asarray(wav, dtype=np.float32)
        if wav.size == 0:
            return ""
        m = self.model
        with torch.inference_mode():
            sig = torch.from_numpy(wav).unsqueeze(0).to(self.device)
            length = torch.tensor([wav.shape[0]], device=self.device)
            enc, enc_len = m.forward(input_signal=sig, input_signal_length=length)
            if self.decoder == "ctc":
                log_probs = m.ctc_decoder(encoder_output=enc) if self.hybrid else m.decoder(encoder_output=enc)
                dec = m.ctc_decoding if self.hybrid else m.decoding
                out = dec.ctc_decoder_predictions_tensor(log_probs, decoder_lengths=enc_len, return_hypotheses=False)
            else:
                out = m.decoding.rnnt_decoder_predictions_tensor(encoder_output=enc, encoded_lengths=enc_len,
                                                                 return_hypotheses=False)
        if isinstance(out, tuple):  # older NeMo: (best, all)
            out = out[0]
        return _text(out[0]).strip() if out else ""


def register() -> None:
    """Add ENGINE to RealtimeSTT's engine registry (in whichever process imports this)."""
    from RealtimeSTT.transcription_engines import factory

    factory.ENGINE_CLASS_PATHS.setdefault(ENGINE, ("nemo_engine", "PlaygroundNemoEngine"))


def _engine_base():
    from RealtimeSTT.transcription_engines.base import BaseTranscriptionEngine

    return BaseTranscriptionEngine


class PlaygroundNemoEngine(_engine_base()):
    """RealtimeSTT engine over NemoAsr. The model is Bangla-only; `language` and beam size are ignored."""

    engine_name = ENGINE

    def __init__(self, config):
        super().__init__(config)
        self.asr = NemoAsr(config.model, config.device)

    def warmup(self, audio):
        self.asr.transcribe(audio)

    def transcribe(self, audio, language=None, use_prompt=True, **kwargs):
        from RealtimeSTT.transcription_engines.base import TranscriptionInfo, TranscriptionResult

        text = self.asr.transcribe(self._normalize_audio(audio))
        return TranscriptionResult(text=text, info=TranscriptionInfo(language="bn", language_probability=1.0))
