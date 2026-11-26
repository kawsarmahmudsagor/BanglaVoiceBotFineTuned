"""Separation stage: demucs htdemucs vocals stem, written as mono 44.1 kHz 16-bit WAV."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

BLOCK_S = 60
MODEL_SR = 44100
_MAX_SEGMENT_S = 7.8  # htdemucs training segment limit


def load_demucs():
    from demucs.pretrained import get_model

    model = get_model("htdemucs")
    return model.to("cuda").eval()


def separate(
    audio_wav: Path,
    out_wav: Path,
    model,
    segment_s: float,
    on_progress: Callable[[int, int], None],
) -> None:
    from demucs.apply import apply_model

    segment_s = min(segment_s, _MAX_SEGMENT_S)
    data, sr = sf.read(str(audio_wav), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != MODEL_SR:
        g = np.gcd(sr, MODEL_SR)
        mono = resample_poly(mono, MODEL_SR // g, sr // g).astype(np.float32)
    device = next(model.parameters()).device
    vocals_idx = model.sources.index("vocals")
    block = BLOCK_S * MODEL_SR
    total = max(1, -(-len(mono) // block))
    with sf.SoundFile(str(out_wav), "w", samplerate=MODEL_SR, channels=1, subtype="PCM_16") as out:
        for i in range(total):
            chunk = mono[i * block : (i + 1) * block]
            x = torch.from_numpy(chunk).unsqueeze(0).repeat(2, 1).unsqueeze(0)
            with torch.no_grad():
                y = apply_model(
                    model, x, device=device, shifts=1, split=True, overlap=0.25,
                    progress=False, segment=segment_s,
                )
            voc = y[0, vocals_idx].mean(dim=0).cpu().numpy()
            out.write(np.clip(voc, -1.0, 1.0))
            on_progress(i + 1, total)
