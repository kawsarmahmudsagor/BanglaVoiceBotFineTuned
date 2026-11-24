"""Build asr.zip and tts.zip for scripts/smoke_backends.sh: 16 sine-tone clips, 1.5 s each, plus metadata.csv.

Clips are 1.5 s so every converter keeps them (CosyVoice skips clips under 1 s). Nothing binary is committed; run it
with the backend venv: cd backend && uv run python ../scripts/smoke/make_audio_fixtures.py <out_dir>
"""
from __future__ import annotations

import csv
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
SECONDS = 1.5
TEXTS = [
    "আজ আকাশ মেঘলা।", "আমি ভাত খাই।", "ঢাকা বাংলাদেশের রাজধানী।", "সে বই পড়ছে।",
    "আপনার নাম কী?", "আমি বাজারে যাচ্ছি।", "আজ খুব গরম পড়েছে।", "নদীর পানি বেড়েছে।",
    "তোমার সঙ্গে দেখা হয়ে ভালো লাগল।", "কাল সকালে আমরা রওনা দেব।", "চা খেতে চাও?",
    "এই রাস্তাটি অনেক লম্বা।", "শিশুরা মাঠে খেলছে।", "দরজাটা বন্ধ করে দাও।",
    "আমার ফোনটা কোথায়?", "বৃষ্টি থেমে গেছে।",
]


def tone(i: int) -> np.ndarray:
    t = np.arange(int(SR * SECONDS)) / SR
    x = 0.3 * np.sin(2 * np.pi * (200 + 40 * i) * t)
    fade = int(0.05 * SR)
    x[:fade] *= np.linspace(0, 1, fade)
    x[-fade:] *= np.linspace(1, 0, fade)
    return x.astype(np.float32)


def build(path: Path, task: str) -> None:
    rows = [("file", "text") + (("speaker",) if task == "tts" else ())]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, text in enumerate(TEXTS):
            name = f"clips/{i:03d}.wav"
            buf = io.BytesIO()
            sf.write(buf, tone(i), SR, format="WAV", subtype="PCM_16")
            zf.writestr(name, buf.getvalue())
            rows.append((name, text) + (("spk1",) if task == "tts" else ()))
        out = io.StringIO(newline="")
        csv.writer(out, lineterminator="\n").writerows(rows)
        zf.writestr("metadata.csv", out.getvalue().encode("utf-8"))


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    out.mkdir(parents=True, exist_ok=True)
    for task in ("asr", "tts"):
        build(out / f"{task}.zip", task)
        print(out / f"{task}.zip")


if __name__ == "__main__":
    main()
