"""CosyVoice 3 speech-tokenizer round-trip gate (spec 2026-11-24 §12).

Question: can `speech_tokenizer_v3`, which never saw Bangla, represent Bangla sounds?
If it cannot, no LLM fine-tuning can fix it, because the LLM only predicts tokens the
tokenizer offers.

`run`
  1. Pick N clips whose text contains both aspirated and retroflex consonants.
  2. Tokenize each clip with speech_tokenizer_v3 and resynthesize it with the frozen
     flow model and vocoder (upstream `CosyVoice3.inference_vc`; no LLM runs).
     A separate prompt clip supplies the timbre. Using the clip itself as the prompt
     would let the flow model copy content from the prompt mel and hide token losses.
  3. Transcribe original and resynthesized audio with the same STT model and score
     CER against the reference text (both through bn_normalize(mode="asr")).
  4. Write results.csv, summary.json and report.html (side-by-side players plus a
     listening checkbox per clip; "Download verdicts" saves verdicts.json).

`finalize`
  Reads verdicts.json and writes gate.json with the CER and listening criteria.

Heavy imports (torch, librosa, transformers, CosyVoice) happen inside functions, so the
selection, scoring and report code is importable and testable on CPU.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import subprocess
import sys
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

from app.core.metrics import cer
from app.core.text import bn_normalize

ASPIRATED = frozenset("খঘছঝঠঢথধফভ")
RETROFLEX = frozenset("টঠডঢণ")  # ড় and ঢ় are ড/ঢ + nukta after NFC, so they count too
TOKENIZER_MAX_S = 30.0  # upstream _extract_speech_token asserts <= 30 s
MAX_CER_DELTA = 0.05  # resynth mean CER may exceed original mean CER by at most 5 points
MIN_PRESERVED_SHARE = 0.9  # 27 of 30 clips judged preserved by a listener

DEFAULT_COSYVOICE_ROOT = "var/vendor/CosyVoice"
DEFAULT_MODEL_DIR = "var/models/Fun-CosyVoice3-0.5B"
DEFAULT_STT_MODEL = "mozilla-ai/whisper-large-v3-turbo-bn"


@dataclass
class Clip:
    id: str
    file: str
    text: str
    duration: float


# ---------------------------------------------------------------- selection

def has_contrasts(text: str) -> bool:
    chars = set(unicodedata.normalize("NFC", text))
    return bool(chars & ASPIRATED) and bool(chars & RETROFLEX)


def _order_key(file: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{file}".encode("utf-8")).hexdigest()


def read_metadata(src: Path) -> list[dict]:
    meta = src / "metadata.csv"
    if not meta.is_file():
        raise SystemExit(f"{meta} not found: --src must be a dataset directory with metadata.csv")
    with open(meta, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for col in ("file", "text"):
        if rows and col not in rows[0]:
            raise SystemExit(f"{meta} has no '{col}' column")
    return rows


def pick_prompt(
    rows: list[dict],
    duration_of: Callable[[str], float | None],
    *,
    min_s: float = 5.0,
    max_s: float = 15.0,
    salt: str = "",
) -> str | None:
    """First clip in hash order with a duration usable as a speaker prompt."""
    for r in sorted(rows, key=lambda r: _order_key(r["file"], "prompt" + salt)):
        d = duration_of(r["file"])
        if d is not None and min_s <= d <= max_s:
            return r["file"]
    return None


def select_clips(
    rows: Iterable[dict],
    n: int,
    duration_of: Callable[[str], float | None],
    *,
    min_s: float = 3.0,
    max_s: float = 15.0,
    exclude: Iterable[str] = (),
    salt: str = "",
) -> list[Clip]:
    """Deterministic pick of n clips with aspirated + retroflex text and a duration in range.

    Rows are visited in hash order of their file name, so the result does not depend on
    metadata order. Durations are read only for rows whose text qualifies.
    """
    if max_s > TOKENIZER_MAX_S:
        raise ValueError(f"max_s {max_s} exceeds the tokenizer limit of {TOKENIZER_MAX_S} s")
    skip = set(exclude)
    picked: list[Clip] = []
    for r in sorted(rows, key=lambda r: _order_key(r["file"], salt)):
        if r["file"] in skip or not has_contrasts(r["text"]):
            continue
        d = duration_of(r["file"])
        if d is None or not (min_s <= d <= max_s):
            continue
        picked.append(Clip(id=f"{len(picked):02d}", file=r["file"], text=r["text"], duration=round(d, 3)))
        if len(picked) == n:
            break
    return picked


# ---------------------------------------------------------------- scoring

def clip_cer(ref: str, hyp: str) -> float:
    return cer([bn_normalize(ref, mode="asr")], [bn_normalize(hyp, mode="asr")])


def cer_gate(results: list[dict], max_delta: float = MAX_CER_DELTA) -> dict:
    n = len(results)
    if n == 0:
        raise ValueError("no results")
    orig = sum(r["cer_orig"] for r in results) / n
    resynth = sum(r["cer_resynth"] for r in results) / n
    delta = round(resynth - orig, 4)  # rounded before comparing, so float noise cannot fail a boundary value
    return {
        "mean_cer_orig": round(orig, 4),
        "mean_cer_resynth": round(resynth, 4),
        "delta": delta,
        "max_delta": max_delta,
        "passed": delta <= max_delta,
    }


def listening_gate(verdicts: dict[str, bool], clip_ids: list[str], min_share: float = MIN_PRESERVED_SHARE) -> dict:
    missing = [i for i in clip_ids if i not in verdicts]
    if missing:
        raise ValueError(f"no listening verdict for clips: {', '.join(missing)}")
    preserved = sum(1 for i in clip_ids if verdicts[i])
    needed = math.ceil(min_share * len(clip_ids))
    return {"preserved": preserved, "total": len(clip_ids), "needed": needed, "passed": preserved >= needed}


# ---------------------------------------------------------------- report

_REPORT_CSS = """
:root { --bg:#fff; --fg:#1a1a1a; --muted:#666; --line:#ddd; --ok:#1a7f37; --bad:#c62828; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#141414; --fg:#eee; --muted:#999; --line:#333; --ok:#4caf50; --bad:#ef5350; }
}
body { background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, sans-serif; margin:24px auto; max-width:1100px; padding:0 16px; }
h1 { font-size:1.3rem; } .muted { color:var(--muted); }
.ok { color:var(--ok); font-weight:600; } .bad { color:var(--bad); font-weight:600; }
table { border-collapse:collapse; width:100%; } td, th { border-bottom:1px solid var(--line); padding:8px 6px; vertical-align:top; text-align:left; }
td.text { font-size:1.05rem; max-width:340px; } audio { width:220px; height:32px; }
.bar { position:sticky; top:0; background:var(--bg); padding:10px 0; border-bottom:1px solid var(--line); display:flex; gap:16px; align-items:center; flex-wrap:wrap; }
button { font:inherit; padding:6px 12px; cursor:pointer; }
"""

_REPORT_JS = """
const KEY = 'roundtrip-verdicts:' + document.body.dataset.run;
const boxes = [...document.querySelectorAll('input[data-clip]')];
function load() { try { return JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (e) { return {}; } }
function save(v) { try { localStorage.setItem(KEY, JSON.stringify(v)); } catch (e) {} }
function current() { const v = {}; boxes.forEach(b => v[b.dataset.clip] = b.checked); return v; }
function update() {
  const n = boxes.filter(b => b.checked).length, need = +document.body.dataset.needed;
  const el = document.getElementById('count');
  el.textContent = n + ' / ' + boxes.length + ' preserved (need ' + need + ')';
  el.className = n >= need ? 'ok' : 'bad';
  save(current());
}
const saved = load(); boxes.forEach(b => { if (b.dataset.clip in saved) b.checked = saved[b.dataset.clip]; b.addEventListener('change', update); });
document.getElementById('dl').addEventListener('click', () => {
  const blob = new Blob([JSON.stringify({preserved: current()}, null, 2)], {type: 'application/json'});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'verdicts.json'; a.click();
});
update();
"""


def render_report(summary: dict) -> str:
    gate = summary["cer_gate"]
    needed = math.ceil(MIN_PRESERVED_SHARE * len(summary["clips"]))
    verdict = '<span class="ok">pass</span>' if gate["passed"] else '<span class="bad">fail</span>'
    rows = []
    for c in summary["clips"]:
        rows.append(
            "<tr>"
            f"<td>{html.escape(c['id'])}</td>"
            f"<td class=\"text\">{html.escape(c['text'])}<div class=\"muted\">{html.escape(c['file'])} · {c['duration']:.1f} s</div></td>"
            f"<td><audio controls preload=\"none\" src=\"orig/{c['id']}.wav\"></audio>"
            f"<div class=\"muted\">CER {c['cer_orig']:.3f}: {html.escape(c['hyp_orig'])}</div></td>"
            f"<td><audio controls preload=\"none\" src=\"resynth/{c['id']}.wav\"></audio>"
            f"<div class=\"muted\">CER {c['cer_resynth']:.3f}: {html.escape(c['hyp_resynth'])}</div></td>"
            f"<td><label><input type=\"checkbox\" data-clip=\"{html.escape(c['id'])}\"> preserved</label></td>"
            "</tr>"
        )
    run_id = html.escape(summary["run_id"])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CosyVoice Round-trip</title><style>{_REPORT_CSS}</style></head>
<body data-run="{run_id}" data-needed="{needed}">
<h1>CosyVoice 3 tokenizer round-trip</h1>
<p class="muted">Run {run_id} · model {html.escape(summary['model_dir'])} · CosyVoice {html.escape(summary['cosyvoice_commit'][:12])} · STT {html.escape(summary['stt_model'])} · prompt <a href="prompt.wav">{html.escape(summary['prompt_file'])}</a></p>
<p>CER: original {gate['mean_cer_orig']:.3f}, resynthesized {gate['mean_cer_resynth']:.3f}, delta {gate['delta']:+.3f} (max {gate['max_delta']:.2f}) · {verdict}</p>
<p class="muted">For each clip, listen to both versions. Tick "preserved" only if aspirated (খ ঘ ছ ঝ ঠ ঢ থ ধ ফ ভ) and retroflex (ট ঠ ড ঢ ণ ড়) consonants sound right in the resynthesized audio. Then download the verdicts and run <code>finalize</code>.</p>
<div class="bar"><span id="count"></span><button id="dl" type="button">Download verdicts.json</button></div>
<table><thead><tr><th>#</th><th>Text</th><th>Original</th><th>Resynthesized</th><th>Verdict</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<script>{_REPORT_JS}</script>
</body></html>
"""


# ---------------------------------------------------------------- models (GPU, VM only)

def _duration_reader(src: Path) -> Callable[[str], float | None]:
    import soundfile as sf

    def duration_of(file: str) -> float | None:
        try:
            return float(sf.info(str(src / file)).duration)
        except Exception:
            return None

    return duration_of


def _git_commit(root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def load_cosyvoice(root: Path, model_dir: Path):
    root = root.resolve()
    if not (root / "cosyvoice" / "cli" / "cosyvoice.py").is_file():
        raise SystemExit(f"CosyVoice source not found under {root}: run tools/dataprep/scripts/setup_cosyvoice.sh")
    if not (model_dir / "cosyvoice3.yaml").is_file():
        raise SystemExit(f"{model_dir} has no cosyvoice3.yaml: run tools/dataprep/scripts/setup_cosyvoice.sh")
    for p in (root, root / "third_party" / "Matcha-TTS"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from cosyvoice.cli.cosyvoice import CosyVoice3  # noqa: PLC0415  (upstream repo, on sys.path above)

    return CosyVoice3(str(model_dir.resolve()))


def resynthesize(model, source_wav: Path, prompt_wav: Path):
    """Speech tokens of source_wav -> flow + vocoder, timbre from prompt_wav. Returns float32 mono."""
    import torch

    chunks = [out["tts_speech"] for out in model.inference_vc(str(source_wav), str(prompt_wav), stream=False)]
    return torch.cat(chunks, dim=1).squeeze(0).float().cpu().numpy()


def load_asr_pipeline(name: str):
    """A transformers ASR pipeline (fp16 on GPU). Shared with dataprep's `transcribe` labeller."""
    import torch
    import transformers
    from packaging.version import Version

    cuda = torch.cuda.is_available()
    # transformers >= 4.56 renamed torch_dtype to dtype (and 5.x warns on the old name); cosyvoice pins 4.51.3.
    dtype_key = "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"
    return transformers.pipeline(
        "automatic-speech-recognition",
        model=name,
        device="cuda:0" if cuda else "cpu",
        **{dtype_key: torch.float16 if cuda else torch.float32},
    )


def load_stt(name: str, language: str | None):
    asr = load_asr_pipeline(name)
    kwargs = {"task": "transcribe", **({"language": language} if language else {})}

    def transcribe(path: Path) -> str:
        import librosa

        y, _ = librosa.load(str(path), sr=16000, mono=True)
        return asr({"raw": y, "sampling_rate": 16000}, generate_kwargs=kwargs)["text"].strip()

    return transcribe


def _free_gpu() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- commands

def cmd_run(args: argparse.Namespace) -> int:
    import librosa
    import soundfile as sf

    src: Path = args.src
    out: Path = args.out
    rows = read_metadata(src)
    duration_of = _duration_reader(src)

    if args.prompt:
        prompt_src, prompt_file = args.prompt, str(args.prompt)
    else:
        prompt_file = pick_prompt(rows, duration_of, salt=args.salt)
        if prompt_file is None:
            raise SystemExit("no clip of 5-15 s found to use as the speaker prompt; pass --prompt")
        prompt_src = src / prompt_file
    exclude = [prompt_file] if not args.prompt else []

    clips = select_clips(rows, args.n, duration_of, min_s=args.min_s, max_s=args.max_s, exclude=exclude, salt=args.salt)
    if len(clips) < args.n:
        raise SystemExit(f"only {len(clips)} clips qualify (aspirated + retroflex, {args.min_s}-{args.max_s} s); need {args.n}")

    (out / "orig").mkdir(parents=True, exist_ok=True)
    (out / "resynth").mkdir(parents=True, exist_ok=True)

    # Originals at their source sample rate: the prompt mel is extracted at 24 kHz upstream,
    # so downsampling first would dull the prompt timbre.
    y, sr = librosa.load(str(prompt_src), sr=None, mono=True)
    sf.write(out / "prompt.wav", y, sr)
    for c in clips:
        y, sr = librosa.load(str(src / c.file), sr=None, mono=True)
        sf.write(out / "orig" / f"{c.id}.wav", y, sr)

    started = time.time()
    model = load_cosyvoice(args.cosyvoice_root, args.model_dir)
    sample_rate = model.sample_rate
    for i, c in enumerate(clips, 1):
        audio = resynthesize(model, out / "orig" / f"{c.id}.wav", out / "prompt.wav")
        sf.write(out / "resynth" / f"{c.id}.wav", audio, sample_rate)
        print(f"[resynth {i}/{len(clips)}] {c.file}", flush=True)
    del model
    _free_gpu()
    synth_s = time.time() - started

    transcribe = load_stt(args.stt_model, None if args.stt_language == "none" else args.stt_language)
    results = []
    for i, c in enumerate(clips, 1):
        hyp_orig = transcribe(out / "orig" / f"{c.id}.wav")
        hyp_resynth = transcribe(out / "resynth" / f"{c.id}.wav")
        results.append({
            **asdict(c),
            "hyp_orig": hyp_orig,
            "hyp_resynth": hyp_resynth,
            "cer_orig": round(clip_cer(c.text, hyp_orig), 4),
            "cer_resynth": round(clip_cer(c.text, hyp_resynth), 4),
        })
        print(f"[stt {i}/{len(clips)}] cer {results[-1]['cer_orig']:.3f} -> {results[-1]['cer_resynth']:.3f}", flush=True)

    summary = {
        "run_id": time.strftime("%Y%m%d-%H%M%S"),
        "src": str(src),
        "model_dir": str(args.model_dir),
        "cosyvoice_commit": _git_commit(args.cosyvoice_root),
        "stt_model": args.stt_model,
        "stt_language": args.stt_language,
        "prompt_file": prompt_file,
        "selection": {"n": args.n, "min_s": args.min_s, "max_s": args.max_s, "salt": args.salt},
        "resynth_sample_rate": sample_rate,
        "resynth_seconds": round(synth_s, 1),
        "cer_gate": cer_gate(results),
        "clips": results,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    (out / "report.html").write_text(render_report(summary), encoding="utf-8")

    g = summary["cer_gate"]
    print(f"CER gate: orig {g['mean_cer_orig']:.3f}, resynth {g['mean_cer_resynth']:.3f}, "
          f"delta {g['delta']:+.3f} -> {'PASS' if g['passed'] else 'FAIL'}")
    print(f"Next: open {out / 'report.html'}, judge every clip, download verdicts.json, then run finalize.")
    return 0


def finalize(out: Path, verdicts_path: Path) -> dict:
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    raw = json.loads(verdicts_path.read_text(encoding="utf-8"))
    verdicts = {str(k): bool(v) for k, v in raw.get("preserved", raw).items()}
    ids = [c["id"] for c in summary["clips"]]
    listening = listening_gate(verdicts, ids)
    gate = {
        "run_id": summary["run_id"],
        "cer": summary["cer_gate"],
        "listening": listening,
        "passed": summary["cer_gate"]["passed"] and listening["passed"],
        "failed_clips": [i for i in ids if not verdicts[i]],
        "decided_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out / "gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    return gate


def cmd_finalize(args: argparse.Namespace) -> int:
    verdicts = args.verdicts or args.out / "verdicts.json"
    try:
        gate = finalize(args.out, verdicts)
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(str(e)) from e
    lis = gate["listening"]
    print(f"Listening: {lis['preserved']}/{lis['total']} preserved (need {lis['needed']}) -> "
          f"{'PASS' if lis['passed'] else 'FAIL'}")
    print(f"CER: delta {gate['cer']['delta']:+.3f} -> {'PASS' if gate['cer']['passed'] else 'FAIL'}")
    print("GATE PASSED: CosyVoice 3 stays the main TTS track." if gate["passed"]
          else "GATE FAILED: switch the main TTS track to F5 from scratch (spec §12.3).")
    return 0 if gate["passed"] else 1


def add_parser(commands) -> None:
    p = commands.add_parser("roundtrip-cosyvoice", help="CosyVoice 3 speech-tokenizer round-trip gate (spec §12)")
    sub = p.add_subparsers(dest="action", required=True)

    run = sub.add_parser("run", help="resynthesize clips through the speech tokenizer and score them")
    run.add_argument("--src", type=Path, required=True, help="dataset dir with metadata.csv (file, text)")
    run.add_argument("--out", type=Path, required=True, help="output dir for audio, results and report")
    run.add_argument("--prompt", type=Path, help="speaker prompt clip (default: a 5-15 s clip from --src, excluded from the test set)")
    run.add_argument("--n", type=int, default=30)
    run.add_argument("--min-s", type=float, default=3.0)
    run.add_argument("--max-s", type=float, default=15.0)
    run.add_argument("--salt", default="", help="change to draw a different deterministic sample")
    run.add_argument("--cosyvoice-root", type=Path, default=Path(DEFAULT_COSYVOICE_ROOT))
    run.add_argument("--model-dir", type=Path, default=Path(DEFAULT_MODEL_DIR))
    run.add_argument("--stt-model", default=DEFAULT_STT_MODEL)
    run.add_argument("--stt-language", default="bengali", help="Whisper language, or 'none' to let the model decide")
    run.set_defaults(func=cmd_run)

    fin = sub.add_parser("finalize", help="combine listening verdicts with the CER result into gate.json")
    fin.add_argument("--out", type=Path, required=True, help="the run's output dir")
    fin.add_argument("--verdicts", type=Path, help="verdicts.json from report.html (default: <out>/verdicts.json)")
    fin.set_defaults(func=cmd_finalize)
