"""Eval loops shared by all training backends; backends inject only the model call.

No numpy at import time: audio helpers are imported lazily inside run_tts_eval.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from app.core.contract import (
    DatasetRef, EvalResults, Phase, RunConfig, RunDir, append_eval_sample, reset_eval_samples, write_eval_results,
)
from app.core.metrics import bangla_script_share, cer, has_emoji, has_markdown, sample_wer, wer
from app.core.text import bn_normalize

if TYPE_CHECKING:
    import numpy as np

    from app.core.backend_kit import Reporter

_SLICE_COLS = ("region", "channel", "noise")
ASR_MAX_SAMPLES = 2000  # ASR eval cap when hparams.eval.max_samples is unset (LLM: 200, TTS: 20)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def run_llm_eval(rd: RunDir, rep: "Reporter", rows: list[dict], generate: Callable[[list[dict]], str], *,
                 loss: float | None, prefix: str = "", max_samples: int = 200) -> dict[str, float]:
    if prefix == "":
        reset_eval_samples(rd)
    items = []
    for row in rows:
        msgs = row["messages"]
        last = max((i for i, m in enumerate(msgs) if m.get("role") == "assistant"), default=None)
        if last is None:
            continue
        items.append((row.get("sample_id"), msgs[:last], msgs[last]["content"]))
        if len(items) >= max_samples:
            break
    replies: list[str] = []
    for i, (sid, prompt, ref) in enumerate(items, 1):
        reply = generate(prompt)
        replies.append(reply)
        if prefix == "":
            append_eval_sample(rd, {"sample_id": sid, "prompt": prompt, "reference": ref, "reply": reply})
        rep.status(Phase.eval, step=i, total_steps=len(items))
    m: dict[str, float] = {}
    if loss is not None:
        m["eval_loss"] = loss
        m["perplexity"] = math.exp(loss)
    m["bangla_script_share"] = _mean([bangla_script_share(r) for r in replies])
    m["markdown_rate"] = _mean([1.0 if has_markdown(r) else 0.0 for r in replies])
    m["emoji_rate"] = _mean([1.0 if has_emoji(r) else 0.0 for r in replies])
    m["avg_reply_chars"] = _mean([float(len(r)) for r in replies])
    return {prefix + k: v for k, v in m.items()}


def run_asr_eval(rd: RunDir, rep: "Reporter", rows: list[dict], transcribe: Callable[[list[str]], list[str]], *,
                 prefix: str = "", batch_size: int = 16, max_samples: int | None = None) -> dict[str, float]:
    rows = rows[:max_samples if max_samples is not None else ASR_MAX_SAMPLES]
    if prefix == "":
        reset_eval_samples(rd)
    refs: list[str] = []
    hyps: list[str] = []
    elapsed = 0.0
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        t0 = time.perf_counter()
        out = transcribe([r["audio_path"] for r in batch])
        elapsed += time.perf_counter() - t0
        if len(out) != len(batch):
            raise ValueError(f"transcribe returned {len(out)} results for {len(batch)} inputs")
        for r, h in zip(batch, out):
            ref_n, hyp_n = bn_normalize(r["text"], mode="asr"), bn_normalize(h, mode="asr")
            refs.append(ref_n)
            hyps.append(hyp_n)
            if prefix == "":
                append_eval_sample(rd, {"sample_id": r.get("sample_id"), "reference": ref_n,
                                        "hypothesis": hyp_n, "wer": sample_wer(ref_n, hyp_n)})
        rep.status(Phase.eval, step=min(i + batch_size, len(rows)), total_steps=len(rows))
    total_audio = sum(float(r.get("duration_s") or 0.0) for r in rows)
    m: dict[str, float] = {"wer": wer(refs, hyps), "cer": cer(refs, hyps)}
    for col in _SLICE_COLS:
        groups: dict[str, list[int]] = {}
        for idx, r in enumerate(rows):
            v = r.get(col)
            if v is not None and v != "":
                groups.setdefault(str(v), []).append(idx)
        for val, idxs in sorted(groups.items()):
            m[f"wer/{col}={val}"] = wer([refs[j] for j in idxs], [hyps[j] for j in idxs])
    if total_audio > 0:
        m["rtf"] = elapsed / total_audio
    return {prefix + k: v for k, v in m.items()}


def run_tts_eval(rd: RunDir, rep: "Reporter", rows: list[dict],
                 synthesize: Callable[[str, dict], tuple["np.ndarray", int]], *,
                 loss: float | None, prefix: str = "", max_samples: int = 20) -> dict[str, float]:
    from app.core.backend_kit.audio import write_wav

    if prefix == "":
        reset_eval_samples(rd)
    rows = rows[:max_samples]
    wall = 0.0
    audio_s = 0.0
    for i, row in enumerate(rows, 1):
        t0 = time.perf_counter()
        audio, sr = synthesize(row["text"], row)
        wall += time.perf_counter() - t0
        audio_s += len(audio) / sr
        if prefix == "":
            write_wav(rd.eval_audio_dir / f"{row['sample_id']}.wav", audio, sr)
            append_eval_sample(rd, {"sample_id": row["sample_id"], "text": row["text"],
                                    "audio": f"audio/{row['sample_id']}.wav"})
        rep.status(Phase.eval, step=i, total_steps=len(rows))
    m: dict[str, float] = {}
    if loss is not None:
        m["eval_loss"] = loss
    if audio_s > 0:
        m["rtf"] = wall / audio_s
    return {prefix + k: v for k, v in m.items()}


def baseline_metrics(cfg: RunConfig, ds: DatasetRef, eval_key: dict,
                     compute: Callable[[], dict[str, float]]) -> dict[str, float]:
    blob = json.dumps({"base_model": cfg.base_model, "dataset_hash": ds.hash, "eval_key": eval_key,
                       "split": ds.split.model_dump() if ds.split else None},
                      sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    key = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
    path = Path(cfg.prepared_root) / "_baselines" / f"{key}.json"
    try:
        cached = json.loads(path.read_bytes())
        if isinstance(cached, dict):
            return cached
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
        pass
    result = {f"base/{k}": v for k, v in compute().items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(json.dumps(result, ensure_ascii=False).encode("utf-8"))
    os.replace(tmp, path)
    return result


def merge_results(rd: RunDir, *parts: dict[str, float]) -> None:
    metrics: dict[str, float] = {}
    for p in parts:
        metrics.update(p)
    samples = rd.eval_samples_path.exists() or (rd.eval_audio_dir.exists() and any(rd.eval_audio_dir.iterdir()))
    write_eval_results(rd, EvalResults(metrics=metrics, samples_dir="eval" if samples else None))
