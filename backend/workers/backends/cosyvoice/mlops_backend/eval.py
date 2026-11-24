"""CosyVoice 3 eval: `eval_loss_<component>` (upstream cross-validation loss on the val split, llm and flow; hifigan has
no meaningful loss without its discriminator) and zero-shot synthesis of up to 20 sentences with rtf, plus the same
metrics (prefix `base/`) for the untouched base model (or the init checkpoint). torch / cosyvoice are imported inside
functions only.

Synthesis follows upstream's CosyVoice3 zero-shot example: `prompt_text = INSTRUCT + reference text`, `tts_text` is the
plain sentence (the instruct prefix belongs to the prompt only; the training sequence is instruct + text, so at inference
the LLM sees instruct + ref text + text). The upstream text frontend (Chinese/English normalizers and splitter) is
switched off (`text_frontend=False`) because it would mangle Bangla; Bangla text is normalized with bn_normalize and
split on the danda here, one sentence per call.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from app.core.backend_kit import eval_opts
from app.core.backend_kit.runutil import eval_ds, eval_rows, free_gpu
from app.core.backend_kit.whisper import asr_baseline_key
from app.core.contract import RunConfig, RunDir
from app.core.evaluation import baseline_metrics, merge_results, run_tts_eval
from app.core.text import bn_normalize
from app.core.tts_refs import DEFAULT_SPEAKER

from .compat import ensure_deepspeed_stub, use_offline_features
from .prepare.cosyvoice import INSTRUCT, _safe
from .train import (
    CKPT_FORMAT, WEIGHT_FILE, base_model_dir, check_init, load_configs, load_weights, patched_yaml, prepared_dir,
    resolve_components,
)

EVAL_CAP = 20
SYNTH_SEED = 0
CV_COMPONENTS = ("llm", "flow")


# ---- pure helpers (light-testable) -------------------------------------------------------------------------------

def split_sentences(text: str) -> list[str]:
    """Split after each danda (kept on its sentence)."""
    out, cur = [], ""
    for c in text:
        cur += c
        if c == "।":
            out.append(cur.strip())
            cur = ""
    if cur.strip():
        out.append(cur.strip())
    return [x for x in out if x]


def pick_ref(refs: Path, speaker: str | None) -> Path:
    """The reference wav of a speaker, else `default`, else the first available."""
    refs = Path(refs)
    for name in (_safe(speaker) if speaker else None, DEFAULT_SPEAKER):
        if name and (refs / f"{name}.wav").is_file():
            return refs / f"{name}.wav"
    found = sorted(refs.glob("*.wav")) if refs.is_dir() else []
    if not found:
        raise FileNotFoundError(f"no reference clips in {refs}")
    return found[0]


def cv_components(hp: dict) -> list[str]:
    return [c for c in resolve_components(hp) if c in CV_COMPONENTS]


def cosyvoice_baseline_key(cfg: RunConfig, max_samples: int | None) -> dict:
    key = asr_baseline_key(cfg, max_samples)  # max_samples + init checkpoint
    key["components"] = cv_components(cfg.hparams)
    return key


def _require_final(rd: RunDir) -> Path:
    final = rd.checkpoints_dir / "final"
    if not (final / ".complete").exists():
        raise RuntimeError(f"no completed final checkpoint at {final}")
    return final


def baseline_model_dir(rd: RunDir, cfg: RunConfig) -> Path:
    """The model the baseline is measured on: the init checkpoint (a complete cosyvoice dir) when set, else the base."""
    if cfg.init_from is not None:
        check_init(rd, cfg)
        if not (rd.init_dir / "cosyvoice3.yaml").is_file():
            raise ValueError("init checkpoint is not a complete CosyVoice model dir (no cosyvoice3.yaml): "
                             "use a run's final checkpoint")
        return rd.init_dir
    return base_model_dir(cfg)


# ---- eval loss (upstream cv pass) --------------------------------------------------------------------------------

def eval_list(rd: RunDir, cfg: RunConfig, work: Path) -> Path:
    """data.list of the eval rows' splits (eval dataset's own splits, else the train dataset's val)."""
    role = "eval" if cfg.datasets.get("eval") is not None else "train"
    prepared = prepared_dir(rd, role)
    _, splits = eval_ds(cfg)
    lines: list[str] = []
    for s in splits:
        lst = prepared / s / "data.list"
        if lst.is_file():
            lines += [ln for ln in lst.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError(f"no prepared data.list for eval splits {splits} in {prepared}")
    out = work / "eval.data.list"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def eval_loss(rd: RunDir, cfg: RunConfig, component: str, weights: Path, base_dir: Path, work: Path) -> float:
    """Mean per-utterance loss of one component over the eval list, with upstream's batch_forward (no dist needed)."""
    import torch
    from argparse import Namespace

    ensure_deepspeed_stub()
    from cosyvoice.utils.train_utils import batch_forward, init_dataset_and_dataloader

    use_offline_features()  # see compat.use_offline_features
    work.mkdir(parents=True, exist_ok=True)
    yaml_path = patched_yaml(base_dir, work, cfg.hparams)
    configs = load_configs(component, yaml_path, base_dir)
    lst = eval_list(rd, cfg, work)
    args = Namespace(train_data=str(lst), cv_data=str(lst), pin_memory=False, num_workers=2, prefetch=10)
    _, _, _, loader = init_dataset_and_dataloader(args, configs, False, False)
    model = configs[component]
    load_weights(model, component, weights)
    model = model.cuda().eval()
    info = {"dtype": "bf16", "train_engine": "torch_ddp"}
    total, n = 0.0, 0
    with torch.inference_mode():
        for batch in loader:
            info = batch_forward(model, batch, None, info)
            k = len(batch["utts"])
            total += float(info["loss_dict"]["loss"].mean()) * k
            n += k
    del model
    free_gpu()
    if n == 0:
        raise RuntimeError("eval split produced no usable utterances")
    return total / n


def eval_losses(rd: RunDir, cfg: RunConfig, model_dir: Path, base_dir: Path, work: Path, prefix: str = "") -> dict[str, float]:
    out = {}
    for comp in cv_components(cfg.hparams):
        weights = Path(model_dir) / WEIGHT_FILE[comp]
        out[f"{prefix}eval_loss_{comp}"] = eval_loss(rd, cfg, comp, weights, base_dir, work)
    return out


# ---- synthesis ---------------------------------------------------------------------------------------------------

def make_synthesizer(model, refs: Path):
    """synthesize(text, row) -> (audio, sample_rate) with CosyVoice3.inference_zero_shot."""
    cache: dict[Path, str] = {}

    def synthesize(text: str, row: dict):
        import numpy as np
        import torch

        wav = pick_ref(refs, row.get("speaker"))
        if wav not in cache:
            cache[wav] = INSTRUCT + wav.with_suffix(".txt").read_text(encoding="utf-8").strip()
        torch.manual_seed(SYNTH_SEED)
        chunks = []
        for sent in split_sentences(bn_normalize(text, mode="tts")) or [text]:
            for out in model.inference_zero_shot(sent, cache[wav], str(wav), stream=False, text_frontend=False):
                chunks.append(out["tts_speech"].squeeze(0).float().cpu().numpy())
        return np.concatenate(chunks), int(model.sample_rate)

    return synthesize


def _tts_metrics(rd: RunDir, rep, rows, model_dir: Path, refs: Path, prefix: str, cap: int) -> dict[str, float]:
    from cosyvoice.cli.cosyvoice import CosyVoice3

    model = CosyVoice3(str(model_dir))
    try:
        return run_tts_eval(rd, rep, rows, make_synthesizer(model, refs), loss=None, prefix=prefix, max_samples=cap)
    finally:
        del model
        free_gpu()


def eval_cosyvoice(rd: RunDir, cfg: RunConfig, rep) -> int:
    final = _require_final(rd)
    do_baseline, max_samples = eval_opts(cfg)
    cap = EVAL_CAP if max_samples is None else min(EVAL_CAP, int(max_samples))
    rows = eval_rows(cfg)
    role = "eval" if cfg.datasets.get("eval") is not None else "train"
    refs = prepared_dir(rd, role) / "refs"
    work = rd.path / "work" / "cosyvoice-eval"
    shutil.rmtree(work, ignore_errors=True)
    base_dir = base_model_dir(cfg)  # tokenizer / config files for the cv dataloader

    tuned = eval_losses(rd, cfg, final, base_dir, work)
    tuned.update(_tts_metrics(rd, rep, rows, final, refs, "", cap))

    base: dict[str, float] = {}
    if do_baseline:
        ds, _ = eval_ds(cfg)

        def compute() -> dict[str, float]:
            mdir = baseline_model_dir(rd, cfg)
            res = eval_losses(rd, cfg, mdir, base_dir, work)
            res.update({k.removeprefix("base/"): v for k, v in _tts_metrics(rd, rep, rows, mdir, refs, "base/", cap).items()})
            return res

        base = baseline_metrics(cfg, ds, cosyvoice_baseline_key(cfg, max_samples), compute)
    shutil.rmtree(work, ignore_errors=True)
    merge_results(rd, tuned, base)
    return 0
