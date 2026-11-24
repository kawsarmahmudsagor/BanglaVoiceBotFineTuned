"""F5-TTS (IndicF5) fine-tune, eval and export. torch, f5_tts, datasets, safetensors and huggingface_hub are
imported inside functions only.

Training uses f5_tts' own Trainer on each prepared split built directly from `<split>/raw.arrow` (F5's
`load_dataset("CustomDataset")` hard-codes a data dir). F5 writes `model_last.pt` into a working dir
(`<run>/work/f5_ckpt`, which is also where a resume is staged: `rd.init_dir` is read-only); every save is then
published as `checkpoints/update-<n>/model_last.pt` + vocab.txt + init meta (`f5-pt`) + `.complete`.

Text embedding: F5's `text_embed` has vocab_size + 1 rows (row 0 is the filler), so new characters appended to the
vocab become new rows at the end, initialised from the mean embedding.
"""
from __future__ import annotations

import math
import os
import shutil
import sys
from pathlib import Path, PurePosixPath

from app.core.backend_kit import (
    NanGuard, Reporter, StopFlag, eval_opts, export_all, init_source, keep_checkpoints, mark_complete,
    write_init_meta,
)
from app.core.backend_kit import prune_checkpoints as kit_prune
from app.core.backend_kit.runutil import dir_artifact, eval_ds, eval_rows, free_gpu
from app.core.backend_kit.whisper import asr_baseline_key
from app.core.contract import Phase, RunConfig, RunDir, read_prepared_refs
from app.core.evaluation import baseline_metrics, merge_results, run_tts_eval
from app.core.text import bn_normalize
from app.core.tts_refs import DEFAULT_SPEAKER

from .prepare.f5 import _safe, pick_vocab_file

BACKEND = "f5"
CKPT_FORMAT = "f5-pt"
SAMPLE_RATE = 24000
HOP = 256
SYNTH_SEED = 0
EVAL_CAP = 20
# F5TTS_Base as the IndicF5 wrapper builds it; text_mask_padding=False matches the pre-v1 recipe it was trained with
# (shapes are unaffected, only inference behaviour; VM check pending).
MODEL_CFG = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4, text_mask_padding=False)
MEL_KWARGS = dict(n_fft=1024, hop_length=HOP, win_length=1024, n_mel_channels=100, target_sample_rate=SAMPLE_RATE,
                  mel_spec_type="vocos")
TEXT_EMBED_KEY = "transformer.text_embed.text_embed.weight"
BOOKKEEPING = ("initted", "step")


class TrainStopped(Exception):
    """SIGTERM handled: a complete checkpoint was published."""


class TrainFinished(Exception):
    """The requested number of updates is done (max_steps reached)."""


# ---- pure helpers (light-testable) -------------------------------------------------------------------------------

def strip_to_model_keys(sd: dict) -> dict:
    """Map a wrapper / EMA / plain state dict onto the bare CFM keys (`transformer.*`, `mel_spec.*`).

    The prefix is whatever precedes `transformer.text_embed.text_embed.weight` (e.g. `ema_model.`, `model.`).
    When several prefixes qualify, an EMA copy wins, then the shortest. Keys outside the prefix and F5's EMA
    bookkeeping entries (`initted`, `step`) are dropped."""
    prefixes = {k[: -len(TEXT_EMBED_KEY)] for k in sd if isinstance(k, str) and k.endswith(TEXT_EMBED_KEY)}
    if not prefixes:
        sample = ", ".join(list(map(str, sd))[:5])
        raise ValueError(f"unrecognized checkpoint layout: no key ends with '{TEXT_EMBED_KEY}' (text_embed); "
                         f"first keys: {sample}")
    prefix = sorted(prefixes, key=lambda p: ("ema" not in p.lower(), len(p), p))[0]
    return {k[len(prefix):]: v for k, v in sd.items()
            if isinstance(k, str) and k.startswith(prefix) and k[len(prefix):] not in BOOKKEEPING}


def pick_state_dict(ckpt: dict, ema: bool = True) -> dict:
    """The flat weights of a loaded checkpoint: F5 `model_last.pt` (EMA by default) or a plain state dict."""
    if ema and "ema_model_state_dict" in ckpt:
        return ckpt["ema_model_state_dict"]
    if "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    if "ema_model_state_dict" in ckpt:
        return ckpt["ema_model_state_dict"]
    return ckpt


def ema_export_state(ckpt: dict) -> dict:
    return strip_to_model_keys(pick_state_dict(ckpt, ema=True))


def pick_weights_file(files: list[str]) -> str:
    """The raw DiT weights in a hub repo: `model.safetensors` at the root, else the shallowest .safetensors,
    else a .pt (preferring a `checkpoints/` dir)."""
    def depth(f: str) -> tuple:
        return (f.count("/"), f)

    st = sorted((f for f in files if f.endswith(".safetensors")), key=lambda f: (f != "model.safetensors", *depth(f)))
    if st:
        return st[0]
    pt = sorted((f for f in files if f.endswith((".pt", ".pth"))),
                key=lambda f: ("checkpoints" not in PurePosixPath(f).parts, *depth(f)))
    if pt:
        return pt[0]
    raise FileNotFoundError("no model weights (.safetensors / .pt) found in the model repo")


def plan_expansion(rows: int, vocab_size: int) -> int:
    """New embedding rows needed: the model has vocab_size + 1 rows; a checkpoint with fewer is expanded."""
    need = vocab_size + 1
    if rows > need:
        raise ValueError(f"vocab is smaller than the checkpoint's text embedding ({need - 1} characters vs "
                         f"{rows - 1}): the prepared vocab must extend the checkpoint's vocab")
    return need - rows


def expand_rows(weight: "torch.Tensor", n_new: int) -> "torch.Tensor":  # noqa: F821
    """Append n_new rows initialised to the mean of the existing rows."""
    import torch

    if n_new <= 0:
        return weight
    mean = weight.float().mean(dim=0, keepdim=True).to(weight.dtype)
    return torch.cat([weight, mean.expand(n_new, -1)], dim=0)


def check_load_result(missing: list[str], unexpected: list[str]) -> None:
    """Fail loudly on key mismatches; the mel-spectrogram buffers (recomputed, not trained) are ignored."""
    missing = [k for k in missing if not k.startswith("mel_spec.")]
    unexpected = [k for k in unexpected if not k.startswith("mel_spec.")]
    if missing or unexpected:
        def short(ks):
            return ", ".join(ks[:20]) + (f" (+{len(ks) - 20} more)" if len(ks) > 20 else "")

        raise RuntimeError(f"checkpoint does not match the F5 DiT: missing keys [{short(missing)}]; "
                           f"unexpected keys [{short(unexpected)}]")


def update_dir(rd: RunDir, n: int) -> Path:
    return rd.checkpoints_dir / f"update-{n}"


def publish_checkpoint(src: Path, dst: Path, cfg: RunConfig, step: int, vocab: Path) -> None:
    """MOVE F5's model_last.pt (+ copy vocab) into a contract checkpoint dir; meta before `.complete`.
    A move, never a link: torch.save rewrites the work file's inode in place."""
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    (dst / ".complete").unlink(missing_ok=True)
    os.replace(src, dst / "model_last.pt")
    shutil.copyfile(vocab, dst / "vocab.txt")
    write_init_meta(dst, backend=BACKEND, base_model=cfg.base_model, method="full", format=CKPT_FORMAT, step=step)
    mark_complete(dst)


def keep_count(hp: dict) -> int:
    return keep_checkpoints(hp)


def prune_checkpoints(rd: RunDir, keep: int) -> None:
    """Keep the latest `keep` complete update-* dirs (the newest is never removed); remove older complete ones
    and incomplete ones older than the newest complete. `final` and newer incomplete dirs are untouched."""
    kit_prune(rd.checkpoints_dir, r"update-(\d+)", keep)


def slim_state(ckpt: dict) -> dict:
    """What a final checkpoint keeps: the EMA weights only."""
    return {"ema_model_state_dict": pick_state_dict(ckpt, ema=True)}


def split_sentences(text: str) -> list[str]:
    """Split after each danda (kept on its sentence); F5 only chunks on ASCII punctuation."""
    out, cur = [], ""
    for c in text:
        cur += c
        if c == "।":
            out.append(cur.strip())
            cur = ""
    if cur.strip():
        out.append(cur.strip())
    return [x for x in out if x]


def write_slim_final(src: Path, final: Path, cfg: RunConfig, step: int, vocab: Path) -> None:
    import torch

    ckpt = torch.load(str(src), map_location="cpu", weights_only=True)
    final.mkdir(parents=True, exist_ok=True)
    (final / ".complete").unlink(missing_ok=True)
    tmp = final / "model_last.pt.tmp"
    torch.save(slim_state(ckpt), str(tmp))
    del ckpt
    os.replace(tmp, final / "model_last.pt")
    shutil.copyfile(vocab, final / "vocab.txt")
    write_init_meta(final, backend=BACKEND, base_model=cfg.base_model, method="full", format=CKPT_FORMAT, step=step)
    mark_complete(final)  # meta before .complete


def updates_per_epoch(durations: list[float], batch_frames: int, accum: int) -> int:
    """Optimizer updates per epoch for frame-based batching (mel frames = seconds * sr / hop)."""
    frames = sum(float(d) for d in durations) * SAMPLE_RATE / HOP
    batches = math.ceil(frames / max(1, batch_frames))
    return max(1, math.ceil(batches / max(1, accum)))


def total_updates(hp: dict, per_epoch: int) -> int:
    if hp.get("max_steps"):
        return int(hp["max_steps"])
    return max(1, math.ceil(per_epoch * float(hp.get("epochs", 1))))


def save_interval(hp: dict, total: int) -> int:
    if hp.get("save_steps") is not None:
        return max(1, int(hp["save_steps"]))
    return max(50, total // 10)


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


def f5_baseline_key(cfg: RunConfig, max_samples: int | None) -> dict:
    return asr_baseline_key(cfg, max_samples)  # max_samples + init checkpoint


class UpdateHook:
    """Stands in for `accelerator.log`: F5 calls it once per optimizer update ({loss, lr}, step=update)."""

    def __init__(self, rep: Reporter, stop: StopFlag, nan: NanGuard, total: int, save) -> None:
        self.rep, self.stop, self.nan, self.total, self.save = rep, stop, nan, total, save
        self.last = 0

    def __call__(self, values: dict, step: int | None = None, log_kwargs=None) -> None:
        if step is None or step == self.last:
            return
        self.last = step = int(step)
        loss = values.get("loss")
        if loss is not None:
            self.rep.metric(step, "train", "loss", float(loss))
            self.nan.feed(float(loss))
        if values.get("lr") is not None:
            self.rep.metric(step, "train", "lr", float(values["lr"]))
        self.rep.status(Phase.train, step, self.total, "training")
        if step >= self.total:
            self.save(step)
            raise TrainFinished
        if self.stop.requested:
            self.save(step)
            raise TrainStopped


# ---- model -------------------------------------------------------------------------------------------------------

def build_model(vocab_path: Path):
    from f5_tts.model import CFM, DiT
    from f5_tts.model.utils import get_tokenizer

    vocab_char_map, vocab_size = get_tokenizer(str(vocab_path), "custom")
    return CFM(transformer=DiT(**MODEL_CFG, text_num_embeds=vocab_size, mel_dim=MEL_KWARGS["n_mel_channels"]),
               mel_spec_kwargs=dict(MEL_KWARGS), odeint_kwargs=dict(method="euler"), vocab_char_map=vocab_char_map)


def load_weights(model, sd: dict) -> None:
    """Load a (prefix-stripped) state dict, expanding the text embedding when the vocab grew. Strict apart from
    the mel buffers: any other missing or unexpected key raises with the list."""
    sd = strip_to_model_keys(sd)
    vocab_size = model.transformer.text_embed.text_embed.num_embeddings - 1
    w = sd[TEXT_EMBED_KEY]
    n_new = plan_expansion(int(w.shape[0]), vocab_size)
    if n_new:
        sd = {**sd, TEXT_EMBED_KEY: expand_rows(w, n_new)}
    res = model.load_state_dict(sd, strict=False)
    check_load_result(list(res.missing_keys), list(res.unexpected_keys))


def hub_files(repo: str) -> tuple[Path, Path]:
    """(weights file, vocab file) of the base model repo, downloaded into the default hub cache ($HF_HOME/hub;
    cache_dir is not passed so `hf cache scan` sees the files and nothing is stored twice). The raw DiT weights are
    found with list_repo_files (IndicF5 ships them inside a trust_remote_code wrapper repo)."""
    from huggingface_hub import hf_hub_download, list_repo_files

    files = list_repo_files(repo)
    weights = pick_weights_file(files)
    vocab = pick_vocab_file(files)
    return Path(hf_hub_download(repo, weights)), Path(hf_hub_download(repo, vocab))


def read_weights(path: Path) -> dict:
    import torch

    p = str(path)
    if p.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(p, device="cpu")
    return torch.load(p, map_location="cpu", weights_only=True)


def init_weights_and_vocab(rd: RunDir, cfg: RunConfig) -> tuple[Path, Path]:
    """Warm-start source: the init checkpoint (read-only: model_last.pt + vocab.txt) else the base model."""
    choice = init_source(cfg.model_copy(update={"resume": False}), rd)
    if choice is None:
        if _gated(cfg):
            from app.core.backend_kit.hf_access import check_hf_access

            check_hf_access(cfg.base_model)
        return hub_files(cfg.base_model)
    fmt = (choice.meta or {}).get("format")
    if fmt != CKPT_FORMAT:
        raise ValueError(f"unsupported init checkpoint format '{fmt}' for backend {BACKEND}: only {CKPT_FORMAT} "
                         "checkpoints can be used")
    return choice.path / "model_last.pt", choice.path / "vocab.txt"


def _gated(cfg: RunConfig) -> bool:
    from .models import MODELS

    info = MODELS.get(cfg.base_model)
    return bool(info and info.gated)


def prepared_dir(rd: RunDir, role: str = "train") -> Path:
    refs = read_prepared_refs(rd)
    ref = next((r for r in refs if r.get("role") == role), None)
    if ref is None:
        raise RuntimeError(f"no prepared {role} dataset (prepared_refs.json has no {role} entry)")
    return Path(ref["path"])


def load_split(prepared: Path, split: str):
    import json

    from datasets import Dataset
    from f5_tts.model.dataset import CustomDataset

    d = Path(prepared) / split
    ds = Dataset.from_file(str(d / "raw.arrow"))
    durations = json.loads((d / "duration.json").read_text(encoding="utf-8"))["duration"]
    return CustomDataset(ds, durations=durations, preprocessed_mel=False, **MEL_KWARGS)


# ---- train -------------------------------------------------------------------------------------------------------

def _trainer_cls(publish):
    from f5_tts.model import Trainer

    class ContractTrainer(Trainer):
        def save_checkpoint(self, update, last=False):
            if not last:  # F5's per-update model_<n>.pt files are not used; `last` saves are published instead
                return
            super().save_checkpoint(update, last=True)
            if self.is_main:
                publish(int(update))

    return ContractTrainer


def train(rd: RunDir, cfg: RunConfig, rep) -> int:
    import torch

    hp = cfg.hparams
    stop = StopFlag().install()
    prepared = prepared_dir(rd, "train")
    vocab = prepared / "vocab.txt"
    dataset = load_split(prepared, "train")
    bf = int(hp.get("batch_size_frames", 3200))
    accum = int(hp.get("grad_accum", 1))
    per_epoch = updates_per_epoch(list(dataset.durations), bf, accum)
    total = total_updates(hp, per_epoch)
    epochs = max(1, math.ceil(total / per_epoch))
    save_steps = save_interval(hp, total)
    warmup = int(hp["warmup_steps"]) if hp.get("warmup_steps") is not None else min(1000, max(1, total // 10))
    torch.manual_seed(cfg.seed)

    work = rd.path / "work" / "f5_ckpt"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    model = build_model(vocab)
    choice = init_source(cfg, rd)
    if choice is not None and choice.kind == "resume":
        shutil.copyfile(choice.path / "model_last.pt", work / "model_last.pt")  # F5's Trainer resumes from it
    else:
        weights, _ = init_weights_and_vocab(rd, cfg)
        ckpt = read_weights(weights)
        load_weights(model, pick_state_dict(ckpt))
        del ckpt

    def publish(n: int) -> None:
        publish_checkpoint(work / "model_last.pt", update_dir(rd, n), cfg, n, vocab)
        prune_checkpoints(rd, keep_count(hp))

    os.environ.setdefault("ACCELERATE_MIXED_PRECISION", "bf16")
    trainer = _trainer_cls(publish)(
        model, epochs, float(hp.get("learning_rate", 1e-5)), num_warmup_updates=warmup, save_per_updates=save_steps,
        last_per_updates=save_steps, checkpoint_path=str(work), batch_size_per_gpu=bf, batch_size_type="frame",
        grad_accumulation_steps=accum, logger=None)
    hook = UpdateHook(rep, stop, NanGuard(), total, lambda n: trainer.save_checkpoint(n, last=True))
    trainer.accelerator.log = hook
    if stop.requested:
        return 143  # SIGTERM before any work: nothing to save
    try:
        trainer.train(dataset, num_workers=min(8, os.cpu_count() or 1), resumable_with_seed=cfg.seed)
    except TrainStopped:
        return 143  # a complete update dir was published; resume continues from it
    except TrainFinished:
        pass
    step = hook.last
    if step < total:
        print(f"f5: training ended at update {step} of {total}", file=sys.stderr)
    src = work / "model_last.pt"
    if not src.is_file():  # the last save was already published as update-<n>
        src = update_dir(rd, step) / "model_last.pt"
    write_slim_final(src, rd.checkpoints_dir / "final", cfg, step, vocab)
    prune_checkpoints(rd, keep_count(hp))
    shutil.rmtree(work, ignore_errors=True)
    return 0


# ---- eval --------------------------------------------------------------------------------------------------------

def _require_final(rd: RunDir) -> Path:
    final = rd.checkpoints_dir / "final"
    if not (final / ".complete").exists():
        raise RuntimeError(f"no completed final checkpoint at {final}")
    return final


def make_synthesizer(model, vocoder, refs: Path, device: str = "cuda"):
    from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text

    cache: dict[Path, tuple[str, str]] = {}

    def quiet(*_a, **_k) -> None:
        return None

    def synthesize(text: str, row: dict):
        import torch

        wav = pick_ref(refs, row.get("speaker"))
        if wav not in cache:
            ref_text = wav.with_suffix(".txt").read_text(encoding="utf-8")
            cache[wav] = preprocess_ref_audio_text(str(wav), ref_text, show_info=quiet)
        ref_audio, ref_text = cache[wav]
        torch.manual_seed(SYNTH_SEED)
        import numpy as np

        waves, sr = [], SAMPLE_RATE
        for sent in split_sentences(bn_normalize(text, mode="tts")) or [text]:
            audio, sr, _ = infer_process(ref_audio, ref_text, sent, model, vocoder, mel_spec_type="vocos",
                                         show_info=quiet, device=device)
            waves.append(np.asarray(audio))
        return np.concatenate(waves), int(sr)

    return synthesize


def _load_for_eval(weights: Path, vocab: Path, device: str):
    model = build_model(vocab)
    ckpt = read_weights(weights)
    load_weights(model, pick_state_dict(ckpt))
    return model.to(device).eval()


def eval_f5(rd: RunDir, cfg: RunConfig, rep) -> int:
    final = _require_final(rd)
    from f5_tts.infer.utils_infer import load_vocoder

    do_baseline, max_samples = eval_opts(cfg)
    cap = EVAL_CAP if max_samples is None else min(EVAL_CAP, int(max_samples))
    rows = eval_rows(cfg)
    role = "eval" if cfg.datasets.get("eval") is not None else "train"
    refs = prepared_dir(rd, role) / "refs"
    device = "cuda"
    vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=device)

    model = _load_for_eval(final / "model_last.pt", final / "vocab.txt", device)
    tuned = run_tts_eval(rd, rep, rows, make_synthesizer(model, vocoder, refs, device),
                         loss=None, prefix="", max_samples=cap)
    del model
    free_gpu()

    base: dict[str, float] = {}
    if do_baseline:
        ds, _ = eval_ds(cfg)

        def compute() -> dict[str, float]:
            weights, vocab = init_weights_and_vocab(rd, cfg)  # init checkpoint when set, else the original IndicF5
            m = _load_for_eval(weights, vocab, device)
            try:
                res = run_tts_eval(rd, rep, rows, make_synthesizer(m, vocoder, refs, device),
                                   loss=None, prefix="base/", max_samples=cap)
                return {k.removeprefix("base/"): v for k, v in res.items()}  # baseline_metrics re-adds `base/`
            finally:
                del m
                free_gpu()

        base = baseline_metrics(cfg, ds, f5_baseline_key(cfg, max_samples), compute)
    merge_results(rd, tuned, base)
    return 0


# ---- export ------------------------------------------------------------------------------------------------------

def f5_exporters(rd: RunDir, cfg: RunConfig) -> dict:
    def f5_ckpt(out: Path):
        final = _require_final(rd)
        import torch
        from safetensors.torch import save_file

        ckpt = torch.load(str(final / "model_last.pt"), map_location="cpu", weights_only=True)
        sd = {k: v.detach().clone().contiguous() for k, v in ema_export_state(ckpt).items()}
        save_file(sd, str(out / "model.safetensors"))
        shutil.copyfile(final / "vocab.txt", out / "vocab.txt")
        refs = prepared_dir(rd, "train") / "refs"
        if refs.is_dir():
            shutil.copytree(refs, out / "refs")
        return [dir_artifact(rd, "f5-ckpt", out, "f5-tts")]

    return {"f5-ckpt": f5_ckpt}


def export_f5(rd: RunDir, cfg: RunConfig, rep) -> int:
    return export_all(rd, cfg, rep, f5_exporters(rd, cfg))

