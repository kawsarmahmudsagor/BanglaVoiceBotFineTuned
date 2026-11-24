"""NeMo ASR family handlers (`nemo-manifest`): prepare, train, eval, export.
nemo, torch, lightning and huggingface_hub are imported inside functions only.

Training uses a plain Lightning Trainer (no NeMo exp_manager, which would add its own checkpoint callback) with the
shared ContractCheckpoint / ContractLightningCallback. Eval of the hybrid IndicConformer uses its default decoder
(RNNT). Manifests reference source audio at its native rate; NeMo resamples to 16 kHz at load (sample_rate below).
"""
from __future__ import annotations

import copy
import math
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

from app.core.backend_kit import (
    NanGuard, StopFlag, eval_opts, export_all, init_source, keep_checkpoints, latest_checkpoint, mark_complete,
    max_skip_share, write_init_meta,
)
from app.core.backend_kit.lightning import ContractCheckpoint, ContractLightningCallback
from app.core.backend_kit.runutil import dir_artifact, eval_ds, eval_rows, free_gpu, total_steps, train_ref
from app.core.backend_kit.whisper import asr_baseline_key
from app.core.contract import RunConfig, RunDir, write_prepared_refs
from app.core.evaluation import baseline_metrics, merge_results, run_asr_eval
from app.core.backend_kit.audio_refs import check_audio_refs, first_last_lines
from app.core.prepare import PrepareError, ensure_prepared

from .models import MODELS
from .prepare.manifest import build_manifest, manifest_spec, tokenizer_hash


def restore_model(base_model: str):
    """Load the model on CPU. A local .nemo path is restored directly; a Hugging Face repo holding a .nemo file
    is downloaded into HF_HOME first; anything else goes through from_pretrained."""
    from nemo.collections.asr.models import ASRModel

    if Path(base_model).is_file():
        return ASRModel.restore_from(base_model, map_location="cpu")
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

    try:
        local = Path(snapshot_download(base_model, allow_patterns=["*.nemo"],
                                       cache_dir=os.environ.get("HF_HOME") or None))
    except (RepositoryNotFoundError, HfHubHTTPError, ValueError) as e:  # not an HF repo: NGC / from_pretrained
        print(f"nemo: no .nemo download for {base_model} ({type(e).__name__}); trying from_pretrained", file=sys.stderr)
        local = None
    found = sorted(local.glob("*.nemo")) if local else []
    if found:
        return ASRModel.restore_from(str(found[0]), map_location="cpu")
    return ASRModel.from_pretrained(base_model, map_location="cpu")


def tokenizer_identity(tok) -> str:
    """Hash of the tokenizer vocabulary (a model change re-prepares)."""
    vocab = getattr(tok, "vocab", None)
    if vocab is None:
        vocab = [tok.ids_to_tokens([i]) for i in range(int(tok.vocab_size))]
    return tokenizer_hash(vocab if isinstance(vocab, dict) else list(vocab))


def resolve_unk_id(tok, model_name: str) -> int:
    unk = getattr(tok, "unk_id", None)
    if unk is None and hasattr(tok, "token_to_id"):
        unk = tok.token_to_id("<unk>")
    if unk is None and hasattr(tok, "tokens_to_ids"):
        try:
            ids = tok.tokens_to_ids(["<unk>"])
            unk = ids[0] if ids else None
        except Exception:  # noqa: BLE001 - tokenizer without that token
            unk = None
    if unk is None:
        raise PrepareError(f"tokenizer of {model_name} has no unk id; cannot guard unknown characters")
    return int(unk)


def check_manifest_refs(info) -> None:
    """Cache-hit safety: the first and last row of each manifest must still point at existing audio."""
    for name in info.files:
        if name.endswith(".jsonl"):
            ends = first_last_lines(Path(info.path) / name)
            check_audio_refs([r["audio_filepath"] for r in ends], str(info.path))


def prepare_nemo(rd: RunDir, cfg: RunConfig, rep) -> int:
    if MODELS[cfg.base_model].gated:
        from app.core.backend_kit.hf_access import check_hf_access

        check_hf_access(cfg.base_model)
    model = restore_model(cfg.base_model)
    tok = model.tokenizer
    unk_id = resolve_unk_id(tok, cfg.base_model)
    sha = tokenizer_identity(tok)
    del model
    spec = manifest_spec(cfg, sha)
    share = max_skip_share(cfg)
    refs = []
    for role, key, splits in (("train", "train", ["train", "val"]), ("eval", "eval", None)):
        ds = cfg.datasets.get(key)
        if ds is None:
            continue
        build = build_manifest(tok.text_to_ids, unk_id, rep)
        pi = ensure_prepared(cfg, ds, spec, build, splits=splits, max_skip_share=share)
        check_manifest_refs(pi)
        refs.append({"role": role, **pi.model_dump()})
    write_prepared_refs(rd, refs)
    return 0


prepare = prepare_nemo  # phase handler looked up by phases._dispatch

BACKEND = "nemo"
SAMPLE_RATE = 16000
MAX_DURATION = 30


# ---- pure helpers (light-testable) -------------------------------------------------------------------------------

def init_path_choice(meta: dict | None, init_dir: Path) -> tuple[str, Path]:
    """How to load an init checkpoint: a `nemo` dir is restored whole; a `lightning-ckpt` dir is a state_dict
    loaded over the base model. `init_dir` is read-only."""
    if meta is None:
        raise ValueError("init checkpoint has no init_meta.json")
    fmt = meta.get("format")
    if fmt == "nemo":
        return "restore_from", Path(init_dir) / "model.nemo"
    if fmt == "lightning-ckpt":
        return "state_dict", Path(init_dir) / "model.ckpt"
    raise ValueError(f"unsupported init checkpoint format '{fmt}' for backend {BACKEND}: only nemo and "
                     "lightning-ckpt checkpoints can be used")


def data_config(manifest: Path, *, batch_size: int, shuffle: bool, cpu_count: int | None = None) -> dict:
    return {"manifest_filepath": str(manifest), "sample_rate": SAMPLE_RATE, "batch_size": int(batch_size),
            "shuffle": shuffle, "num_workers": min(8, cpu_count or os.cpu_count() or 1),
            "max_duration": MAX_DURATION, "pin_memory": True}


def apply_data_config(model, prepared_dir: Path, *, batch_size: int) -> bool:
    """Point the model at train.jsonl (+ val.jsonl when non-empty). Only the data loaders are configured: the
    aux_ctc weight and the encoder streaming keys (att_context_size, conv_context_size) stay as in the
    checkpoint. Returns whether a validation set was set up."""
    d = Path(prepared_dir)
    model.setup_training_data(data_config(d / "train.jsonl", batch_size=batch_size, shuffle=True))
    has_val = has_val_manifest(d)
    if has_val:
        model.setup_validation_data(data_config(d / "val.jsonl", batch_size=batch_size, shuffle=False))
    return has_val


def merged_optim(optim: dict, hp: dict, total: int) -> dict:
    """The model's own optim config with only lr / warmup overridden. Without a recipe warmup_steps the
    scheduler's own warmup (steps, or ratio * total) is clamped to max(1, total // 10) so short runs still decay."""
    out = copy.deepcopy(optim)
    if hp.get("learning_rate") is not None:
        out["lr"] = float(hp["learning_rate"])
    sched = out.get("sched")
    if not isinstance(sched, dict):
        return out
    if hp.get("warmup_steps") is not None:
        sched["warmup_steps"] = int(hp["warmup_steps"])
        sched.pop("warmup_ratio", None)
        return out
    existing = sched.get("warmup_steps")
    if existing is None and sched.get("warmup_ratio") is not None:
        existing = float(sched["warmup_ratio"]) * total
    cap = max(1, total // 10)
    if existing is not None and existing > cap:
        sched["warmup_steps"] = cap
        sched.pop("warmup_ratio", None)
    return out


def save_interval(hp: dict, total: int) -> int:
    """Optimizer steps between checkpoints: hparams.save_steps, else max(50, total // 10)."""
    if hp.get("save_steps") is not None:
        return max(1, int(hp["save_steps"]))
    return max(50, total // 10)


def has_val_manifest(prepared_dir: Path) -> bool:
    val = Path(prepared_dir) / "val.jsonl"
    return val.is_file() and val.stat().st_size > 0


def hyp_texts(out) -> list[str]:
    """transcribe() returns strings, Hypothesis objects, n-best lists, or a (best, all) tuple for RNNT."""
    if isinstance(out, tuple):
        out = out[0]
    texts = []
    for h in out:
        if isinstance(h, (list, tuple)):
            h = h[0] if h else ""
        texts.append(h if isinstance(h, str) else str(getattr(h, "text", h)))
    return texts


# ---- model loading -----------------------------------------------------------------------------------------------

def load_model(rd: RunDir, cfg: RunConfig):
    """Starting model: the warm-start init dir (read-only; nemo -> restore_from, lightning-ckpt -> base +
    state_dict) or the base model. A resumed run starts from the same source; Lightning then restores its own
    checkpoint over it."""
    choice = init_source(cfg.model_copy(update={"resume": False}), rd)
    if choice is None:
        return restore_model(cfg.base_model)
    kind, path = init_path_choice(choice.meta, choice.path)
    if kind == "restore_from":
        return restore_model(str(path))
    import torch

    model = restore_model(cfg.base_model)
    model.load_state_dict(torch.load(str(path), map_location="cpu", weights_only=False)["state_dict"])
    return model


def _write_final(model, rd: RunDir, cfg: RunConfig, step: int) -> None:
    final = rd.checkpoints_dir / "final"
    final.mkdir(parents=True, exist_ok=True)
    model.save_to(str(final / "model.nemo"))
    write_init_meta(final, backend=BACKEND, base_model=cfg.base_model, method="full", format="nemo", step=step)
    mark_complete(final)  # meta before .complete


# ---- train -------------------------------------------------------------------------------------------------------

def train(rd: RunDir, cfg: RunConfig, rep) -> int:
    from lightning.pytorch import Callback, Trainer
    from omegaconf import DictConfig, OmegaConf, open_dict

    hp = cfg.hparams
    stop = StopFlag().install()
    ref = train_ref(rd)
    prepared = Path(ref["path"])
    batch = int(hp.get("batch_size", 2))
    accum = int(hp.get("grad_accum", 4))
    total = total_steps(cfg, rd)  # optimizer steps; max_steps overrides epochs
    save_steps = save_interval(hp, total)
    has_val = has_val_manifest(prepared)
    batches_per_epoch = max(1, math.ceil(int(ref.get("counts", {}).get("train", 0)) / batch))

    class _InstallStop(Callback):  # Lightning installs its own SIGTERM handler at fit start; ours must win
        def on_train_start(self, trainer, pl_module):
            stop.install()

    model = load_model(rd, cfg)
    trainer = Trainer(
        devices=1, accelerator="gpu", precision="bf16-mixed", max_steps=total, max_epochs=-1,
        accumulate_grad_batches=accum, logger=False, enable_checkpointing=True,
        callbacks=[_InstallStop(), ContractLightningCallback(rep, stop, NanGuard(), total),
                   ContractCheckpoint(rd, cfg.base_model, every_n_train_steps=save_steps,
                                      keep=keep_checkpoints(hp))],
        check_val_every_n_epoch=None, val_check_interval=min(save_steps * accum, batches_per_epoch),
        num_sanity_val_steps=0, enable_progress_bar=False,
        log_every_n_steps=1,  # keeps train_loss in callback_metrics at every optimizer step (the contract callback reads it)
        limit_val_batches=1.0 if has_val else 0)
    model.set_trainer(trainer)
    apply_data_config(model, prepared, batch_size=batch)
    # Keep the checkpoint's optimizer/scheduler config; Lightning builds it via model.setup_optimization().
    if model.cfg.get("optim") is None:
        print("nemo: model config has no optim section; leaving optimizer settings to NeMo defaults", file=sys.stderr)
    else:
        optim = OmegaConf.to_container(model.cfg.optim, resolve=True)
        with open_dict(model.cfg):
            model.cfg.optim = DictConfig(merged_optim(optim, hp, total))

    last = latest_checkpoint(rd) if cfg.resume else None
    ckpt_path = str(last / "model.ckpt") if last is not None and (last / "model.ckpt").is_file() else None
    trainer.fit(model, ckpt_path=ckpt_path)
    step = int(trainer.global_step)
    if stop.requested and step < total:
        return 143  # ContractLightningCallback already saved a complete checkpoint; resume continues from it
    if trainer.is_global_zero:
        _write_final(model, rd, cfg, step)  # also when stop arrived at the natural end
    return 0


# ---- eval --------------------------------------------------------------------------------------------------------

def make_transcriber(model, batch_size: int):
    def transcribe(paths: list[str]) -> list[str]:
        return hyp_texts(model.transcribe(paths, batch_size=batch_size))

    return transcribe


def _eval_model(model, cfg: RunConfig, rd: RunDir, rep, rows, prefix: str, max_samples) -> dict[str, float]:
    model.eval()
    model.cuda()
    bs = int((cfg.hparams.get("eval") or {}).get("batch_size", 16))
    return run_asr_eval(rd, rep, rows, make_transcriber(model, bs), prefix=prefix, batch_size=bs,
                        max_samples=max_samples)


def eval(rd: RunDir, cfg: RunConfig, rep) -> int:  # noqa: A001  (phase handler name looked up by phases._dispatch)
    final = rd.checkpoints_dir / "final"
    if not (final / ".complete").exists():
        raise RuntimeError(f"no completed final checkpoint at {final}")
    do_baseline, max_samples = eval_opts(cfg)
    rows = eval_rows(cfg)
    check_manifest_refs(SimpleNamespace(**{k: train_ref(rd)[k] for k in ("files", "path")}))

    model = restore_model(str(final / "model.nemo"))
    tuned = _eval_model(model, cfg, rd, rep, rows, "", max_samples)
    del model
    free_gpu()

    base: dict[str, float] = {}
    if do_baseline:
        ds, _ = eval_ds(cfg)

        def compute() -> dict[str, float]:
            m = load_model(rd, cfg)  # init checkpoint when set, else the base model
            try:
                res = _eval_model(m, cfg, rd, rep, rows, "base/", max_samples)
                return {k.removeprefix("base/"): v for k, v in res.items()}  # baseline_metrics re-adds `base/`
            finally:
                del m
                free_gpu()

        base = baseline_metrics(cfg, ds, asr_baseline_key(cfg, max_samples), compute)
    merge_results(rd, tuned, base)
    return 0


# ---- export ------------------------------------------------------------------------------------------------------

def nemo_exporters(rd: RunDir) -> dict:
    final = rd.checkpoints_dir / "final"

    def nemo(out: Path):
        if not (final / ".complete").exists():
            raise RuntimeError(f"no completed final checkpoint at {final}")
        shutil.copy2(final / "model.nemo", out / "model.nemo")
        return [dir_artifact(rd, "nemo", out, "nemo")]

    return {"nemo": nemo}


def export(rd: RunDir, cfg: RunConfig, rep) -> int:
    return export_all(rd, cfg, rep, nemo_exporters(rd))
