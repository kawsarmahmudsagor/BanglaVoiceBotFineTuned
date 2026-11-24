"""CosyVoice 3 fine-tuning (llm / flow / hifigan), single GPU. torch, hyperpyyaml and cosyvoice (upstream, on
sys.path through the venv's .pth) are imported inside functions only.

Single-GPU adapter (upstream cosyvoice/bin/train.py assumes torchrun): we keep upstream's own pieces
(init_distributed, init_dataset_and_dataloader, wrap_cuda_model = DDP, init_optimizer_and_scheduler,
check_modify_and_save_config and the Executor epoch loops) and run them in a ONE-process torch.distributed group
(RANK=0, WORLD_SIZE=1, MASTER_ADDR=127.0.0.1, a free MASTER_PORT, nccl). That is the least-patching route: every
dist.* call upstream makes (barrier, DDP, model.join) is valid in a 1-process group, so no upstream code is
modified. Only four module-level names inside `cosyvoice.utils.executor` are swapped for the duration of a
component (and restored in `finally`):
  - log_per_step    -> original + Reporter metrics + NanGuard + StopFlag/max_steps check
  - log_per_save    -> original + `eval_loss_<component>` metric (the upstream CV pass)
  - save_model      -> original into the work dir, then published as `checkpoints/<component>-epoch-<n>/model.pt`
  - cosyvoice_join  -> returns True when a halt (SIGTERM / max_steps) is pending, which makes the upstream loop
                       `break` cleanly (no exception through DDP.join); the instance's `cv` is wrapped so the halt
                       saves a checkpoint instead of running a full CV pass.
Mixed precision: upstream `use_amp` (bf16 autocast + GradScaler), on whenever CUDA is available.

Checkpoints (`checkpoints/`): `<component>-epoch-<n>/model.pt` after each finished epoch n (1-based), plus
`<component>-epoch-<n>-step-<s>/` for `save_steps` and for a stop / max_steps halt inside epoch n (stored with
the epoch field of epoch n-1, so a resume REPEATS epoch n from its start with the saved weights and step counter;
upstream does not save optimizer state). A finished component is `<component>-final/model.pt` (inference format: no epoch/step
keys; hifigan = generator only, no `generator.` prefix) and its epoch dirs are removed. `final/` is assembled at the
end (see assemble_final). Every complete dir carries init_meta.json before `.complete`: format `cosyvoice-pt` for
`final/`, `cosyvoice-component` for the per-component dirs (not warm-start sources).

Resume semantics (upstream saves WEIGHTS ONLY): a resume restores the weights and the step counter (scheduler position)
but restarts the optimizer state (Adam moments) and the GradScaler. A mid-epoch save (save_steps, SIGTERM, max_steps) makes
the resume repeat that epoch from its start; the step counter is NOT rolled back, so the repeated epoch's steps count
again (total steps can exceed epochs x steps-per-epoch).
"""
from __future__ import annotations

import logging
import math
import os
import re
import shutil
import socket
import sys
from pathlib import Path

from app.core.backend_kit import (
    NanGuard, StopFlag, init_source, keep_checkpoints, mark_complete, read_init_meta, write_init_meta,  # noqa: F401
)
from app.core.backend_kit import prune_checkpoints as kit_prune
from app.core.contract import Phase, RunConfig, RunDir, read_prepared_refs

from .compat import ensure_deepspeed_stub, use_offline_features
from .files import link_or_copy, link_tree

BACKEND = "cosyvoice"
CKPT_FORMAT = "cosyvoice-pt"  # `final/` only: the three weight files CosyVoice3 loads
COMPONENT_FORMAT = "cosyvoice-component"  # one component's weights; not a warm-start source (resolve_init rejects it)
COMPONENTS = ("llm", "flow", "hifigan")
DEFAULT_COMPONENTS = ("llm",)  # upstream examples/libritts/cosyvoice3 trains the LLM only; flow is opt-in
WEIGHT_FILE = {"llm": "llm.pt", "flow": "flow.pt", "hifigan": "hift.pt"}
AUX_FILES = ("cosyvoice3.yaml", "speech_tokenizer_v3.onnx", "campplus.onnx")
AUX_DIR = "CosyVoice-BlankEN"
NUM_WORKERS = 2
PREFETCH = 20


# ---- pure helpers (light-testable) -------------------------------------------------------------------------------

def resolve_components(hp: dict) -> list[str]:
    """The requested components in training order (default llm only)."""
    return plan_components(list(hp.get("components") or DEFAULT_COMPONENTS), [])


def plan_components(requested: list[str], completed: list[str]) -> list[str]:
    """Components to train now: requested ones not yet completed, in the fixed order llm, flow, hifigan."""
    unknown = [c for c in requested if c not in COMPONENTS]
    if unknown:
        raise ValueError(f"unknown component(s): {', '.join(unknown)} (choose from {', '.join(COMPONENTS)})")
    return [c for c in COMPONENTS if c in set(requested) and c not in set(completed)]


def keep_count(hp: dict) -> int:
    return keep_checkpoints(hp)


def epoch_dir(rd: RunDir, component: str, n: int, step: int | None = None) -> Path:
    name = f"{component}-epoch-{n}" + (f"-step-{step}" if step is not None else "")
    return rd.checkpoints_dir / name


def final_component_dir(rd: RunDir, component: str) -> Path:
    return rd.checkpoints_dir / f"{component}-final"


def is_complete(d: Path) -> bool:
    return (Path(d) / ".complete").exists()


def _epoch_key(m: re.Match) -> tuple[int, int, int]:
    n, s = int(m.group(1)), m.group(2)
    return (n, 0, int(s)) if s is not None else (n, 1, 0)


def checkpoint_dirs(rd: RunDir, component: str) -> list[tuple[tuple[int, int, int], Path, bool]]:
    """(sort key, dir, complete) of the component's epoch dirs, oldest first. A step save inside epoch n sorts
    before the end of epoch n and after the end of epoch n-1."""
    pat = re.compile(rf"{re.escape(component)}-epoch-(\d+)(?:-step-(\d+))?")
    found = []
    if not rd.checkpoints_dir.is_dir():
        return found
    for d in rd.checkpoints_dir.iterdir():
        m = pat.fullmatch(d.name)
        if d.is_dir() and m:
            found.append((_epoch_key(m), d, is_complete(d)))
    return sorted(found, key=lambda t: t[0])


def latest_complete(rd: RunDir, component: str) -> Path | None:
    done = [d for _, d, ok in checkpoint_dirs(rd, component) if ok]
    return done[-1] if done else None


def prune_checkpoints(rd: RunDir, component: str, keep: int) -> None:
    """Keep the latest `keep` complete `<component>-epoch-*` dirs (the newest is never removed); remove older
    complete ones and incomplete ones older than the newest complete. `<component>-final`, `final`, `init`, other
    components' dirs and newer incomplete dirs are untouched."""
    kit_prune(rd.checkpoints_dir, rf"{re.escape(component)}-epoch-(\d+)(?:-step-(\d+))?", keep, key=_epoch_key)


def clean_state(sd: dict, component: str) -> dict:
    """Inference-format weights of a saved state dict: no `epoch` / `step` bookkeeping (CosyVoice loads strictly);
    hifigan keeps only the generator, without the `generator.` prefix (a plain hift.pt is returned unchanged)."""
    out = {k: v for k, v in sd.items() if k not in ("epoch", "step")}
    if component == "hifigan":
        gen = {k[len("generator."):]: v for k, v in out.items() if k.startswith("generator.")}
        if gen:
            return gen
    return out


def publish_checkpoint(src: Path, dst: Path, cfg: RunConfig, step: int) -> None:
    """MOVE a saved weights file into a contract checkpoint dir; meta before `.complete`. A move, never a link:
    the writer replaces files instead of rewriting them in place."""
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    (dst / ".complete").unlink(missing_ok=True)
    os.replace(src, dst / "model.pt")
    write_init_meta(dst, backend=BACKEND, base_model=cfg.base_model, method="full", format=COMPONENT_FORMAT,
                    step=step)
    mark_complete(dst)


def component_sources(rd: RunDir, cfg: RunConfig, base_dir: Path, trained: dict[str, Path]) -> dict[str, Path]:
    """`final/` weight file name -> source file. Trained components use their final weights; the others the warm-start
    init dir's file when `init_from` is set and has it, else the base model snapshot."""
    out: dict[str, Path] = {}
    for comp, fname in WEIGHT_FILE.items():
        if comp in trained:
            out[fname] = Path(trained[comp])
            continue
        init = rd.init_dir / fname
        out[fname] = init if cfg.init_from is not None and init.is_file() else Path(base_dir) / fname
    return out


def assemble_final(final: Path, sources: dict[str, Path], base_dir: Path, cfg: RunConfig, step: int) -> None:
    """Build a directory CosyVoice3(final) loads: the three weight files, cosyvoice3.yaml, the two onnx files and
    CosyVoice-BlankEN/ (hard links where possible). Meta before `.complete`. Existing files are replaced (never
    rewritten in place), so sources and linked checkpoints stay intact."""
    final, base_dir = Path(final), Path(base_dir)
    final.mkdir(parents=True, exist_ok=True)
    (final / ".complete").unlink(missing_ok=True)
    for fname, src in sources.items():
        link_or_copy(src, final / fname)
    for fname in AUX_FILES:
        link_or_copy(base_dir / fname, final / fname)
    shutil.rmtree(final / AUX_DIR, ignore_errors=True)
    link_tree(base_dir / AUX_DIR, final / AUX_DIR)
    write_init_meta(final, backend=BACKEND, base_model=cfg.base_model, method="full", format=CKPT_FORMAT, step=step)
    mark_complete(final)


# ---- run-level helpers -------------------------------------------------------------------------------------------

def prepared_dir(rd: RunDir, role: str = "train") -> Path:
    ref = next((r for r in read_prepared_refs(rd) if r.get("role") == role), None)
    if ref is None:
        raise RuntimeError(f"no prepared {role} dataset (prepared_refs.json has no {role} entry)")
    return Path(ref["path"])


def base_model_dir(cfg: RunConfig) -> Path:
    from .cosyvoice import download_model

    return download_model(cfg.base_model)


def check_init(rd: RunDir, cfg: RunConfig) -> None:
    """Warm start must be a cosyvoice-pt checkpoint holding llm.pt / flow.pt / hift.pt (a run's `final`)."""
    choice = init_source(cfg.model_copy(update={"resume": False}), rd)
    if choice is None:
        return
    fmt = (choice.meta or {}).get("format")
    if fmt != CKPT_FORMAT:
        raise ValueError(f"unsupported init checkpoint format '{fmt}' for backend {BACKEND}: only {CKPT_FORMAT} "
                         "checkpoints can be used")
    if not any((choice.path / f).is_file() for f in WEIGHT_FILE.values()):
        raise ValueError("init checkpoint has no llm.pt / flow.pt / hift.pt: use a run's final checkpoint")


def weights_for_start(rd: RunDir, cfg: RunConfig, base_dir: Path, component: str) -> Path:
    return component_sources(rd, cfg, base_dir, {})[WEIGHT_FILE[component]]


def patched_yaml(base_dir: Path, work: Path, hp: dict) -> Path:
    """cosyvoice3.yaml of the snapshot, with the batch size (max_frames_in_batch) overridden by the hparam."""
    text = (Path(base_dir) / "cosyvoice3.yaml").read_text(encoding="utf-8")
    if hp.get("max_frames_in_batch"):
        text = re.sub(r"max_frames_in_batch:\s*\d+", f"max_frames_in_batch: {int(hp['max_frames_in_batch'])}", text)
    out = Path(work) / "cosyvoice3.yaml"
    out.write_text(text, encoding="utf-8")
    return out


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def load_configs(component: str, yaml_path: Path, base_dir: Path) -> dict:
    """upstream train.py: only the trained model is built (the others are overridden with None)."""
    from hyperpyyaml import load_hyperpyyaml

    gan = component == "hifigan"
    overrides = {k: None for k in ("llm", "flow", "hift", "hifigan") if k != component}
    if gan:
        overrides.pop("hift")
    overrides["qwen_pretrain_path"] = str(Path(base_dir) / AUX_DIR)
    with open(yaml_path, "r", encoding="utf-8") as f:
        configs = load_hyperpyyaml(f, overrides=overrides)
    if gan:
        configs["train_conf"] = configs["train_conf_gan"]
    return configs


def load_weights(model, component: str, path: Path) -> tuple[int, int]:
    """Load a base / init / own checkpoint into the model; returns (step, epoch) stored in it (0, -1 if none).
    A plain hift.pt (generator keys without prefix) goes into `model.generator`. Missing keys fail loudly."""
    import torch

    sd = torch.load(str(path), map_location="cpu", weights_only=True)
    step, epoch = int(sd.pop("step", 0)), int(sd.pop("epoch", -1))
    target = model
    if component == "hifigan" and not any(k.startswith("generator.") for k in sd):
        target = model.generator
    res = target.load_state_dict(sd, strict=False)
    if res.missing_keys:
        shown = ", ".join(res.missing_keys[:20])
        raise RuntimeError(f"{path} does not match the {component} model: missing keys [{shown}]"
                           + (f" (+{len(res.missing_keys) - 20} more)" if len(res.missing_keys) > 20 else ""))
    if res.unexpected_keys:
        print(f"cosyvoice: {len(res.unexpected_keys)} unexpected keys ignored in {path.name}", file=sys.stderr)
    return step, epoch


_ENV_KEYS = ("MASTER_ADDR", "MASTER_PORT", "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "onnx_path")


def snapshot_env() -> dict[str, str | None]:
    return {k: os.environ.get(k) for k in _ENV_KEYS}


def restore_env(snap: dict[str, str | None]) -> None:
    for k, v in snap.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def train_metrics(info: dict, gan: bool, seen: dict) -> list[tuple[str, float]]:
    """(name, value) train metrics of one upstream log_per_step call; [] when it is not an optimizer-step boundary.
    Upstream's loss is divided by accum_grad in batch_backward: multiplied back here. The GAN loop logs twice per batch
    (discriminator turn, then generator turn): `loss_d` / `loss_g`, with lr only on the generator turn."""
    if info["tag"] != "TRAIN" or (info["batch_idx"] + 1) % info["accum_grad"] != 0:
        return []
    ld = info["loss_dict"]
    loss = float(ld["loss"]) * info["accum_grad"]
    if not gan:
        out = [("loss", loss)]
        if "acc" in ld:
            out.append(("acc", float(ld["acc"])))
        return out + [("lr", float(info["lr"]))]
    key = (info.get("epoch"), info["batch_idx"])
    generator = seen.get("batch") == key  # second call for this batch
    seen["batch"] = key
    return [("loss_g", loss), ("lr", float(info["lr"]))] if generator else [("loss_d", loss)]


def make_component_final(rd: RunDir, cfg: RunConfig, component: str) -> Path:
    """`<component>-final/model.pt` from the latest complete epoch dir; the epoch dirs are removed afterwards."""
    import torch

    src_dir = latest_complete(rd, component)
    if src_dir is None:
        raise RuntimeError(f"no complete {component} checkpoint to finalize")
    meta = read_init_meta(src_dir) or {}
    sd = torch.load(str(src_dir / "model.pt"), map_location="cpu", weights_only=True)
    out = final_component_dir(rd, component)
    out.mkdir(parents=True, exist_ok=True)
    (out / ".complete").unlink(missing_ok=True)
    tmp = out / "model.pt.tmp"
    torch.save(clean_state(sd, component), str(tmp))
    del sd
    os.replace(tmp, out / "model.pt")
    write_init_meta(out, backend=BACKEND, base_model=cfg.base_model, method="full", format=COMPONENT_FORMAT,
                    step=int(meta.get("step", 0)))
    mark_complete(out)
    for _, d, _ in checkpoint_dirs(rd, component):
        shutil.rmtree(d, ignore_errors=True)
    return out


class _Halt:
    """Pending halt: 'stop' (SIGTERM) or 'max' (max_steps reached)."""

    def __init__(self) -> None:
        self.reason: str | None = None


# ---- one component -----------------------------------------------------------------------------------------------

def train_component(name: str, cfg: RunConfig, rd: RunDir, rep, stop: StopFlag, base_dir: Path | None = None) -> int:
    """Train one component; 0 = finished (`<name>-final` complete), 143 = stopped (resumable checkpoint saved)."""
    import gc
    from argparse import Namespace
    from copy import deepcopy

    import torch
    import torch.distributed as dist

    hp = cfg.hparams
    if stop.requested:
        return 143
    base_dir = Path(base_dir) if base_dir is not None else base_model_dir(cfg)
    prepared = prepared_dir(rd, "train")
    train_list, cv_list = prepared / "train" / "data.list", prepared / "val" / "data.list"
    for p in (train_list, cv_list):
        if not p.is_file():
            raise RuntimeError(f"prepared dataset has no {p.parent.name}/data.list ({p})")
    work = rd.path / "work" / f"cosyvoice-{name}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    own = latest_complete(rd, name) if cfg.resume else None
    if not cfg.resume:  # a fresh attempt never mixes with leftovers of an earlier one
        for _, d, _ in checkpoint_dirs(rd, name):
            shutil.rmtree(d, ignore_errors=True)
        shutil.rmtree(final_component_dir(rd, name), ignore_errors=True)

    gan = name == "hifigan"
    max_steps = int(hp["max_steps"]) if hp.get("max_steps") else None
    epochs = max(1, math.ceil(float(hp.get("epochs", 10))))
    accum = 1 if gan else max(1, int(hp.get("grad_accum", 2)))
    warmup = int(hp.get("warmup_steps", 2500))
    lr = float(hp.get("learning_rate", 1e-5))
    use_amp = torch.cuda.is_available()
    if not use_amp:
        raise RuntimeError("CosyVoice fine-tuning needs a CUDA GPU")

    env_snap = snapshot_env()
    os.environ.update({"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(_free_port()), "RANK": "0", "LOCAL_RANK": "0",
                       "WORLD_SIZE": "1", "LOCAL_WORLD_SIZE": "1"})
    use_offline_features()  # tokens/embeddings come from prepare; online mode needs an unpublished batch onnx
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)

    ensure_deepspeed_stub()
    from cosyvoice.utils import executor as ex_mod
    from cosyvoice.utils.executor import Executor
    from cosyvoice.utils.train_utils import (
        check_modify_and_save_config, init_dataset_and_dataloader, init_distributed, init_optimizer_and_scheduler,
        wrap_cuda_model,
    )

    yaml_path = patched_yaml(base_dir, work, hp)
    configs = load_configs(name, yaml_path, base_dir)
    torch.manual_seed(cfg.seed)
    tc = configs["train_conf"]
    tc["max_epoch"] = epochs if max_steps is None else 10 ** 6
    tc["accum_grad"] = accum
    tc["save_per_step"] = int(hp["save_steps"]) if hp.get("save_steps") else -1
    tc["log_interval"] = 10
    tc["optim_conf"]["lr"] = lr
    if gan:
        tc["optim_conf_d"]["lr"] = lr
    elif warmup > 0:
        tc["scheduler"], tc["scheduler_conf"] = "warmuplr", {"warmup_steps": warmup}
    else:
        tc["scheduler"] = "constantlr"
    args = Namespace(train_engine="torch_ddp", model=name, config=str(yaml_path), train_data=str(train_list),
                     cv_data=str(cv_list), qwen_pretrain_path=str(base_dir / AUX_DIR), onnx_path=str(base_dir),
                     checkpoint=None, model_dir=str(work), tensorboard_dir=str(work / "tb"), dist_backend="nccl",
                     num_workers=NUM_WORKERS, prefetch=PREFETCH, pin_memory=True, use_amp=use_amp, dpo=False,
                     save_states="model_only", timeout=60)
    tc.update(vars(args))

    try:
        init_distributed(args)
    except Exception as e:  # noqa: BLE001 - the free port can be taken between probe and bind: retry once
        print(f"cosyvoice: init_distributed failed ({e}); retrying on a new port", file=sys.stderr)
        if dist.is_initialized():
            dist.destroy_process_group()
        os.environ["MASTER_PORT"] = str(_free_port())
        init_distributed(args)
    halted = None
    model = optimizer = scheduler = optimizer_d = scheduler_d = scaler = executor = None
    train_loader = cv_loader = train_dataset = info_dict = None
    try:
        train_dataset, _, train_loader, cv_loader = init_dataset_and_dataloader(args, configs, gan, False)
        configs = check_modify_and_save_config(args, configs)
        model = configs[name]
        # weights: own checkpoint (resume) > warm-start init dir > base snapshot
        if own is not None:
            start_step, start_epoch = load_weights(model, name, own / "model.pt")
        else:
            load_weights(model, name, weights_for_start(rd, cfg, base_dir, name))
            start_step, start_epoch = 0, -1
        model = wrap_cuda_model(args, model)
        model, optimizer, scheduler, optimizer_d, scheduler_d = init_optimizer_and_scheduler(args, configs, model, gan)
        scheduler.set_step(start_step)
        if scheduler_d is not None:
            scheduler_d.set_step(start_step)
        info_dict = deepcopy(tc)
        info_dict["step"], info_dict["epoch"] = start_step, start_epoch
        executor = Executor(gan=gan)
        executor.step = start_step
        scaler = torch.cuda.amp.GradScaler() if use_amp else None
        halted = _run_epochs(name, cfg, rd, rep, stop, ex_mod, executor, model, optimizer, scheduler, optimizer_d, scheduler_d,
                    train_dataset, train_loader, cv_loader, info_dict, scaler, work, start_step, start_epoch,
                    max_steps, epochs, accum)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        restore_env(env_snap)
    # drop every reference to GPU tensors before emptying the cache
    model = optimizer = scheduler = optimizer_d = scheduler_d = scaler = executor = None
    train_loader = cv_loader = train_dataset = info_dict = configs = None
    gc.collect()
    torch.cuda.empty_cache()
    if halted == "stop":
        return 143
    make_component_final(rd, cfg, name)
    shutil.rmtree(work, ignore_errors=True)
    return 0


def _run_epochs(name, cfg, rd, rep, stop, ex_mod, executor, model, optimizer, scheduler, optimizer_d, scheduler_d,
                train_dataset, train_loader, cv_loader, info_dict, scaler, work, start_step, start_epoch,
                max_steps, epochs, accum) -> str | None:
    """Run the epoch loop with the executor patches; returns the halt reason ('stop', 'max') or None."""
    gan = name == "hifigan"
    hp = cfg.hparams
    halt = _Halt()
    nan = NanGuard()
    total = {"v": max_steps}
    orig = {k: getattr(ex_mod, k) for k in ("log_per_step", "log_per_save", "save_model", "cosyvoice_join")}

    def publish(model_, kind: str, epoch_: int, step_: int) -> None:
        """Save with upstream save_model into the work dir, then publish as a contract checkpoint dir."""
        stored_epoch = epoch_ if kind == "epoch" else epoch_ - 1  # a mid-epoch save makes resume repeat that epoch
        orig["save_model"](model_, "save", {"model_dir": str(work), "train_engine": "torch_ddp",
                                            "epoch": stored_epoch, "step": step_})
        (work / "save.yaml").unlink(missing_ok=True)
        dst = epoch_dir(rd, name, epoch_ + 1, None if kind == "epoch" else step_)
        publish_checkpoint(work / "save.pt", dst, cfg, step_)
        prune_checkpoints(rd, name, keep_count(hp))

    def save_model(model_, model_name, info):
        e = int(info["epoch"])
        if model_name.endswith("_whole"):
            publish(model_, "epoch", e, executor.step)
        elif "_step_" in model_name:
            publish(model_, "step", e, int(model_name.rsplit("_", 1)[1]))
        else:  # "halt"
            publish(model_, "step", e, executor.step)

    gan_seen: dict = {}

    def log_per_step(writer, info):
        orig["log_per_step"](writer, info)
        metrics = train_metrics(info, gan, gan_seen)
        if not metrics:
            return
        step = int(info["step"]) + 1  # optimizer updates done (executor.step is incremented right after this call)
        for mname, value in metrics:
            rep.metric(step, "train", mname, value)
            if mname.startswith("loss"):
                nan.feed(value)
        rep.status(Phase.train, step, total["v"], f"{name}: epoch {int(info['epoch']) + 1}")
        if max_steps is not None and step >= max_steps:
            halt.reason = "max"
        elif stop.requested:
            halt.reason = "stop"

    def log_per_save(writer, info):
        orig["log_per_save"](writer, info)
        loss = info["loss_dict"].get("loss")
        if loss is not None:
            rep.metric(executor.step, "eval", f"eval_loss_{name}", float(loss))

    def cosyvoice_join(group, info):
        if stop.requested and halt.reason is None:
            halt.reason = "stop"  # checked at every batch start: bounded stop latency
        return halt.reason is not None

    orig_cv = executor.cv

    def cv(model_, loader, writer, info, on_batch_end=True):
        if halt.reason is not None:
            save_model(model_, "halt", info)  # the halted epoch is not finished: skip the CV pass, save and leave
            return None
        return orig_cv(model_, loader, writer, info, on_batch_end=on_batch_end)

    had_cv = "cv" in vars(executor)
    executor.cv = cv
    ex_mod.log_per_step, ex_mod.log_per_save = log_per_step, log_per_save
    ex_mod.save_model, ex_mod.cosyvoice_join = save_model, cosyvoice_join
    try:
        if max_steps is not None and start_step >= max_steps:
            return None
        for epoch in range(start_epoch + 1, info_dict["max_epoch"]):
            executor.epoch = epoch
            train_dataset.set_epoch(epoch)
            if gan:
                executor.train_one_epoc_gan(model, optimizer, scheduler, optimizer_d, scheduler_d, train_loader,
                                            cv_loader, None, info_dict, scaler, None)
            else:
                executor.train_one_epoc(model, optimizer, scheduler, train_loader, cv_loader, None, info_dict, scaler,
                                        None)
            if halt.reason is None and stop.requested:
                halt.reason = "stop"  # the epoch (and its end-of-epoch save) is complete: nothing more to save
            if halt.reason is not None:
                break
            if total["v"] is None:  # steps per epoch are known after the first epoch of this attempt
                total["v"] = start_step + (executor.step - start_step) * (epochs - start_epoch - 1)
        return halt.reason
    finally:
        if had_cv:
            executor.cv = orig_cv
        else:
            del executor.cv
        ex_mod.log_per_step, ex_mod.log_per_save = orig["log_per_step"], orig["log_per_save"]
        ex_mod.save_model, ex_mod.cosyvoice_join = orig["save_model"], orig["cosyvoice_join"]


# ---- phase handler -----------------------------------------------------------------------------------------------

def train_cosyvoice(rd: RunDir, cfg: RunConfig, rep) -> int:
    hp = cfg.hparams
    requested = resolve_components(hp)
    stop = StopFlag().install()
    check_init(rd, cfg)
    completed = [c for c in requested if cfg.resume and is_complete(final_component_dir(rd, c))]
    base_dir = base_model_dir(cfg)
    for name in plan_components(requested, completed):
        rep.status(Phase.train, 0, None, f"{name}: starting")
        if train_component(name, cfg, rd, rep, stop, base_dir) == 143:
            return 143
    trained = {c: final_component_dir(rd, c) / "model.pt" for c in requested}
    steps = sum(int((read_init_meta(final_component_dir(rd, c)) or {}).get("step", 0)) for c in requested)
    assemble_final(rd.checkpoints_dir / "final", component_sources(rd, cfg, base_dir, trained), base_dir, cfg, steps)
    return 0
