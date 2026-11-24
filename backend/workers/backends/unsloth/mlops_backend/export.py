"""LLM export for the unsloth backend: merged safetensors and GGUF from checkpoints/final.

unsloth is imported inside functions only. GGUF needs llama.cpp, which unsloth builds on first use
(cmake and build-essential must exist on the machine).
"""
from __future__ import annotations

from pathlib import Path

from app.core.backend_kit import export_all
from app.core.contract import ExportArtifact, RunConfig, RunDir

DEFAULT_GGUF_QUANT = "q4_k_m"


def _method_type(cfg: RunConfig) -> str:
    return str(cfg.method.get("type", "lora"))


def gguf_quant(cfg: RunConfig) -> str:
    return str(cfg.method.get("gguf_quant", DEFAULT_GGUF_QUANT))


def _artifact(rd: RunDir, fmt: str, path: Path, serve_with: str) -> ExportArtifact:
    return ExportArtifact(format=fmt, path=path.relative_to(rd.export_dir).as_posix(),
                          size_bytes=path.stat().st_size, serve_with=serve_with)


def _dir_artifact(rd: RunDir, fmt: str, d: Path, serve_with: str) -> ExportArtifact:
    size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
    return ExportArtifact(format=fmt, path=d.relative_to(rd.export_dir).as_posix(),
                          size_bytes=size, serve_with=serve_with)


def find_gguf(out: Path, quant: str) -> Path:
    """The .gguf produced by save_pretrained_gguf (inside `out`, or in the sibling `<out>_gguf` of older unsloth)."""
    found: list[Path] = []
    for d in (out, out.with_name(out.name + "_gguf")):
        if d.is_dir():
            found.extend(d.glob("*.gguf"))
    if not found:
        raise RuntimeError(f"gguf export produced no .gguf file under {out}")
    # unsloth may leave an intermediate f16/bf16 file next to the quantized one; prefer the requested quant.
    wanted = [p for p in found if quant.lower() in p.name.lower()]
    return max(wanted or found, key=lambda p: p.stat().st_mtime)


def llm_exporters(rd: RunDir, cfg: RunConfig) -> dict:
    final = rd.checkpoints_dir / "final"
    state: dict = {}

    def loaded():
        if "m" not in state:
            if not (final / ".complete").exists():
                raise RuntimeError(f"no completed final checkpoint at {final}")
            import unsloth  # noqa: F401  (must precede transformers/trl/datasets imports)
            from unsloth import FastLanguageModel

            state["m"] = FastLanguageModel.from_pretrained(
                str(final), max_seq_length=int(cfg.hparams.get("max_seq_len", 2048)),
                load_in_4bit=_method_type(cfg) == "qlora")
        return state["m"]

    def merged(out: Path) -> list[ExportArtifact]:
        model, tok = loaded()
        if hasattr(model, "peft_config"):
            model.save_pretrained_merged(str(out), tok, save_method="merged_16bit")
        else:  # full fine-tune: no adapter to merge
            model.save_pretrained(str(out))
            tok.save_pretrained(str(out))
        return [_dir_artifact(rd, "merged-safetensors", out, "transformers")]

    def gguf(out: Path) -> list[ExportArtifact]:
        model, tok = loaded()
        quant = gguf_quant(cfg)
        model.save_pretrained_gguf(str(out), tok, quantization_method=quant)
        return [_artifact(rd, "gguf", find_gguf(out, quant), "llama-server")]

    return {"merged-safetensors": merged, "gguf": gguf}


def export(rd: RunDir, cfg: RunConfig, rep) -> int:
    return export_all(rd, cfg, rep, llm_exporters(rd, cfg))
