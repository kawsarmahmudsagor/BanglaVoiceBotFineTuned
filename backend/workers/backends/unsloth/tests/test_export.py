import pytest

from app.core.contract import read_config

from mlops_backend.export import find_gguf, gguf_quant, llm_exporters

QWEN = "Qwen/Qwen3-4B-Instruct-2507"


def test_exporters_cover_llm_formats(run_dir):
    rd = run_dir("llm-sft", QWEN)
    assert set(llm_exporters(rd, read_config(rd))) == {"merged-safetensors", "gguf"}


def test_exporter_requires_complete_final(run_dir):
    rd = run_dir("llm-sft", QWEN)
    (rd.checkpoints_dir / "final").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="no completed final checkpoint"):
        llm_exporters(rd, read_config(rd))["gguf"](rd.export_dir / "gguf")


def test_gguf_quant_default_and_override(run_dir):
    cfg = read_config(run_dir("llm-sft", QWEN))
    assert gguf_quant(cfg) == "q4_k_m"
    assert gguf_quant(cfg.model_copy(update={"method": {"type": "lora", "gguf_quant": "q8_0"}})) == "q8_0"


def test_find_gguf(tmp_path):
    out = tmp_path / "gguf"
    with pytest.raises(RuntimeError, match="no .gguf"):
        find_gguf(out, "q4_k_m")
    out.mkdir()
    (out / "model-unsloth.F16.gguf").write_bytes(b"1")
    (out / "model-unsloth.Q4_K_M.gguf").write_bytes(b"2")
    assert find_gguf(out, "q4_k_m").name.endswith("Q4_K_M.gguf")


@pytest.mark.heavy
def test_llm_export_gguf_file_exists(run_dir):
    from app.core.contract import read_export_manifest
    from mlops_backend.export import export
    from test_llm_eval import _heavy_run
    rd, cfg, rep = _heavy_run(run_dir, {})
    cfg = cfg.model_copy(update={"export": ["merged-safetensors", "gguf"]})
    assert export(rd, cfg, rep) == 0
    arts = {a.format: a for a in read_export_manifest(rd).artifacts}
    assert set(arts) == {"merged-safetensors", "gguf"}
    assert arts["merged-safetensors"].serve_with == "transformers"
    assert arts["gguf"].serve_with == "llama-server"
    assert (rd.export_dir / arts["gguf"].path).is_file() and arts["gguf"].path.endswith(".gguf")
