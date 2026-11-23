import pytest
from pydantic import ValidationError

from mlops_shared.recipes import (
    BaseModelMeta,
    Capabilities,
    RecipeDatasets,
    RecipeSpec,
    TaskCapability,
    commercial_ok,
    parse_dataset_ref,
    validate_recipe,
)


def make_spec(**over) -> RecipeSpec:
    base = dict(
        name="qwen-4b-lora",
        task="llm-sft",
        backend="fake",
        base_model="Qwen/Qwen3-4B",
        base_model_meta=BaseModelMeta(license="apache-2.0", commercial_ok=True),
        datasets=RecipeDatasets(train="ecom-instruct@v1"),
        method={"type": "none"},
        hparams={"steps": 5},
        export=[],
    )
    base.update(over)
    return RecipeSpec(**base)


def test_parse_dataset_ref():
    assert parse_dataset_ref("ecom-instruct@v3") == ("ecom-instruct", 3)
    for bad in ["ecom", "ecom@3", "ecom@v0", "Ecom@v1"]:
        with pytest.raises(ValueError):
            parse_dataset_ref(bad)


def test_validate_recipe_reports_each_problem():
    caps = {"fake": Capabilities(name="fake", label="Fake", tasks={"llm-sft": TaskCapability(
        methods=["none"], export_formats=["dummy"], serve_formats=[],
        schema={"type": "object", "properties": {"hparams": {"type": "object",
                "properties": {"steps": {"type": "integer", "minimum": 1}}}}})})}
    spec = make_spec(task="asr", backend="fake", method={"type": "lora"},
                     hparams={"steps": 0}, export=["gguf:q4_k_m"])
    errs = validate_recipe(spec, caps)
    assert any("task 'asr' not supported by backend 'fake'" in e for e in errs)
    errs = validate_recipe(make_spec(task="llm-sft", backend="fake", method={"type": "lora"},
                                     hparams={"steps": 0}, export=["gguf:q4_k_m"]), caps)
    assert "method 'lora' not supported" in " ".join(errs)
    assert "export format 'gguf' not supported" in " ".join(errs)
    assert any(e.startswith("hparams.steps") for e in errs)
    assert validate_recipe(make_spec(backend="nope"), caps) == ["unknown backend 'nope'"]


def test_validate_recipe_valid_returns_empty():
    caps = {"fake": Capabilities(name="fake", label="Fake", tasks={"llm-sft": TaskCapability(
        methods=["none"], export_formats=["dummy"], serve_formats=[],
        schema={"type": "object"})})}
    assert validate_recipe(make_spec(export=["dummy"]), caps) == []


def test_commercial_ok():
    yes = BaseModelMeta(license="apache-2.0", commercial_ok=True)
    assert commercial_ok(yes, [True, True])
    assert not commercial_ok(yes, [True, False])
    assert not commercial_ok(BaseModelMeta(license="cc-by-nc", commercial_ok=False), [True])


def test_recipe_name_must_be_slug():
    with pytest.raises(ValidationError):
        make_spec(name="Qwen 4B")


def test_method_requires_type():
    with pytest.raises(ValidationError):
        make_spec(method={})


def test_seed_property():
    assert make_spec(hparams={}).seed == 42
    assert make_spec(hparams={"seed": 7}).seed == 7
