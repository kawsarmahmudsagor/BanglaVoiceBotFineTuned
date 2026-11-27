import pytest

from ytds import models
from ytds.config import Settings
from ytds.store import Store


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "ytds.db")


def test_options_and_defaults():
    assert [(o.id, o.slot) for o in models.OPTIONS] == [
        ("qwen3-asr-bn-ft1", "a"), ("qwen3-asr-bn-sayedshaun", "a"),
        ("whisper-bn-ft1", "b"), ("whisper-bn-mozilla", "b")]
    assert models.DEFAULTS == {"a": "qwen3-asr-bn-ft1", "b": "whisper-bn-ft1"}
    assert all(o.label for o in models.OPTIONS)


def test_available_missing_folder_returns_expected_path(tmp_path):
    missing = str(tmp_path / "nope")
    s = Settings(data_dir=tmp_path, model_a_path=missing, model_b_path=missing)
    assert models.available("qwen3-asr-bn-ft1", s) == (False, missing)
    assert models.available("whisper-bn-ft1", s) == (False, missing)


def test_available_existing_folder_and_repo_ids(tmp_path):
    s = Settings(data_dir=tmp_path, model_a_path=str(tmp_path), model_b_path="someone/whisper-ct2")
    assert models.available("qwen3-asr-bn-ft1", s) == (True, str(tmp_path))
    assert models.available("whisper-bn-ft1", s) == (True, "someone/whisper-ct2")
    assert models.available("qwen3-asr-bn-sayedshaun", s) == (True, "SayedShaun/Qwen3-ASR-0.6B-Bengali")
    # no conversion is triggered just to report availability
    assert models.available("whisper-bn-mozilla", s) == (True, "mozilla-ai/whisper-large-v3-turbo-bn")
    assert not (tmp_path / "models").exists()


@pytest.mark.parametrize("value", ["a/b/c", "C:/x", "a\\b", "plain"])
def test_non_repo_id_values_need_a_folder(tmp_path, value):
    s = Settings(data_dir=tmp_path, model_b_path=value)
    assert models.available("whisper-bn-ft1", s)[0] is False


def test_unknown_option_raises(tmp_path):
    with pytest.raises(KeyError):
        models.available("nope", Settings(data_dir=tmp_path))


def test_current_reads_setting_or_default(store):
    assert models.current(store, "a") == "qwen3-asr-bn-ft1"
    assert models.current(store, "b") == "whisper-bn-ft1"
    store.set_setting("model_b", "whisper-bn-mozilla")
    assert models.current(store, "b") == "whisper-bn-mozilla"
    store.set_setting("model_a", "whisper-bn-mozilla")  # stale or wrong slot -> default
    assert models.current(store, "a") == "qwen3-asr-bn-ft1"


def test_resolve(tmp_path):
    s = Settings(data_dir=tmp_path, model_a_path="A", model_b_path="B")
    assert models.resolve("qwen3-asr-bn-ft1", s) == "A"
    assert models.resolve("whisper-bn-ft1", s) == "B"
    assert models.resolve("qwen3-asr-bn-sayedshaun", s) == "SayedShaun/Qwen3-ASR-0.6B-Bengali"
    cached = tmp_path / "models" / "mozilla-ai--whisper-large-v3-turbo-bn-float16"
    cached.mkdir(parents=True)
    (cached / "model.bin").write_bytes(b"")  # already converted: no conversion runs
    assert models.resolve("whisper-bn-mozilla", s) == str(cached)


def test_loader_reads_choice_at_load_time(store, tmp_path, monkeypatch):
    loaded = []

    def fake_a(path):
        loaded.append(("a", path))
        return lambda audios: ["x"] * len(audios)

    def fake_b(path):
        loaded.append(("b", path))
        return lambda audio, pieces: [{"text": "y"} for _ in pieces]

    monkeypatch.setattr("ytds.pipeline.transcribe.load_model_a", fake_a)
    monkeypatch.setattr("ytds.pipeline.whisper_b.load_whisper", fake_b)
    s = Settings(data_dir=tmp_path, model_a_path="A", model_b_path="B")
    load_a, load_b = models.loader("a", store, s), models.loader("b", store, s)
    assert loaded == []  # building the loader loads nothing

    store.set_setting("model_a", "qwen3-asr-bn-sayedshaun")
    a = load_a()
    assert a.option_id == "qwen3-asr-bn-sayedshaun" and a([1, 2]) == ["x", "x"]
    b = load_b()
    assert b.option_id == "whisper-bn-ft1" and b(None, [{}]) == [{"text": "y"}]
    assert loaded == [("a", "SayedShaun/Qwen3-ASR-0.6B-Bengali"), ("b", "B")]

    store.set_setting("model_b", "whisper-bn-mozilla")
    assert b.option_id == "whisper-bn-ft1"  # an already loaded model keeps its id
