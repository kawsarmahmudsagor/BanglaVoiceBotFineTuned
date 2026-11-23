import soundfile as sf
from app.core.contract import read_config
from app.core.prepare import iter_rows


def test_run_dir_llm(run_dir):
    rd = run_dir("llm-sft", "m", {"epochs": 1})
    cfg = read_config(rd)
    assert cfg.hparams == {"epochs": 1}
    rows = list(iter_rows(cfg.datasets["train"], "train"))
    assert len(rows) == 2 and rows[0]["messages"][0]["role"] == "user"


def test_run_dir_asr_audio(run_dir):
    cfg = read_config(run_dir("asr", "m"))
    rows = list(iter_rows(cfg.datasets["train"], "val"))
    assert sf.info(rows[0]["audio_path"]).samplerate == 16000


def test_tiny_audio_ds(tiny_audio_ds):
    root, names = tiny_audio_ds(sr=8000)
    assert len(names) == 4 and sf.info(str(root / "files" / names[0])).samplerate == 8000
