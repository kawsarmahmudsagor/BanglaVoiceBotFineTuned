from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dataprep.__main__ import main
from dataprep.sources.voice import import_voice
from dataprep.workdir import StepFile, WorkDir
from fixtures import make_voice


def test_import_voice_columns_and_source_kind(tmp_path):
    src = make_voice(tmp_path / "src", [
        {"name": "a", "text": "আমি", "category": "greeting"},
        {"name": "b", "text": "তুমি", "category": "price"},
    ])
    wd = WorkDir(tmp_path / "wd")
    assert import_voice(src, wd) == 2
    df = pd.read_parquet(wd.clips_path)
    assert list(df.columns) == ["id", "src_path", "text_raw", "src_split", "category"]
    assert (df.src_split == "").all()
    assert df.src_path.map(lambda s: Path(s).is_absolute() and Path(s).is_file()).all()
    assert df.set_index("id").category["b"] == "price"
    assert StepFile._meta(wd.clips_path)["params"]["source"] == "voice"


def test_import_voice_without_extra_columns(tmp_path):
    src = make_voice(tmp_path / "src", [{"name": "a", "text": "আমি"}])
    import_voice(src, WorkDir(tmp_path / "wd"))
    assert list(pd.read_parquet(tmp_path / "wd" / "clips.parquet").columns) == ["id", "src_path", "text_raw", "src_split"]


def test_import_voice_missing_audio_is_reported(tmp_path, capsys):
    src = make_voice(tmp_path / "src", [{"name": "a", "text": "x"}, {"name": "b", "text": "y"}])
    (src / "wavs" / "b.wav").unlink()
    assert import_voice(src, WorkDir(tmp_path / "wd")) == 1
    assert "1 missing audio" in capsys.readouterr().out


def test_import_voice_wrong_dir(tmp_path):
    with pytest.raises(SystemExit, match="metadata.csv"):
        main(["import", "voice", "--src", str(tmp_path), "--work", str(tmp_path / "wd")])


def test_import_voice_duplicate_ids_fail(tmp_path):
    src = make_voice(tmp_path / "src", [{"name": "a", "text": "x"}])
    with open(src / "metadata.csv", "a", encoding="utf-8") as f:
        f.write("wavs/a.wav,again\n")
    with pytest.raises(SystemExit, match="duplicate"):
        import_voice(src, WorkDir(tmp_path / "wd"))
