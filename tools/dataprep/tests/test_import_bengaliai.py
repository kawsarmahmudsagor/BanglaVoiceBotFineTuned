from pathlib import Path

import pandas as pd
import pytest

from dataprep.__main__ import main
from dataprep.sources.bengaliai import import_bengaliai
from dataprep.workdir import WorkDir
from fixtures import make_bengaliai


def test_import_bengaliai(tmp_path):
    src = make_bengaliai(tmp_path / "src", [("a", "আমি", "train"), ("b", "তুমি", "valid")])
    n = import_bengaliai(src, WorkDir(tmp_path / "wd"))
    df = pd.read_parquet(tmp_path / "wd" / "clips.parquet")
    assert n == 2 and sorted(df.src_split) == ["train", "val"]
    assert list(df.columns) == ["id", "src_path", "text_raw", "src_split"]
    assert df.src_path.map(lambda s: Path(s).is_absolute() and Path(s).is_file()).all()
    assert df.set_index("id").text_raw["a"] == "আমি"


def test_import_bengaliai_missing_audio_is_reported(tmp_path, capsys):
    src = make_bengaliai(tmp_path / "src", [("a", "x", "train"), ("b", "y", "train")], with_audio={"a"})
    n = import_bengaliai(src, WorkDir(tmp_path / "wd"))
    assert n == 1
    assert pd.read_parquet(tmp_path / "wd" / "clips.parquet").id.tolist() == ["a"]
    assert "1 missing audio" in capsys.readouterr().out


def test_import_bengaliai_wrong_dir_names_expected_files(tmp_path):
    with pytest.raises(SystemExit, match="train.csv") as e:
        main(["import", "bengaliai", "--src", str(tmp_path), "--work", str(tmp_path / "wd")])
    assert "train_mp3s" in str(e.value)


def test_import_bengaliai_cli(tmp_path):
    src = make_bengaliai(tmp_path / "src", [("a", "x", "valid")])
    assert main(["import", "bengaliai", "--src", str(src), "--work", str(tmp_path / "wd")]) == 0
    assert (tmp_path / "wd" / "clips.parquet").is_file()
