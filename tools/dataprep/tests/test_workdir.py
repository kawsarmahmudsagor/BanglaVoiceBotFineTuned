import pandas as pd
import pytest

from dataprep.pool import run_rows
from dataprep.workdir import StaleStepError, StepFile, WorkDir


def test_stepfile_roundtrip_and_stale(tmp_path):
    p = tmp_path / "s.parquet"
    StepFile.write(p, pd.DataFrame({"id": ["a"], "x": [1], "error": [None]}), step="s", version=1, params={"k": 1})
    assert StepFile.read(p, step="s", version=1, params={"k": 1}).x.tolist() == [1]
    with pytest.raises(StaleStepError):
        StepFile.read(p, step="s", version=2)
    with pytest.raises(StaleStepError):
        StepFile.read(p, step="s", version=1, params={"k": 2})
    with pytest.raises(StaleStepError):
        StepFile.read(p, step="other", version=1)


def test_done_ids_excludes_errors(tmp_path):
    p = tmp_path / "s.parquet"
    df = pd.DataFrame({"id": ["a", "b"], "error": [None, "ValueError: x"]})
    StepFile.write(p, df, step="s", version=1, params={})
    assert StepFile.done_ids(p) == {"a"}


def test_workdir_paths(tmp_path):
    wd = WorkDir(tmp_path / "w")
    assert wd.clips_path == tmp_path / "w" / "clips.parquet"
    assert wd.step_path("text") == tmp_path / "w" / "text.parquet"
    assert wd.selection_path("tts") == tmp_path / "w" / "selection" / "tts.parquet"
    assert wd.export_dir("ds") == tmp_path / "w" / "export" / "ds"


def test_step_records_row_errors():
    out = run_rows(lambda r: 1 / 0 if r["id"] == "b" else {"id": r["id"], "v": 1}, [{"id": "a"}, {"id": "b"}], workers=1, desc="t")
    assert {o["id"]: o["error"] for o in out} == {"a": None, "b": "ZeroDivisionError: division by zero"}
