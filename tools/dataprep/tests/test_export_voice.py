from __future__ import annotations

import csv
import hashlib
import json

import pandas as pd
import pytest
from app.datasets.validate_audio import validate_audio

from dataprep.__main__ import main
from dataprep.export import ExportError, export, pick_test_ids
from dataprep.profiles import Profile, load_profile
from dataprep.select import select
from dataprep.sources.voice import import_voice
from dataprep.steps import audio_stats, text
from dataprep.workdir import StepFile, WorkDir
from fixtures import make_voice

PROFILE = Profile(name="t", task="tts")
PROV = {"source": "Recorded in-house", "license": "contract 2026-11", "consent": "written consent on file",
        "commercial_ok": True, "notes": "stage-B voice"}


def _h(i: str) -> str:
    return hashlib.sha256(i.encode("utf-8")).hexdigest()


def make_wd(tmp_path, n=12, categories=("greet", "price", "date")) -> WorkDir:
    rows = [{"name": f"t{i:02d}", "text": f"আমি ভাত খাই {i}"} | ({"category": categories[i % len(categories)]} if categories else {})
            for i in range(n)]
    src = make_voice(tmp_path / "src", rows, seconds=1.0)
    wd = WorkDir(tmp_path / "wd")
    import_voice(src, wd)
    text.run(wd)
    audio_stats.run(wd)
    ids = [r["name"] for r in rows]
    StepFile.write(
        wd.step_path("brouhaha"),
        pd.DataFrame({"id": ids, "snr_db": 40.0, "c50_db": 50.0, "max_pause_s": 0.1,
                      "speech_segments": [[[0.1, 0.9]]] * n, "error": None}),
        step="brouhaha", version=1, params={},
    )
    select(wd, PROFILE)
    return wd


def split_of(out) -> dict[str, str]:
    with open(out / "metadata.csv", newline="", encoding="utf-8") as f:
        return {r["file"].split("/")[-1][:-4]: r["split"] for r in csv.DictReader(f)}


def test_pick_test_ids_exact_count_and_covers_every_category():
    ids = [f"x{i}" for i in range(300)]
    cats = {i: ("a", "b", "c")[k % 3] for k, i in enumerate(ids)}
    picked = pick_test_ids(ids, cats, 150)
    assert len(picked) == len(set(picked)) == 150
    assert {cats[i] for i in picked} == {"a", "b", "c"}
    assert [sum(cats[i] == c for i in picked) for c in "abc"] == [50, 50, 50]


def test_pick_test_ids_round_robin_in_hash_order():
    ids = [f"x{i}" for i in range(9)]
    cats = {i: ("a", "b", "c")[k % 3] for k, i in enumerate(ids)}
    picked = pick_test_ids(ids, cats, 4)
    first = {c: min((i for i in ids if cats[i] == c), key=_h) for c in "abc"}
    assert picked[:3] == [first["a"], first["b"], first["c"]]
    assert picked[3] == min((i for i in ids if cats[i] == "a" and i != first["a"]), key=_h)


def test_pick_test_ids_without_categories_is_hash_order():
    ids = [f"x{i}" for i in range(20)]
    assert pick_test_ids(ids, None, 5) == sorted(ids, key=_h)[:5]


def test_pick_test_ids_uneven_categories_exhaust_small_ones():
    ids = ["a1", "b1", "b2", "b3"]
    cats = {"a1": "a", "b1": "b", "b2": "b", "b3": "b"}
    assert len(pick_test_ids(ids, cats, 3)) == 3 and "a1" in pick_test_ids(ids, cats, 3)


def test_export_assigns_test_split_and_leaves_rest_empty(tmp_path):
    wd = make_wd(tmp_path)
    out = export(wd, PROFILE, "voice", workers=1, speaker="voice-1", provenance=PROV, test_count=6)
    splits = split_of(out)
    assert sorted(v for v in splits.values() if v) == ["test"] * 6
    assert sum(1 for v in splits.values() if v == "") == 6
    cats = pd.read_parquet(wd.clips_path).set_index("id")["category"]
    assert {cats[i] for i, v in splits.items() if v == "test"} == {"greet", "price", "date"}
    report, rows = validate_audio(out, "tts")
    assert not report.errors and len(rows) == 12


def test_export_test_split_without_category_column(tmp_path):
    wd = make_wd(tmp_path, categories=None)
    out = export(wd, PROFILE, "voice", workers=1, speaker="v", provenance=PROV, test_count=5)
    test_ids = sorted(i for i, v in split_of(out).items() if v == "test")
    assert test_ids == sorted(sorted(split_of(out), key=_h)[:5])


def test_export_test_count_larger_than_available_fails_before_writing(tmp_path):
    wd = make_wd(tmp_path)
    with pytest.raises(ExportError, match="test-count"):
        export(wd, PROFILE, "voice", workers=1, speaker="v", provenance=PROV, test_count=13)
    assert not wd.export_dir("voice").exists()


def test_voice_export_requires_provenance(tmp_path):
    wd = make_wd(tmp_path)
    with pytest.raises(ExportError, match="--provenance"):
        export(wd, PROFILE, "voice", workers=1, speaker="v")
    assert not wd.export_dir("voice").exists()


def test_provenance_is_written_to_curation_json(tmp_path):
    wd = make_wd(tmp_path)
    out = export(wd, PROFILE, "voice", workers=1, speaker="v", provenance=PROV)
    cur = json.loads((out / "curation.json").read_text(encoding="utf-8"))
    assert cur["provenance"] == PROV
    assert all(v == "" for v in split_of(out).values())


@pytest.mark.parametrize("bad", [{"source": "x", "license": "y"}, {**PROV, "extra": 1}, {**PROV, "license": ""}])
def test_invalid_provenance_is_rejected(tmp_path, bad):
    wd = make_wd(tmp_path)
    with pytest.raises(ExportError, match="provenance"):
        export(wd, PROFILE, "voice", workers=1, speaker="v", provenance=bad)
    assert not wd.export_dir("voice").exists()


def test_cli_provenance_file_and_test_count(tmp_path):
    wd = make_wd(tmp_path)
    prov = tmp_path / "prov.json"
    prov.write_text(json.dumps(PROV), encoding="utf-8")
    sel = ["select", "--work", str(wd.root), "--profile", "voice"]
    assert main(sel) == 0
    args = ["export", "--work", str(wd.root), "--profile", "voice", "--dataset", "v", "--speaker", "s",
            "--workers", "1", "--test-count", "3"]
    assert main(args) == 1  # voice source without --provenance
    assert main([*args, "--provenance", str(prov)]) == 0
    assert sum(v == "test" for v in split_of(wd.export_dir("v")).values()) == 3
    assert main([*args, "--provenance", str(tmp_path / "missing.json")]) == 1


def test_bengaliai_workdir_keeps_default_provenance(tmp_path):
    wd = make_wd(tmp_path)
    clips = pd.read_parquet(wd.clips_path)
    StepFile.write(wd.clips_path, clips, step="import", version=1, params={"source": "bengaliai"})
    out = export(wd, PROFILE, "d", workers=1, speaker="v")
    cur = json.loads((out / "curation.json").read_text(encoding="utf-8"))
    assert cur["provenance"]["commercial_ok"] is False and "Bengali.AI" in cur["provenance"]["source"]


def test_voice_profile_loads_for_cli():
    assert load_profile("voice").task == "tts"
