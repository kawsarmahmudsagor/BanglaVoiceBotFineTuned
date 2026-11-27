import pytest

from ytds.config import Thresholds
from ytds.store import Store


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "ytds.db")


def _clip(i, status="auto", video="v1"):
    return {"id": f"{video}_{i:04d}", "idx": i, "start_s": float(i), "end_s": i + 3.0, "status": status}


def test_add_video_dedupes(store):
    ds = store.create_dataset("d", "spk")
    assert store.add_video(ds, "v1", "http://x/v1") is True
    assert store.add_video(ds, "v1", "http://x/v1") is False
    assert len(store.list_videos(ds)) == 1
    assert store.list_videos(ds)[0]["status"] == "queued"


def test_stage_marks_roundtrip(store):
    ds = store.create_dataset("d", "spk")
    store.add_video(ds, "v1", "u")
    assert store.stage_done(ds, "v1", "download") is False
    store.mark_stage(ds, "v1", "download", True)
    assert store.stage_done(ds, "v1", "download") is True
    store.mark_stage(ds, "v1", "download", False)
    assert store.stage_done(ds, "v1", "download") is False


def test_replace_clips_is_idempotent(store):
    ds = store.create_dataset("d", "spk")
    store.add_video(ds, "v1", "u")
    clips = [_clip(0), _clip(1)]
    store.replace_clips(ds, "v1", clips)
    store.replace_clips(ds, "v1", clips)
    assert [c["id"] for c in store.clips(ds, video_id="v1")] == ["v1_0000", "v1_0001"]
    store.replace_clips(ds, "v1", [_clip(0)])
    assert len(store.clips(ds)) == 1


def test_clips_filter_by_status(store):
    ds = store.create_dataset("d", "spk")
    store.add_video(ds, "v1", "u")
    store.replace_clips(ds, "v1", [_clip(0, "auto"), _clip(1, "flagged"), _clip(2, "dropped")])
    assert len(store.clips(ds, statuses=["auto", "flagged"])) == 2
    store.update_clip(ds, "v1_0002", status="approved", text_final="x")
    assert store.clips(ds, statuses=["approved"])[0]["text_final"] == "x"


def test_update_rejects_unknown_columns(store):
    ds = store.create_dataset("d", "spk")
    store.add_video(ds, "v1", "u")
    with pytest.raises(ValueError):
        store.update_video(ds, "v1", **{"status=1; --": "x"})
    with pytest.raises(ValueError):
        store.update_clip(ds, "v1_0000", bogus=1)
    store.update_video(ds, "v1", status="done", title="T")
    assert store.list_videos(ds)[0]["title"] == "T"


def test_thresholds_default_then_saved(store):
    assert store.get_thresholds() == Thresholds()
    store.set_thresholds(Thresholds(min_bn_ratio=0.7))
    assert store.get_thresholds().min_bn_ratio == 0.7


def test_settings_and_datasets(store):
    assert store.get_setting("k") is None
    assert store.get_setting("k", "d") == "d"
    store.set_setting("k", "v")
    store.set_setting("k", "v2")
    assert store.get_setting("k") == "v2"
    ds = store.create_dataset("d", "spk")
    assert store.get_dataset(ds)["speaker"] == "spk"
    assert store.get_dataset("nope") is None
    assert [d["id"] for d in store.list_datasets()] == [ds]


def test_get_clip_speaker_and_delete_video(store):
    ds = store.create_dataset("d", "")
    store.add_video(ds, "v1", "u1")
    store.add_video(ds, "v2", "u2")
    store.mark_stage(ds, "v1", "download", True)
    store.replace_clips(ds, "v1", [_clip(0)])
    store.replace_clips(ds, "v2", [_clip(0, video="v2")])
    assert store.get_clip(ds, "v1_0000")["video_id"] == "v1"
    assert store.get_clip(ds, "nope") is None
    store.set_dataset_speaker(ds, "Rahim")
    assert store.get_dataset(ds)["speaker"] == "Rahim"
    store.delete_video(ds, "v1")
    assert [v["id"] for v in store.list_videos(ds)] == ["v2"]
    assert [c["id"] for c in store.clips(ds)] == ["v2_0000"]
    assert not store.stage_done(ds, "v1", "download")


def test_update_clip_if_applies_only_when_unchanged(store):
    ds = store.create_dataset("d", "spk")
    store.add_video(ds, "v1", "u")
    store.replace_clips(ds, "v1", [_clip(0, status="flagged")])
    expect = {"status": "flagged", "text_final": None}
    assert store.update_clip_if(ds, "v1_0000", expect, text_final="machine") is True
    assert store.get_clip(ds, "v1_0000")["text_final"] == "machine"

    store.update_clip(ds, "v1_0000", text_final="human", status="approved")
    assert store.update_clip_if(ds, "v1_0000", {"status": "flagged", "text_final": "machine"},
                                text_final="machine 2") is False
    c = store.get_clip(ds, "v1_0000")
    assert (c["status"], c["text_final"]) == ("approved", "human")
    with pytest.raises(ValueError):
        store.update_clip_if(ds, "v1_0000", {"bogus": 1}, text_final="x")


_OLD_SCHEMA = """
CREATE TABLE datasets(
  id TEXT PRIMARY KEY, name TEXT UNIQUE, speaker TEXT, created REAL);
CREATE TABLE videos(
  dataset_id TEXT, id TEXT, url TEXT, title TEXT, channel TEXT, upload_date TEXT,
  duration_s REAL, license TEXT, status TEXT DEFAULT 'queued', error TEXT, warning TEXT,
  PRIMARY KEY(dataset_id, id));
CREATE TABLE stages(
  dataset_id TEXT, video_id TEXT, stage TEXT, done INTEGER, updated REAL,
  PRIMARY KEY(dataset_id, video_id, stage));
CREATE TABLE clips(
  dataset_id TEXT, id TEXT, video_id TEXT, idx INTEGER, start_s REAL, end_s REAL,
  dnsmos REAL, speaker_sim REAL, drop_reason TEXT, text_a TEXT, text_b TEXT, cer REAL,
  text_en TEXT, text_translated TEXT, text_final TEXT, text_source TEXT, status TEXT,
  no_punct INTEGER DEFAULT 0,
  PRIMARY KEY(dataset_id, id));
CREATE TABLE settings(key TEXT PRIMARY KEY, value TEXT);
"""


def _old_db(path, thresholds=None):
    import sqlite3

    db = sqlite3.connect(str(path))
    db.executescript(_OLD_SCHEMA)
    db.execute("INSERT INTO datasets VALUES ('ds','d','spk',1.0)")
    db.execute("INSERT INTO videos(dataset_id, id, url, status) VALUES ('ds','v1','u','done')")
    db.execute(
        "INSERT INTO clips(dataset_id, id, video_id, idx, start_s, end_s, text_a, status) "
        "VALUES ('ds','v1_0000','v1',0,0,3,'x','auto')"
    )
    for st in ("segment", "transcribe"):
        db.execute("INSERT INTO stages VALUES ('ds','v1',?,1,1.0)", (st,))
    if thresholds is not None:
        db.execute("INSERT INTO settings VALUES ('thresholds', ?)", (thresholds,))
    db.commit()
    db.close()


def test_old_db_gains_columns_and_keeps_rows(tmp_path):
    path = tmp_path / "old.db"
    _old_db(path)
    s = Store(path)
    v = s.list_videos("ds")[0]
    assert v["origin"] is None or v["origin"] == "dashboard"
    assert s.get_clip("ds", "v1_0000")["text_a"] == "x"
    s.update_clip("ds", "v1_0000", flag="disagree", text_pick="b", score_a=0.5, score_b=0.9, range_idx=1)
    assert s.get_clip("ds", "v1_0000")["text_pick"] == "b"
    s.update_video("ds", "v1", origin="studio", source="local", cuts="[[0,1]]", media="m", model_a="a", model_b="b")


def test_stage_marks_migrated(tmp_path):
    path = tmp_path / "old.db"
    _old_db(path)
    s = Store(path)
    for st in ("cut", "pieces", "transcribe_a", "transcribe_b", "punctuate_pieces", "group", "select"):
        assert s.stage_done("ds", "v1", st)
    assert s.stage_done("ds", "v1", "segment")


def test_stage_migration_keeps_existing_marks(tmp_path):
    path = tmp_path / "old.db"
    _old_db(path)
    s = Store(path)
    s.mark_stage("ds", "v1", "cut", False)
    s2 = Store(path)
    assert not s2.stage_done("ds", "v1", "cut")


def test_new_pipeline_auto_clips_move_to_review_on_open(tmp_path):
    s = Store(tmp_path / "db.sqlite")
    ds = s.create_dataset("d", "spk")
    s.add_video(ds, "v1", "u")
    s.replace_clips(ds, "v1", [
        {"id": f"v1_{i:04d}", "idx": i, "start_s": i, "end_s": i + 1, "status": st, "text_pick": pick}
        for i, (st, pick) in enumerate([("auto", "b"), ("flagged", "a"), ("approved", "b"), ("dropped", "b"),
                                        ("auto", None), ("flagged", None)])
    ])
    s2 = Store(tmp_path / "db.sqlite")
    assert [c["status"] for c in s2.clips(ds)] == ["needs_review", "needs_review", "approved", "dropped",
                                                   "auto", "flagged"]  # legacy clips (no pick) unchanged


def test_threshold_migration_keeps_user_values(tmp_path):
    path = tmp_path / "old.db"
    _old_db(path, '{"clip_min_s": 2.0, "clip_max_s": 15.0, "dnsmos_min": 3.4, "merge_gap_s": 0.3}')
    t = Store(path).get_thresholds()
    assert (t.clip_min_s, t.clip_max_s, t.dnsmos_min) == (15.0, 25.0, 3.4)
    path2 = tmp_path / "old2.db"
    _old_db(path2, '{"clip_min_s": 2.0, "clip_max_s": 20.0}')
    t2 = Store(path2).get_thresholds()
    assert (t2.clip_min_s, t2.clip_max_s) == (15.0, 20.0)


def test_pieces_roundtrip_and_delete(store):
    ds = store.create_dataset("d", "spk")
    store.add_video(ds, "v1", "u")
    store.replace_pieces(ds, "v1", [
        {"idx": 1, "range_idx": 0, "start_s": 4.0, "end_s": 6.0},
        {"idx": 0, "range_idx": 0, "start_s": 0.0, "end_s": 4.0},
    ])
    store.update_piece(ds, "v1", 0, text_a="ক", broken=0)
    ps = store.pieces(ds, "v1")
    assert [p["idx"] for p in ps] == [0, 1]
    assert ps[0]["text_a"] == "ক"
    with pytest.raises(ValueError):
        store.update_piece(ds, "v1", 0, nope=1)
    store.delete_video(ds, "v1")
    assert store.pieces(ds, "v1") == []


def test_add_video_studio(store):
    ds = store.create_dataset("d", "spk")
    assert store.add_video(ds, "local_abc", "file", origin="studio", source="local", status="fetching")
    v = store.list_videos(ds)[0]
    assert v["status"] == "fetching"
    assert (v["origin"], v["source"]) == ("studio", "local")


def test_threshold_migration_runs_once(tmp_path):
    path = tmp_path / "old.db"
    _old_db(path, '{"clip_min_s": 2.0, "clip_max_s": 15.0}')
    s = Store(path)
    assert s.get_thresholds().clip_max_s == 25.0
    s.set_thresholds(Thresholds(clip_min_s=2.0, clip_max_s=15.0))
    t = Store(path).get_thresholds()
    assert (t.clip_min_s, t.clip_max_s) == (2.0, 15.0)


def test_fresh_db_user_old_values_survive_restart(tmp_path):
    path = tmp_path / "new.db"
    Store(path).set_thresholds(Thresholds(clip_max_s=15.0))
    assert Store(path).get_thresholds().clip_max_s == 15.0
