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
    store.set_thresholds(Thresholds(cer_auto=0.1))
    assert store.get_thresholds().cer_auto == 0.1


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
