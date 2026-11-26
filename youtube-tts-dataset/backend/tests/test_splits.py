from ytds.splits import assign_splits


def _clips(n_videos=10, n=60, dur=5.0):
    return [{"id": f"v{v}_{i:04d}", "video_id": f"v{v}", "idx": i, "start_s": i * dur, "end_s": i * dur + dur}
            for v in range(n_videos) for i in range(n)]


def _dur(c):
    return c["end_s"] - c["start_s"]


def test_ratios_within_tolerance():
    clips = _clips()
    sp = assign_splits(clips)
    total = sum(_dur(c) for c in clips)
    for name in ("val", "test"):
        share = sum(_dur(c) for c in clips if sp[c["id"]] == name) / total
        assert 0.04 <= share <= 0.06


def test_val_test_contiguous():
    clips = _clips()
    sp = assign_splits(clips)
    for v in range(10):
        seq = [sp[c["id"]] for c in clips if c["video_id"] == f"v{v}"]
        for name in ("val", "test"):
            idx = [i for i, s in enumerate(seq) if s == name]
            assert idx and idx[-1] - idx[0] == len(idx) - 1


def test_deterministic_for_seed():
    clips = _clips()
    assert assign_splits(clips, seed=1) == assign_splits(clips, seed=1)
    assert assign_splits(clips, seed=1) != assign_splits(clips, seed=2)


def test_short_video_all_train():
    clips = _clips(n_videos=1, n=11)  # 55 s
    assert set(assign_splits(clips).values()) == {"train"}
