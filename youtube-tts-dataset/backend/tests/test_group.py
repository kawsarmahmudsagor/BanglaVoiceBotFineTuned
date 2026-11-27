from ytds.config import Thresholds
from ytds.pipeline.group import PAD_S, ends_sentence, group_pieces

TH = Thresholds()
END = "আমি যাই।"
MID = "আমি যাই"


def P(idx, s, e, text, rng=0, broken=0):
    return {"idx": idx, "range_idx": rng, "start_s": s, "end_s": e,
            "text": text, "broken": broken}


def chain(n, dur, text_at=None, gap=0.5, start=0.0, rng=0, first=0):
    """n pieces of `dur` seconds separated by `gap`; text_at maps index->text."""
    out, t = [], start
    for k in range(n):
        txt = (text_at or {}).get(k, MID)
        out.append(P(first + k, t, t + dur, txt, rng))
        t += dur + gap
    return out


def test_ends_sentence():
    for ch in "।?!.":
        assert ends_sentence(f"আমি {ch}  ")
    assert not ends_sentence("আমি")
    assert not ends_sentence("")


def test_sentence_end_at_16s_closes_there():
    ps = [P(0, 0, 8, MID), P(1, 8, 16, END), P(2, 16, 24, MID), P(3, 24, 32, END)]
    clips = group_pieces(ps, TH, True)
    assert clips[0]["piece_idxs"] == [0, 1]
    assert clips[0]["flag"] is None
    assert clips[1]["piece_idxs"][0] == 2
    assert [c["idx"] for c in clips] == list(range(len(clips)))


def test_early_sentence_end_used_when_cap_hit():
    # end at 15 s (clip 5-15 s of the span 5..), then next would pass 25 s
    ps = [P(0, 5, 9, MID), P(1, 9, 15, END), P(2, 15, 22, MID), P(3, 22, 31, MID)]
    clips = group_pieces(ps, TH, True)
    assert clips[0]["piece_idxs"] == [0, 1]
    assert clips[0]["flag"] is None
    assert clips[0]["status"] is None


def test_no_sentence_end_closes_at_largest_gap():
    ps = [P(0, 0, 9, MID), P(1, 9.2, 17, MID), P(2, 18.5, 24, MID), P(3, 24.1, 30, MID)]
    clips = group_pieces(ps, TH, True)
    assert clips[0]["piece_idxs"] == [0, 1, 2][:len(clips[0]["piece_idxs"])]
    assert clips[0]["piece_idxs"] == [0, 1]  # gap 1.5 after piece 1, span 17 in window
    assert clips[0]["flag"] == "incomplete_sentence"


def test_unpunctuated_all_unsplit_and_in_window():
    ps = chain(14, 7.0, text_at={k: END for k in range(14)})  # terminators ignored
    clips = group_pieces(ps, TH, False)
    kept = [c for c in clips if c["status"] != "dropped"]
    assert kept and all(c["flag"] == "unsplit" for c in kept)
    for c in kept:
        assert 15 <= c["end_s"] - c["start_s"] - 2 * PAD_S + 1e-6 <= 25 + 1e-6


def test_unpunctuated_short_tail_dropped():
    ps = chain(4, 7.0)  # 4*7+1.5 = 29.5 s
    clips = group_pieces(ps, TH, False)
    assert clips[-1]["status"] == "dropped" and clips[-1]["drop_reason"] == "length"


def test_tail_6s_with_terminator_kept():
    ps = [P(0, 0, 8, MID), P(1, 8, 16, END), P(2, 16.5, 22.5, END)]
    clips = group_pieces(ps, TH, True)
    assert len(clips) == 2 and clips[1]["status"] is None
    assert clips[1]["piece_idxs"] == [2]


def test_tail_4s_dropped():
    ps = [P(0, 0, 8, MID), P(1, 8, 16, END), P(2, 16.5, 20.5, END)]
    clips = group_pieces(ps, TH, True)
    assert clips[1]["status"] == "dropped" and clips[1]["drop_reason"] == "length"


def test_tail_8s_without_terminator_dropped():
    ps = [P(0, 0, 8, MID), P(1, 8, 16, END), P(2, 16.5, 24.5, MID)]
    clips = group_pieces(ps, TH, True)
    assert clips[1]["status"] == "dropped" and clips[1]["drop_reason"] == "length"


def test_broken_piece_splits_and_is_in_no_clip():
    ps = [P(0, 0, 8, MID), P(1, 8, 16, END), P(2, 16, 17, MID, broken=1),
          P(3, 17, 25, MID), P(4, 25, 33, END)]
    clips = group_pieces(ps, TH, True)
    assert all(2 not in c["piece_idxs"] for c in clips)
    assert clips[0]["piece_idxs"] == [0, 1]
    assert clips[1]["piece_idxs"] == [3, 4]
    for c in clips:
        assert c["end_s"] <= 16 or c["start_s"] >= 17


def test_broken_closes_prior_clip_by_end_rules():
    ps = [P(0, 0, 4, END), P(1, 4, 5, MID, broken=1), P(2, 5, 25, END)]
    clips = group_pieces(ps, TH, True)
    assert clips[0]["status"] == "dropped" and clips[0]["piece_idxs"] == [0]
    assert clips[1]["piece_idxs"] == [2] and clips[1]["status"] is None


def test_ranges_never_share_clip():
    ps = chain(2, 4.0, text_at={1: END}, rng=0) + chain(2, 4.0, text_at={1: END},
                                                       start=12.0, rng=1, first=2)
    clips = group_pieces(ps, TH, True)
    for c in clips:
        rs = {p["range_idx"] for p in ps if p["idx"] in c["piece_idxs"]}
        assert len(rs) == 1 and rs == {c["range_idx"]}
    assert len(clips) == 2 and [c["idx"] for c in clips] == [0, 1]


def test_padding_and_no_overlap():
    ps = [P(0, 0, 8, MID), P(1, 8, 16.05, END), P(2, 16.1, 24, MID), P(3, 24, 32, END)]
    clips = group_pieces(ps, TH, True)
    assert clips[0]["start_s"] == 0 - PAD_S
    for a, b in zip(clips, clips[1:]):
        assert a["end_s"] <= b["start_s"] + 1e-9
    assert abs(clips[0]["end_s"] - 16.075) < 1e-9  # midpoint of the 0.05 gap
    assert abs(clips[-1]["end_s"] - (32 + PAD_S)) < 1e-9


def test_padding_normal_gap():
    ps = [P(0, 1, 9, MID), P(1, 9, 17, END), P(2, 18, 26, MID), P(3, 26, 34, END)]
    clips = group_pieces(ps, TH, True)
    assert abs(clips[0]["start_s"] - 0.9) < 1e-9
    assert abs(clips[0]["end_s"] - 17.1) < 1e-9
    assert abs(clips[1]["start_s"] - 17.9) < 1e-9


def test_no_terminators_80s_gives_flagged_clips():
    ps = chain(10, 7.5, gap=0.5)  # ~80 s
    clips = group_pieces(ps, TH, True)
    kept = [c for c in clips if c["status"] != "dropped"]
    dropped = [c for c in clips if c["status"] == "dropped"]
    assert 3 <= len(kept) <= 5
    assert all(c["flag"] == "incomplete_sentence" for c in kept)
    assert len(dropped) <= 1 and all(d is clips[-1] for d in dropped)
