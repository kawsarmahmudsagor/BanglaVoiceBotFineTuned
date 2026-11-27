"""Group VAD pieces into 15-25 s clips that end on complete sentences."""
from __future__ import annotations

PAD_S = 0.1
_TERMINATORS = ("।", "?", "!", ".")


def ends_sentence(text: str) -> bool:
    return (text or "").strip().endswith(_TERMINATORS)


def _span(seg: list[dict], i: int, k: int) -> float:
    return seg[k]["end_s"] - seg[i]["start_s"]


def _pause_close(seg, i, last, th) -> int:
    """Index of the piece to close at: largest gap to the next piece among
    boundaries in [clip_min_s, clip_max_s]; else among all that fit clip_max_s."""
    cands = [k for k in range(i, last)
             if _span(seg, i, k) <= th.clip_max_s]
    window = [k for k in cands if _span(seg, i, k) >= th.clip_min_s]
    pool = window or cands or [i]
    return max(pool, key=lambda k: (seg[k + 1]["start_s"] - seg[k]["end_s"]
                                    if k + 1 < len(seg) else 0.0, k))


def _group_segment(seg: list[dict], th, punctuated: bool) -> list[dict]:
    """seg: consecutive pieces of one range, none broken. Returns raw clips."""
    out: list[dict] = []

    def emit(i, k, flag=None, dropped=False):
        out.append({"range_idx": seg[i]["range_idx"],
                    "start_s": seg[i]["start_s"], "end_s": seg[k]["end_s"],
                    "piece_idxs": [p["idx"] for p in seg[i:k + 1]],
                    "flag": flag,
                    "status": "dropped" if dropped else None,
                    "drop_reason": "length" if dropped else None})

    i, j = 0, 0
    while j < len(seg):
        if _span(seg, i, j) > th.clip_max_s and j > i:
            last = j - 1
            ends = [k for k in range(i, last + 1)
                    if punctuated and ends_sentence(seg[k]["text"])
                    and _span(seg, i, k) >= th.tail_min_s]
            if ends:
                k, flag = ends[-1], None
            else:
                k = _pause_close(seg, i, j, th)
                flag = "incomplete_sentence" if punctuated else "unsplit"
            emit(i, k, flag)
            i = j = k + 1
            continue
        if (punctuated and ends_sentence(seg[j]["text"])
                and _span(seg, i, j) >= th.clip_min_s):
            emit(i, j)
            i = j = j + 1
            continue
        j += 1
    if i < len(seg):
        last = len(seg) - 1
        span = _span(seg, i, last)
        if punctuated:
            keep = span >= th.tail_min_s and ends_sentence(seg[last]["text"])
            flag = None
        else:
            keep = span >= th.clip_min_s
            flag = "unsplit"
        emit(i, last, flag if keep else None, dropped=not keep)
    return out


def group_pieces(pieces: list[dict], th, punctuated: bool) -> list[dict]:
    clips: list[dict] = []
    seg: list[dict] = []

    def flush():
        nonlocal seg
        if seg:
            clips.extend(_group_segment(seg, th, punctuated))
        seg = []

    for p in pieces:
        if p.get("broken"):
            flush()
        elif seg and seg[-1]["range_idx"] != p["range_idx"]:
            flush()
            seg.append(p)
        else:
            seg.append(p)
    flush()

    lo, hi = {}, {}
    for p in pieces:
        r = p["range_idx"]
        lo[r] = min(lo.get(r, p["start_s"]), p["start_s"])
        hi[r] = max(hi.get(r, p["end_s"]), p["end_s"])
    broken = [p for p in pieces if p.get("broken")]

    raw = [(c["start_s"], c["end_s"]) for c in clips]
    for n, c in enumerate(clips):
        r = c["range_idx"]
        s0, e0 = raw[n]
        lower = lo[r] - PAD_S
        upper = hi[r] + PAD_S
        for b in broken:
            if b["range_idx"] == r:
                if b["end_s"] <= s0 + 1e-9:
                    lower = max(lower, b["end_s"])
                if b["start_s"] >= e0 - 1e-9:
                    upper = min(upper, b["start_s"])
        if n > 0 and raw[n - 1][1] <= s0 + 1e-9:
            lower = max(lower, (raw[n - 1][1] + s0) / 2)
        if n + 1 < len(clips) and raw[n + 1][0] >= e0 - 1e-9:
            upper = min(upper, (e0 + raw[n + 1][0]) / 2)
        c["start_s"] = max(s0 - PAD_S, min(lower, s0))
        c["end_s"] = min(e0 + PAD_S, max(upper, e0))
        c["idx"] = n
    return [{k: c[k] for k in ("idx", "range_idx", "start_s", "end_s",
                               "piece_idxs", "flag", "status", "drop_reason")}
            for c in clips]
