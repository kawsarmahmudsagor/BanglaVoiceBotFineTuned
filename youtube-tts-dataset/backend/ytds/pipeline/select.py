"""Scoring of the two transcripts (model A and B) of a clip and choice of the one to keep (spec 4.6)."""
from __future__ import annotations

from typing import Literal

from app.core.text import bn_normalize

from ..config import Thresholds
from .punct import _PUNCT
from .routing import _chars, longest_repeat, route


def _outside(x: float, lo: float, hi: float) -> float:
    return max(0.0, lo - x, x - hi)


def score(text: str, duration_s: float, th: Thresholds, b_stats: dict | None = None) -> float:
    if route(text, duration_s, th) == "broken":
        return 0.0
    from .routing import bn_ratio
    cps = len(_chars(text)) / max(duration_s, 0.1)
    s = (1.0 - (1 - bn_ratio(text)) - 0.1 * max(0, longest_repeat(text) - 2)
         - min(0.3, 0.02 * _outside(cps, 8, 20)))
    if b_stats:
        if (b_stats.get("max_compression") or 0) > 2.4:
            s -= 0.3
        if b_stats.get("retry_failed"):
            s -= 0.2
    return min(1.0, max(0.0, s))


def pick(score_a: float, score_b: float, th: Thresholds) -> Literal["a", "b"]:
    return "a" if score_a > score_b + th.tie_margin else "b"


def _norm(text: str) -> str:
    return bn_normalize(_PUNCT.sub("", text or ""), mode="asr")


def ab_cer(text_a: str, text_b: str) -> float:
    a, b = _norm(text_a), _norm(text_b)
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    from torchmetrics.functional import char_error_rate
    return float(char_error_rate(preds=a, target=b))


def select_clip(piece_rows: list[dict], duration_s: float, th: Thresholds) -> dict:
    text_a = " ".join(t for t in ((p.get("text_a") or "").strip() for p in piece_rows) if t)
    text_b = " ".join(t for t in ((p.get("text_b") or "").strip() for p in piece_rows) if t)
    comps = [p["b_compression"] for p in piece_rows if p.get("b_compression") is not None]
    b_stats = {"max_compression": max(comps) if comps else 0.0,
               "retry_failed": any(p.get("b_retry_failed") for p in piece_rows)}
    sa = score(text_a, duration_s, th)
    sb = score(text_b, duration_s, th, b_stats)
    cer = ab_cer(text_a, text_b)
    out = {"text_a": text_a, "text_b": text_b, "score_a": sa, "score_b": sb,
           "text_pick": pick(sa, sb, th), "cer": cer, "flag": None, "status": "auto", "drop_reason": None}
    if sa == 0.0 and sb == 0.0:
        out.update(status="dropped", drop_reason="bad_stt")
    elif cer > th.disagree_cer:
        out.update(flag="disagree", status="flagged")
    return out
