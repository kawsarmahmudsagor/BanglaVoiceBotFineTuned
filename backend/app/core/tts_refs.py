"""Per-speaker reference-clip choice shared by the TTS converters (f5, cosyvoice). Pure Python."""
from __future__ import annotations

from collections.abc import Iterable

DEFAULT_SPEAKER = "default"


def pick_reference(rows: Iterable[dict]) -> dict[str, dict]:
    """The median-duration row of each speaker (`duration_s`), keyed by speaker.

    Rows are ordered by (duration_s, sample_id), so ties are broken by sample_id. For an even number of rows
    the lower-middle element is picked (index (n - 1) // 2). Rows without a speaker belong to "default".
    """
    by_speaker: dict[str, list[dict]] = {}
    for r in rows:
        by_speaker.setdefault(str(r.get("speaker") or DEFAULT_SPEAKER), []).append(r)

    def key(r: dict) -> tuple[float, str]:
        return float(r.get("duration_s") or 0), str(r.get("sample_id", ""))

    return {spk: sorted(rs, key=key)[(len(rs) - 1) // 2] for spk, rs in by_speaker.items()}
