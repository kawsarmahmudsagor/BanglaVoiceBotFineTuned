"""Quality stage: DNSMOS gate and speaker-consistency filter (runs on the GPU thread)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import Thresholds
from .audio16k import SR, load_16k


def load_ecapa():
    from dataprep.steps.speakers import make_embedder

    return make_embedder()


def speaker_filter(
    embeddings: np.ndarray, dnsmos: np.ndarray, min_sim: float
) -> tuple[np.ndarray, np.ndarray]:
    n = len(embeddings)
    if n < 2:
        return np.ones(n), np.ones(n, dtype=bool)
    emb = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    best = emb[dnsmos >= np.median(dnsmos)]
    centroid = np.median(best, axis=0)
    centroid = centroid / max(np.linalg.norm(centroid), 1e-12)
    sims = emb @ centroid
    return sims, sims >= min_sim


def quality(
    vocals_wav: Path, clips: list[dict], th: Thresholds, scorer=None, embedder=None
) -> list[dict]:
    if not clips:
        return clips
    from dataprep.steps.dnsmos import score_dnsmos

    if embedder is None:
        embedder = load_ecapa()
    y = load_16k(vocals_wav)
    valid, scores, embs = [], [], []
    for c in clips:
        seg = y[int(c["start_s"] * SR):int(c["end_s"] * SR)]
        if len(seg) == 0:
            c.update(dnsmos=None, speaker_sim=None, drop_reason="length", status="dropped")
            continue
        valid.append(c)
        scores.append(score_dnsmos(seg, scorer=scorer)["dnsmos_ovrl"])
        embs.append(np.asarray(embedder(seg), dtype=float).reshape(-1))
    if not valid:
        return clips
    dn = np.array(scores)
    sims, keep = speaker_filter(np.stack(embs), dn, th.speaker_sim_min)
    for c, d, s, k in zip(valid, dn, sims, keep):
        c["dnsmos"] = float(d)
        c["speaker_sim"] = float(s)
        if d < th.dnsmos_min:
            c["drop_reason"] = "dnsmos"
        elif not k:
            c["drop_reason"] = "other_speaker"
        else:
            c["drop_reason"] = None
        if c["drop_reason"]:
            c["status"] = "dropped"
    return clips
