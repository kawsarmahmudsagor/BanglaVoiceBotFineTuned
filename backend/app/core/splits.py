"""Train / val / test split configuration and per-run assignment (spec section 15).

Pure: depends on pydantic and the standard library only.
"""
from __future__ import annotations

import hashlib

from pydantic import BaseModel, model_validator

SPLITS = ("train", "val", "test")

SPLIT_ALIASES: dict[str, str] = {
    "train": "train", "training": "train",
    "valid": "val", "validation": "val", "val": "val", "dev": "val",
    "test": "test", "testing": "test",
}


class EmptySplitError(ValueError):
    def __init__(self, name: str):
        super().__init__(f"split '{name}' is empty: add data or change datasets.split.ratios")
        self.split = name


class SplitRatios(BaseModel):
    train: float = 0.9
    val: float = 0.05
    test: float = 0.05

    @model_validator(mode="after")
    def _check(self) -> "SplitRatios":
        if min(self.train, self.val, self.test) < 0:
            raise ValueError("split ratios must be >= 0")
        if self.train <= 0:
            raise ValueError("train ratio must be > 0")
        if abs(self.train + self.val + self.test - 1) > 1e-6:
            raise ValueError("split ratios must sum to 1")
        return self


class SplitConfig(BaseModel):
    use_dataset_splits: bool = True
    ratios: SplitRatios = SplitRatios()


def normalize_split(value: str | None) -> str | None:
    """Map a user-supplied split name to train/val/test; empty means unassigned."""
    if value is None:
        return None
    v = value.strip().lower()
    if not v:
        return None
    try:
        return SPLIT_ALIASES[v]
    except KeyError:
        raise ValueError(f"unknown split '{value}'") from None


def for_source(cfg: SplitConfig, split_source: str | None) -> SplitConfig:
    """Ignore stored labels of versions that have no dataset-provided splits (legacy 90/5/5 hash labels)."""
    if split_source == "dataset":
        return cfg
    return cfg.model_copy(update={"use_dataset_splits": False})


def _position(sample_id: str) -> float:
    return int(hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:8], 16) / 2**32


def assign(rows: list[dict], cfg: SplitConfig, *, strict: bool = True) -> list[str]:
    """Return the split of every row. Deterministic and independent of row order.

    strict=False skips the empty-split check (used for browsing, never for runs).
    """
    labels = [(r.get("split") if cfg.use_dataset_splits else None) for r in rows]
    present = {lab for lab in labels if lab}
    targets = ["train"] + [s for s in ("val", "test") if s not in present]
    ratios = {s: getattr(cfg.ratios, s) for s in targets}
    total = sum(ratios.values())
    thresholds: list[tuple[float, str]] = []
    acc = 0.0
    for s in targets:
        acc += ratios[s] / total
        thresholds.append((acc, s))

    fallback = next(s for s in reversed(targets) if ratios[s] > 0)
    out: list[str] = []
    for r, lab in zip(rows, labels):
        if lab and lab != "train":
            out.append(lab)
            continue
        u = _position(r["sample_id"])
        out.append(next((s for t, s in thresholds if u < t), fallback))
    if strict:
        counts = {s: out.count(s) for s in SPLITS}
        for s in ("val", "test", "train"):  # report held-out splits before train
            if counts[s] == 0 and getattr(cfg.ratios, s) > 0:
                raise EmptySplitError(s)
    return out
