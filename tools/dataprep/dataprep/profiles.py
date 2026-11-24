"""Filter profiles (spec §6.3).

Bound semantics, used consistently by `select`:
- `Range.min` is inclusive (value >= min) and `Range.max` is inclusive (value <= max) unless
  `max_exclusive` is true. Only `clip_frac < 0.001` uses `max_exclusive: true`, as the spec table says "<".
- `Percentile(lo, hi)` takes percents (0.5 = p0.5); a row is kept when quantile(lo) <= value <= quantile(hi),
  with quantiles computed once over a fixed population (rows with a valid text and audio_stats row that are
  not empty_asr), before any later step file is considered; `select` records the absolute cut-offs in the
  report entry as `lo_value` / `hi_value`.
- A row whose value is missing (NaN) never passes a Range or Percentile filter.
"""
from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

PROFILES_DIR = Path(__file__).resolve().parent.parent / "profiles"


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Range(_Base):
    kind: Literal["range"] = "range"
    column: str
    min: float | None = None
    max: float | None = None
    max_exclusive: bool = False


class Percentile(_Base):
    kind: Literal["percentile"] = "percentile"
    column: str
    lo: float
    hi: float


class Flag(_Base):
    kind: Literal["flag"] = "flag"
    column: str
    keep: bool


Filter = Annotated[Range | Percentile | Flag, Field(discriminator="kind")]


class Profile(_Base):
    name: str
    task: Literal["asr", "tts"]
    filters: list[Filter] = []
    # Thresholds the spec marks "(flag)": reported by check-voice (Task 13), never applied by select.
    flags: list[Filter] = []
    drop_latin: bool = False
    min_speaker_minutes: float | None = None
    cer_labeller: str | None = None


def load_profile(name_or_path: str | Path) -> Profile:
    p = Path(name_or_path)
    if not (p.suffix in (".yaml", ".yml") or p.exists()):
        p = PROFILES_DIR / f"{name_or_path}.yaml"
    if not p.is_file():
        raise FileNotFoundError(f"profile not found: {name_or_path} (looked at {p})")
    return Profile.model_validate(yaml.safe_load(p.read_text(encoding="utf-8")))
