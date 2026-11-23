from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Issue:
    code: str
    message: str
    sample_id: str | None = None


@dataclass
class Report:
    errors: list[Issue] = field(default_factory=list)
    warnings: list[Issue] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors
