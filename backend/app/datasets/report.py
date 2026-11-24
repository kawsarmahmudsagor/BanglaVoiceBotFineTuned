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

    def add_warning(self, issue: Issue, cap: int = 1000) -> None:
        """Store at most `cap` warnings per code; stats["warning_counts"] always counts every one."""
        counts = self.stats.setdefault("warning_counts", {})
        counts[issue.code] = counts.get(issue.code, 0) + 1
        if counts[issue.code] <= cap:
            self.warnings.append(issue)
