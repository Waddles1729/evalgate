"""Baselines and diffs.

A single score tells you nothing — 0.82 is neither good nor bad. What matters
is 0.82 against last week's 0.87, and *which* cases moved. The baseline is a
committed file so that the number a pull request is measured against is itself
reviewable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .runner import RunResult

BASELINE_VERSION = 1


@dataclass(frozen=True)
class Baseline:
    version: int
    suite: str
    score: float
    case_scores: dict[str, float]
    case_passed: dict[str, bool]
    scorer_scores: dict[str, float]
    provider: str
    model: str
    created_at: str
    commit: str | None = None

    @classmethod
    def from_run(cls, run: RunResult, *, commit: str | None = None) -> Baseline:
        return cls(
            version=BASELINE_VERSION,
            suite=run.suite,
            score=run.score,
            case_scores=run.case_scores(),
            case_passed={c.id: c.passed for c in run.cases},
            scorer_scores=run.scorer_scores,
            provider=run.provider,
            model=run.model,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            commit=commit,
        )

    @classmethod
    def load(cls, path: str | Path) -> Baseline | None:
        path = Path(path)
        if not path.is_file():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: baseline is not valid JSON — {exc.msg}") from exc

        version = int(raw.get("version", 0))
        if version > BASELINE_VERSION:
            raise ValueError(
                f"{path}: baseline was written by a newer evalgate "
                f"(version {version} > {BASELINE_VERSION}); upgrade evalgate"
            )
        return cls(
            version=version,
            suite=str(raw.get("suite", "")),
            score=float(raw.get("score", 0.0)),
            case_scores={k: float(v) for k, v in (raw.get("case_scores") or {}).items()},
            case_passed={k: bool(v) for k, v in (raw.get("case_passed") or {}).items()},
            scorer_scores={
                k: float(v) for k, v in (raw.get("scorer_scores") or {}).items()
            },
            provider=str(raw.get("provider", "")),
            model=str(raw.get("model", "")),
            created_at=str(raw.get("created_at", "")),
            commit=raw.get("commit"),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "version": self.version,
            "suite": self.suite,
            "score": round(self.score, 4),
            "provider": self.provider,
            "model": self.model,
            "created_at": self.created_at,
            "commit": self.commit,
            "scorer_scores": {k: round(v, 4) for k, v in sorted(self.scorer_scores.items())},
            "case_scores": {k: round(v, 4) for k, v in sorted(self.case_scores.items())},
            "case_passed": dict(sorted(self.case_passed.items())),
        }
        # Sorted keys and a trailing newline keep the committed diff minimal.
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )


@dataclass(frozen=True)
class CaseDelta:
    id: str
    before: float
    after: float

    @property
    def delta(self) -> float:
        return self.after - self.before


@dataclass
class Diff:
    """What changed between the baseline and this run."""

    score_before: float
    score_after: float
    regressions: list[CaseDelta] = field(default_factory=list)
    improvements: list[CaseDelta] = field(default_factory=list)
    newly_failing: list[str] = field(default_factory=list)
    newly_passing: list[str] = field(default_factory=list)
    added_cases: list[str] = field(default_factory=list)
    removed_cases: list[str] = field(default_factory=list)
    scorer_deltas: dict[str, float] = field(default_factory=dict)

    @property
    def score_delta(self) -> float:
        return self.score_after - self.score_before

    @property
    def has_changes(self) -> bool:
        return bool(
            self.regressions
            or self.improvements
            or self.newly_failing
            or self.newly_passing
            or self.added_cases
            or self.removed_cases
        )


def diff_run(run: RunResult, baseline: Baseline | None, *, epsilon: float = 0.005) -> Diff:
    """Compare a run to a baseline.

    ``epsilon`` suppresses float noise so that a rerun with no changes produces
    an empty diff rather than a wall of ±0.0001 movements.
    """
    if baseline is None:
        return Diff(
            score_before=0.0,
            score_after=run.score,
            added_cases=[c.id for c in run.cases],
        )

    diff = Diff(score_before=baseline.score, score_after=run.score)
    current = run.case_scores()

    for case in run.cases:
        if case.id not in baseline.case_scores:
            diff.added_cases.append(case.id)
            continue

        before = baseline.case_scores[case.id]
        change = case.score - before
        if change < -epsilon:
            diff.regressions.append(CaseDelta(case.id, before, case.score))
        elif change > epsilon:
            diff.improvements.append(CaseDelta(case.id, before, case.score))

        was_passing = baseline.case_passed.get(case.id, before >= 0.5)
        if was_passing and not case.passed:
            diff.newly_failing.append(case.id)
        elif not was_passing and case.passed:
            diff.newly_passing.append(case.id)

    diff.removed_cases = [
        case_id for case_id in baseline.case_scores if case_id not in current
    ]

    for name, after in run.scorer_scores.items():
        before = baseline.scorer_scores.get(name)
        if before is not None and abs(after - before) > epsilon:
            diff.scorer_deltas[name] = after - before

    # Worst regressions first — that is the list someone actually reads.
    diff.regressions.sort(key=lambda d: d.delta)
    diff.improvements.sort(key=lambda d: -d.delta)
    return diff
