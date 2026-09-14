"""The merge decision.

Deliberately boring and fully deterministic: given a run, a baseline and a
config, the verdict is a pure function. That is what makes it safe to block a
pull request on — an engineer can reproduce the decision locally and argue with
it, rather than being told "the eval failed".
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .baseline import Baseline, Diff
from .config import GateConfig
from .runner import RunResult


@dataclass
class Verdict:
    passed: bool
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        if self.passed:
            return "PASS" if not self.warnings else "PASS (with warnings)"
        return "FAIL"


def evaluate_gate(
    run: RunResult,
    diff: Diff,
    baseline: Baseline | None,
    config: GateConfig,
) -> Verdict:
    """Decide whether this run may be merged."""
    verdict = Verdict(passed=True)

    if config.min_score > 0 and run.score < config.min_score:
        verdict.failures.append(
            f"overall score {run.score:.3f} is below the floor of {config.min_score:.3f}"
        )

    if config.fail_on_must_pass:
        broken = run.failed_must_pass
        if broken:
            names = ", ".join(c.id for c in broken[:5])
            more = f" (+{len(broken) - 5} more)" if len(broken) > 5 else ""
            verdict.failures.append(f"must-pass cases failed: {names}{more}")

    if run.errors:
        verdict.failures.append(
            f"{run.errors} case(s) errored; the run is not a valid measurement"
        )

    if baseline is not None:
        drop = -diff.score_delta
        if drop > config.max_regression:
            verdict.failures.append(
                f"overall score fell by {drop:.3f}, over the allowed "
                f"{config.max_regression:.3f} "
                f"({diff.score_before:.3f} → {diff.score_after:.3f})"
            )

        if config.fail_on_new_case_failure and diff.newly_failing:
            names = ", ".join(diff.newly_failing[:5])
            more = (
                f" (+{len(diff.newly_failing) - 5} more)"
                if len(diff.newly_failing) > 5
                else ""
            )
            verdict.failures.append(
                f"cases that were passing now fail: {names}{more}"
            )

        # Worth surfacing, not worth blocking on.
        if diff.regressions and not diff.newly_failing:
            verdict.warnings.append(
                f"{len(diff.regressions)} case(s) scored lower without crossing "
                f"their threshold"
            )
        if baseline.model and baseline.model != run.model:
            verdict.warnings.append(
                f"baseline was recorded on {baseline.model!r} but this run used "
                f"{run.model!r}; the comparison mixes two variables"
            )
    else:
        verdict.warnings.append(
            "no baseline found — recording one with 'evalgate baseline' makes the "
            "next run gateable"
        )

    verdict.passed = not verdict.failures
    return verdict
