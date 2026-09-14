"""Reports.

Three audiences, three formats: the terminal for the person iterating, a
Markdown summary for the pull request, and JSON for anything downstream.
"""

from __future__ import annotations

import json
from pathlib import Path

from .baseline import Baseline, Diff
from .gate import Verdict
from .runner import RunResult

_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _colour(text: str, code: str, enabled: bool) -> str:
    return f"{code}{text}{_RESET}" if enabled else text


def _arrow(delta: float) -> str:
    if delta > 0:
        return f"+{delta:.3f}"
    return f"{delta:.3f}"


def render_terminal(
    run: RunResult,
    diff: Diff,
    verdict: Verdict,
    baseline: Baseline | None,
    *,
    colour: bool = True,
    show_cases: int = 10,
) -> str:
    lines: list[str] = []
    head = f"{run.suite} — {run.provider}:{run.model}"
    lines.append(_colour(head, _BOLD, colour))

    score_line = f"score {run.score:.3f}"
    if baseline is not None:
        delta = diff.score_delta
        code = _GREEN if delta >= 0 else _RED
        score_line += f"  ({_colour(_arrow(delta), code, colour)} vs baseline {diff.score_before:.3f})"
    lines.append(score_line)

    lines.append(
        f"cases {run.passed_cases}/{len(run.cases)} passed"
        + (f", {run.errors} errored" if run.errors else "")
        + f"  ·  {run.duration_s:.1f}s"
        + (f"  ·  {run.cache_hits} cached" if run.cache_hits else "")
    )

    if run.scorer_scores:
        lines.append("")
        for name, value in sorted(run.scorer_scores.items()):
            bar = _bar(value)
            delta = diff.scorer_deltas.get(name)
            suffix = ""
            if delta is not None:
                code = _GREEN if delta >= 0 else _RED
                suffix = "  " + _colour(_arrow(delta), code, colour)
            lines.append(f"  {name:<20} {bar} {value:.3f}{suffix}")

    failing = [c for c in run.cases if not c.passed]
    if failing:
        lines.append("")
        lines.append(_colour(f"failing cases ({len(failing)})", _BOLD, colour))
        for case in failing[:show_cases]:
            marker = _colour("must-pass", _RED, colour) if case.must_pass else "failed"
            lines.append(f"  {case.id}  [{marker}]  {case.score:.3f}")
            if case.error:
                lines.append(f"    {_colour('error: ' + case.error, _RED, colour)}")
            for scorer in case.failed_scorers:
                reason = case.reasons.get(scorer, "")
                lines.append(f"    {_colour(scorer + ': ' + reason, _DIM, colour)}")
        if len(failing) > show_cases:
            lines.append(f"  … {len(failing) - show_cases} more")

    if diff.newly_failing:
        lines.append("")
        lines.append(
            _colour("regressed since baseline: ", _RED, colour)
            + ", ".join(diff.newly_failing[:10])
        )
    if diff.newly_passing:
        lines.append(
            _colour("fixed since baseline: ", _GREEN, colour)
            + ", ".join(diff.newly_passing[:10])
        )
    if diff.added_cases and baseline is not None:
        lines.append(f"new cases: {', '.join(diff.added_cases[:10])}")

    lines.append("")
    code = _GREEN if verdict.passed else _RED
    lines.append(_colour(f"gate: {verdict.summary}", code, colour))
    for failure in verdict.failures:
        lines.append(_colour(f"  ✗ {failure}", _RED, colour))
    for warning in verdict.warnings:
        lines.append(_colour(f"  ! {warning}", _YELLOW, colour))

    return "\n".join(lines)


def _bar(value: float, width: int = 20) -> str:
    filled = round(max(0.0, min(1.0, value)) * width)
    return "█" * filled + "·" * (width - filled)


def render_markdown(
    run: RunResult,
    diff: Diff,
    verdict: Verdict,
    baseline: Baseline | None,
) -> str:
    """A pull-request comment: verdict first, then only what changed."""
    icon = "✅" if verdict.passed else "❌"
    lines = [f"## {icon} evalgate — `{run.suite}`", ""]

    if baseline is not None:
        lines.append(
            f"**{run.score:.3f}** ({_arrow(diff.score_delta)} vs baseline "
            f"`{diff.score_before:.3f}`) · {run.passed_cases}/{len(run.cases)} cases "
            f"passed · `{run.provider}:{run.model}`"
        )
    else:
        lines.append(
            f"**{run.score:.3f}** · {run.passed_cases}/{len(run.cases)} cases passed "
            f"· `{run.provider}:{run.model}`"
        )
    lines.append("")

    if verdict.failures:
        lines.append("### Blocking")
        lines += [f"- {failure}" for failure in verdict.failures]
        lines.append("")
    if verdict.warnings:
        lines.append("### Warnings")
        lines += [f"- {warning}" for warning in verdict.warnings]
        lines.append("")

    if run.scorer_scores:
        lines.append("| Scorer | Score | Δ |")
        lines.append("| --- | ---: | ---: |")
        for name, value in sorted(run.scorer_scores.items()):
            delta = diff.scorer_deltas.get(name)
            lines.append(
                f"| {name} | {value:.3f} | {_arrow(delta) if delta is not None else '—'} |"
            )
        lines.append("")

    if diff.newly_failing:
        lines.append("### Newly failing")
        for case_id in diff.newly_failing[:20]:
            case = next((c for c in run.cases if c.id == case_id), None)
            if case is None:
                continue
            why = "; ".join(
                f"{name}: {case.reasons.get(name, '')}" for name in case.failed_scorers
            )
            lines.append(f"- `{case_id}` — {why}")
        lines.append("")

    if diff.regressions:
        lines.append("<details><summary>Score drops that stayed above threshold</summary>")
        lines.append("")
        for delta in diff.regressions[:20]:
            lines.append(
                f"- `{delta.id}` {delta.before:.3f} → {delta.after:.3f} "
                f"({_arrow(delta.delta)})"
            )
        lines.append("")
        lines.append("</details>")
        lines.append("")

    if diff.newly_passing:
        lines.append(f"Fixed: {', '.join(f'`{c}`' for c in diff.newly_passing[:20])}")
        lines.append("")

    lines.append(
        f"<sub>{run.duration_s:.1f}s · {run.cache_hits} cached call(s)</sub>"
    )
    return "\n".join(lines)


def write_reports(
    directory: Path,
    run: RunResult,
    diff: Diff,
    verdict: Verdict,
    baseline: Baseline | None,
) -> dict[str, Path]:
    """Write JSON and Markdown side by side; return where they landed."""
    directory.mkdir(parents=True, exist_ok=True)

    json_path = directory / "report.json"
    payload = run.to_dict()
    payload["gate"] = {
        "passed": verdict.passed,
        "failures": verdict.failures,
        "warnings": verdict.warnings,
    }
    payload["diff"] = {
        "score_before": round(diff.score_before, 4),
        "score_after": round(diff.score_after, 4),
        "score_delta": round(diff.score_delta, 4),
        "newly_failing": diff.newly_failing,
        "newly_passing": diff.newly_passing,
        "added_cases": diff.added_cases,
        "removed_cases": diff.removed_cases,
        "regressions": [
            {"id": d.id, "before": round(d.before, 4), "after": round(d.after, 4)}
            for d in diff.regressions
        ],
    }
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    markdown_path = directory / "report.md"
    markdown_path.write_text(
        render_markdown(run, diff, verdict, baseline) + "\n", encoding="utf-8"
    )

    return {"json": json_path, "markdown": markdown_path}
