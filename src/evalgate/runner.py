"""Run a suite: generate answers, score them, aggregate."""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .cache import CachedProvider, CompletionCache
from .config import SuiteConfig
from .dataset import Case
from .providers import Provider, ProviderError
from .providers import build as build_provider
from .scorers import ScorerError
from .scorers import build as build_scorer


@dataclass
class CaseResult:
    """Everything known about one case after a run."""

    id: str
    input: str
    answer: str
    score: float
    passed: bool
    must_pass: bool
    tags: tuple[str, ...]
    scores: dict[str, float] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)
    failed_scorers: tuple[str, ...] = ()
    error: str | None = None
    latency_ms: int = 0
    tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tags"] = list(self.tags)
        payload["failed_scorers"] = list(self.failed_scorers)
        return payload


@dataclass
class RunResult:
    """The outcome of a whole suite."""

    suite: str
    score: float
    cases: list[CaseResult]
    scorer_scores: dict[str, float]
    provider: str
    model: str
    duration_s: float
    tokens: int
    cache_hits: int
    errors: int

    @property
    def passed_cases(self) -> int:
        return sum(1 for c in self.cases if c.passed)

    @property
    def failed_must_pass(self) -> list[CaseResult]:
        return [c for c in self.cases if c.must_pass and not c.passed]

    def case_scores(self) -> dict[str, float]:
        return {c.id: c.score for c in self.cases}

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "score": round(self.score, 4),
            "provider": self.provider,
            "model": self.model,
            "duration_s": round(self.duration_s, 2),
            "tokens": self.tokens,
            "cache_hits": self.cache_hits,
            "errors": self.errors,
            "passed_cases": self.passed_cases,
            "total_cases": len(self.cases),
            "scorer_scores": {k: round(v, 4) for k, v in self.scorer_scores.items()},
            "cases": [c.to_dict() for c in self.cases],
        }


def run_suite(
    config: SuiteConfig,
    cases: list[Case],
    *,
    answer_fn: Callable[[Case], str] | None = None,
    on_case: Callable[[CaseResult], None] | None = None,
) -> RunResult:
    """Execute ``cases`` against ``config``.

    ``answer_fn`` lets a project plug in its own application instead of a bare
    prompt — which is the point, because what you want to gate is the system
    you ship, not a model in isolation.
    """
    started = time.monotonic()

    scorers = [build_scorer(spec) for spec in config.scorers]
    cache = CompletionCache(config.cache_dir)

    system_prompt: str | None = None
    if config.prompt is not None:
        if not config.prompt.is_file():
            raise ScorerError(f"prompt file not found: {config.prompt}")
        system_prompt = config.prompt.read_text(encoding="utf-8")

    if answer_fn is None:
        raw_provider = build_provider(config.provider)
        provider: Provider = CachedProvider(raw_provider, cache) if cache.enabled else raw_provider

        def answer_fn(case: Case) -> str:  # type: ignore[misc]
            prompt = case.input
            if case.context:
                prompt = (
                    "CONTEXT:\n" + "\n\n".join(case.context) + "\n\nQUESTION:\n" + case.input
                )
            return provider.complete(prompt, system=system_prompt).text

    needs_judge = any(s.uses_judge for s in scorers) or any(
        s.type == "similarity" for s in scorers
    )
    judge: Provider | None = None
    if needs_judge:
        raw_judge = build_provider(config.judge)
        judge = CachedProvider(raw_judge, cache) if cache.enabled else raw_judge

    def evaluate(case: Case) -> CaseResult:
        case_started = time.monotonic()
        try:
            answer = answer_fn(case)
        except ProviderError as exc:
            return CaseResult(
                id=case.id,
                input=case.input,
                answer="",
                score=0.0,
                passed=False,
                must_pass=case.must_pass,
                tags=case.tags,
                error=str(exc),
                latency_ms=int((time.monotonic() - case_started) * 1000),
            )

        scores: dict[str, float] = {}
        reasons: dict[str, str] = {}
        failed: list[str] = []
        weighted_total = 0.0
        applicable_weight = 0.0
        error: str | None = None

        for scorer, spec in zip(scorers, config.scorers, strict=True):
            if not spec.applies_to(case.tags):
                continue
            applicable_weight += spec.weight
            try:
                result = scorer.score(case, answer, judge=judge)
            except (ScorerError, ProviderError) as exc:
                # One broken scorer should not hide the rest of the signal.
                scores[scorer.name] = 0.0
                reasons[scorer.name] = f"scorer failed: {exc}"
                failed.append(scorer.name)
                error = error or str(exc)
                continue
            scores[scorer.name] = result.score
            reasons[scorer.name] = result.reason
            weighted_total += result.score * spec.weight
            if result.score < spec.threshold:
                failed.append(scorer.name)

        # Divide by the weight that actually applied to this case, so scoping a
        # scorer out does not silently penalise the cases it skipped.
        overall = weighted_total / applicable_weight if applicable_weight else 0.0
        return CaseResult(
            id=case.id,
            input=case.input,
            answer=answer,
            score=overall,
            passed=not failed,
            must_pass=case.must_pass,
            tags=case.tags,
            scores=scores,
            reasons=reasons,
            failed_scorers=tuple(failed),
            error=error,
            latency_ms=int((time.monotonic() - case_started) * 1000),
        )

    results: list[CaseResult] = []
    if config.concurrency > 1 and len(cases) > 1:
        with ThreadPoolExecutor(max_workers=config.concurrency) as pool:
            for result in pool.map(evaluate, cases):
                results.append(result)
                if on_case:
                    on_case(result)
    else:
        for case in cases:
            result = evaluate(case)
            results.append(result)
            if on_case:
                on_case(result)

    # Preserve dataset order regardless of completion order.
    order = {case.id: index for index, case in enumerate(cases)}
    results.sort(key=lambda r: order.get(r.id, 0))

    overall = sum(r.score for r in results) / len(results) if results else 0.0

    # Average each scorer only over the cases it was applied to; including
    # skipped cases as zeros would make a scoped scorer look broken.
    scorer_scores: dict[str, float] = {}
    for scorer in scorers:
        applied = [r.scores[scorer.name] for r in results if scorer.name in r.scores]
        scorer_scores[scorer.name] = sum(applied) / len(applied) if applied else 0.0

    return RunResult(
        suite=config.name,
        score=overall,
        cases=results,
        scorer_scores=scorer_scores,
        provider=config.provider.name,
        model=config.provider.model,
        duration_s=time.monotonic() - started,
        tokens=sum(r.tokens for r in results),
        cache_hits=cache.hits,
        errors=sum(1 for r in results if r.error),
    )


def resolve_answer_fn(spec: str | None, root: Path) -> Callable[[Case], str] | None:
    """Load ``module:function`` so a suite can evaluate a real application.

    The function receives the :class:`~evalgate.dataset.Case` and returns the
    answer string; whatever happens in between — retrieval, tools, a whole
    agent graph — is the thing being measured.
    """
    if not spec:
        return None
    if ":" not in spec:
        raise ScorerError(f"--answer-fn must look like 'module:function', got {spec!r}")

    import importlib
    import sys

    module_name, function_name = spec.split(":", 1)
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ScorerError(f"could not import {module_name!r}: {exc}") from exc

    try:
        function = getattr(module, function_name)
    except AttributeError as exc:
        raise ScorerError(
            f"{module_name!r} has no attribute {function_name!r}"
        ) from exc

    if not callable(function):
        raise ScorerError(f"{spec} is not callable")
    return function
