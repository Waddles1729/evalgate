"""Scorer registry."""

from __future__ import annotations

from ..config import ScorerConfig
from .base import Scorer, ScorerError, ScoreResult
from .deterministic import (
    ContainsScorer,
    ExactMatchScorer,
    JSONValidScorer,
    LengthScorer,
    RegexScorer,
    SimilarityScorer,
)
from .judge import AnswerRelevancyScorer, FaithfulnessScorer, JudgeScorer

_REGISTRY: dict[str, type[Scorer]] = {
    cls.type: cls
    for cls in (
        ExactMatchScorer,
        ContainsScorer,
        RegexScorer,
        JSONValidScorer,
        SimilarityScorer,
        LengthScorer,
        JudgeScorer,
        FaithfulnessScorer,
        AnswerRelevancyScorer,
    )
}


def register(scorer: type[Scorer]) -> None:
    """Register a project-specific scorer."""
    _REGISTRY[scorer.type] = scorer


def available() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def build(config: ScorerConfig) -> Scorer:
    try:
        factory = _REGISTRY[config.type]
    except KeyError as exc:
        raise ScorerError(
            f"unknown scorer type {config.type!r}; available: {', '.join(available())}"
        ) from exc
    return factory(config)


__all__ = [
    "AnswerRelevancyScorer",
    "ContainsScorer",
    "ExactMatchScorer",
    "FaithfulnessScorer",
    "JSONValidScorer",
    "JudgeScorer",
    "LengthScorer",
    "RegexScorer",
    "ScoreResult",
    "Scorer",
    "ScorerError",
    "SimilarityScorer",
    "available",
    "build",
    "register",
]
