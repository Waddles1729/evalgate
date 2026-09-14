"""Scorer abstraction.

Every scorer returns a value in 0..1 plus a short human-readable reason. The
reason is what ends up in the report next to a failing case, so it is the
difference between "this dropped" and "this dropped because the answer stopped
citing the refund window".
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

from ..config import ScorerConfig
from ..dataset import Case
from ..providers import Provider


@dataclass(frozen=True)
class ScoreResult:
    score: float
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be within 0..1, got {self.score}")


class ScorerError(ValueError):
    """Raised when a scorer is misconfigured."""


class Scorer(abc.ABC):
    """Judge one answer against one case."""

    type: str = "scorer"
    #: Scorers that call an LLM are run with the judge provider and are cached.
    uses_judge: bool = False

    def __init__(self, config: ScorerConfig) -> None:
        self.config = config
        self.name = config.name
        self.options = config.options
        self.validate()

    def validate(self) -> None:  # noqa: B027 - an optional hook, not a contract
        """Check options at construction time.

        Deliberately concrete and empty: most scorers have nothing to validate,
        and forcing every subclass to declare that would be noise. Scorers that
        do — a regex to compile, mutually exclusive options — override it so a
        bad suite file fails before any model is called.
        """

    @abc.abstractmethod
    def score(self, case: Case, answer: str, *, judge: Provider | None) -> ScoreResult:
        """Score ``answer`` for ``case``."""

    def option(self, key: str, default: Any = None) -> Any:
        return self.options.get(key, default)
