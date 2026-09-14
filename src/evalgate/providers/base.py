"""Provider abstraction.

Every provider is a function from a prompt to text plus token usage. Keeping
the surface this small is what lets the same suite run against OpenAI,
Anthropic, a local model, or the offline fake used in tests and CI.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached: bool = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ProviderError(RuntimeError):
    """Raised when a provider cannot produce a completion."""


class Provider(abc.ABC):
    """Minimal text-in/text-out interface."""

    name: str = "provider"

    def __init__(self, model: str, **options: object) -> None:
        self.model = model
        self.options = options

    @abc.abstractmethod
    def complete(self, prompt: str, *, system: str | None = None) -> Completion:
        """Return a completion for ``prompt``."""

    def embed(self, text: str) -> list[float]:
        """Return an embedding for ``text``.

        Providers that cannot embed raise, and the similarity scorer falls back
        to a lexical measure rather than failing the run.
        """
        raise NotImplementedError(f"{self.name} does not support embeddings")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} model={self.model!r}>"
