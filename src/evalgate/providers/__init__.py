"""Provider registry."""

from __future__ import annotations

from ..config import ProviderConfig
from .base import Completion, Provider, ProviderError
from .fake import FakeJudgeProvider, FakeProvider
from .http import AnthropicProvider, OpenAIProvider

_REGISTRY: dict[str, type[Provider]] = {
    "fake": FakeProvider,
    "fake-judge": FakeJudgeProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
}


def register(name: str, provider: type[Provider]) -> None:
    """Register a custom provider, e.g. to point at an internal gateway."""
    _REGISTRY[name] = provider


def available() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def build(config: ProviderConfig) -> Provider:
    try:
        factory = _REGISTRY[config.name]
    except KeyError as exc:
        raise ProviderError(
            f"unknown provider {config.name!r}; available: {', '.join(available())}"
        ) from exc
    return factory(config.model, **config.options)


__all__ = [
    "AnthropicProvider",
    "Completion",
    "FakeJudgeProvider",
    "FakeProvider",
    "OpenAIProvider",
    "Provider",
    "ProviderError",
    "available",
    "build",
    "register",
]
