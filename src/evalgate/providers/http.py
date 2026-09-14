"""OpenAI- and Anthropic-compatible providers.

Both are implemented with :mod:`urllib` rather than the vendor SDKs. A repo
whose only job is to run in someone else's CI should not drag two large
dependency trees along with it, and the request shapes here are stable.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from .base import Completion, Provider, ProviderError

_RETRYABLE = frozenset({408, 409, 429, 500, 502, 503, 504})


def _post(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    *,
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    last_error: Exception | None = None

    for attempt in range(retries + 1):
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            last_error = ProviderError(f"HTTP {exc.code} from {url}: {detail}")
            if exc.code not in _RETRYABLE or attempt == retries:
                raise last_error from exc
        except urllib.error.URLError as exc:
            last_error = ProviderError(f"could not reach {url}: {exc.reason}")
            if attempt == retries:
                raise last_error from exc
        # Exponential backoff with a conservative ceiling.
        time.sleep(min(2**attempt, 8))

    raise last_error or ProviderError("request failed")  # pragma: no cover


def _api_key(explicit: object, env_var: str, provider: str) -> str:
    key = str(explicit) if explicit else os.environ.get(env_var, "")
    if not key:
        raise ProviderError(
            f"{provider} needs an API key: set {env_var}, or use the 'fake' "
            f"provider to run the suite offline"
        )
    return key


class OpenAIProvider(Provider):
    """Works with the OpenAI API and any server exposing the same routes."""

    name = "openai"

    def complete(self, prompt: str, *, system: str | None = None) -> Completion:
        base = str(self.options.get("base_url", "https://api.openai.com/v1")).rstrip("/")
        key = _api_key(self.options.get("api_key"), "OPENAI_API_KEY", "openai")

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": float(self.options.get("temperature", 0.0)),
        }
        if "max_tokens" in self.options:
            payload["max_tokens"] = int(self.options["max_tokens"])  # type: ignore[arg-type]

        data = _post(
            f"{base}/chat/completions",
            payload,
            {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            timeout=float(self.options.get("timeout", 60.0)),
            retries=int(self.options.get("retries", 2)),
        )

        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"unexpected openai response shape: {data}") from exc

        usage = data.get("usage") or {}
        return Completion(
            text=text,
            model=str(data.get("model", self.model)),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
        )

    def embed(self, text: str) -> list[float]:
        base = str(self.options.get("base_url", "https://api.openai.com/v1")).rstrip("/")
        key = _api_key(self.options.get("api_key"), "OPENAI_API_KEY", "openai")
        model = str(self.options.get("embedding_model", "text-embedding-3-small"))

        data = _post(
            f"{base}/embeddings",
            {"model": model, "input": text},
            {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            timeout=float(self.options.get("timeout", 60.0)),
            retries=int(self.options.get("retries", 2)),
        )
        try:
            return [float(v) for v in data["data"][0]["embedding"]]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"unexpected embedding response: {data}") from exc


class AnthropicProvider(Provider):
    name = "anthropic"

    def complete(self, prompt: str, *, system: str | None = None) -> Completion:
        base = str(self.options.get("base_url", "https://api.anthropic.com/v1")).rstrip("/")
        key = _api_key(self.options.get("api_key"), "ANTHROPIC_API_KEY", "anthropic")

        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": int(self.options.get("max_tokens", 1024)),
            "temperature": float(self.options.get("temperature", 0.0)),
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system

        data = _post(
            f"{base}/messages",
            payload,
            {
                "x-api-key": key,
                "anthropic-version": str(
                    self.options.get("api_version", "2023-06-01")
                ),
                "Content-Type": "application/json",
            },
            timeout=float(self.options.get("timeout", 60.0)),
            retries=int(self.options.get("retries", 2)),
        )

        blocks = data.get("content") or []
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        usage = data.get("usage") or {}
        return Completion(
            text=text,
            model=str(data.get("model", self.model)),
            prompt_tokens=int(usage.get("input_tokens", 0)),
            completion_tokens=int(usage.get("output_tokens", 0)),
        )
