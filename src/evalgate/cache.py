"""On-disk cache for model calls.

Judge calls dominate the cost of an eval suite, and most of them are repeated
unchanged between runs — you edited one prompt, not the dataset. Caching on a
hash of the exact request turns a full re-run into a handful of calls, which is
the difference between running evals on every pull request and running them
once a month because they are too expensive.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from .providers import Completion, Provider


def request_key(provider: str, model: str, prompt: str, system: str | None) -> str:
    payload = json.dumps(
        {"provider": provider, "model": model, "prompt": prompt, "system": system},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CompletionCache:
    """A content-addressed cache of completions, one JSON file per entry."""

    def __init__(self, directory: Path | None) -> None:
        self.directory = Path(directory) if directory else None
        self.hits = 0
        self.misses = 0
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self.directory is not None

    def get(self, key: str) -> Completion | None:
        if self.directory is None:
            return None
        path = self.directory / f"{key}.json"
        if not path.is_file():
            self.misses += 1
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.hits += 1
            return Completion(
                text=raw["text"],
                model=raw["model"],
                prompt_tokens=int(raw.get("prompt_tokens", 0)),
                completion_tokens=int(raw.get("completion_tokens", 0)),
                cached=True,
            )
        except (json.JSONDecodeError, KeyError, OSError):
            # A corrupt entry is a cache miss, never a failed run.
            self.misses += 1
            return None

    def put(self, key: str, completion: Completion) -> None:
        if self.directory is None:
            return
        path = self.directory / f"{key}.json"
        payload = asdict(completion)
        payload.pop("cached", None)
        temporary = path.with_suffix(".json.tmp")
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            temporary.replace(path)
        except OSError:
            pass  # Caching is an optimisation; never let it break a run.

    def clear(self) -> int:
        if self.directory is None or not self.directory.is_dir():
            return 0
        removed = 0
        for entry in self.directory.glob("*.json"):
            try:
                entry.unlink()
                removed += 1
            except OSError:
                pass
        return removed


class CachedProvider(Provider):
    """Wraps a provider so identical requests are served from disk."""

    def __init__(self, inner: Provider, cache: CompletionCache) -> None:
        super().__init__(inner.model, **inner.options)
        self.inner = inner
        self.cache = cache
        self.name = inner.name

    def complete(self, prompt: str, *, system: str | None = None) -> Completion:
        key = request_key(self.inner.name, self.inner.model, prompt, system)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        completion = self.inner.complete(prompt, system=system)
        self.cache.put(key, completion)
        return completion

    def embed(self, text: str) -> list[float]:
        return self.inner.embed(text)
