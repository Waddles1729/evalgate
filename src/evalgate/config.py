"""Configuration model for an evalgate suite.

A suite is declared in a single YAML file so that the whole evaluation is
reviewable in a pull request diff, the same way a test suite is.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class ConfigError(ValueError):
    """Raised when a suite file is malformed."""


def _expand_env(value: Any) -> Any:
    """Expand ``${VAR}`` and ``${VAR:-default}`` inside strings, recursively."""
    if isinstance(value, str):

        def replace(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            resolved = os.environ.get(name)
            if resolved is None:
                if default is None:
                    raise ConfigError(
                        f"environment variable {name!r} is referenced by the suite "
                        f"but is not set (use ${{{name}:-default}} to make it optional)"
                    )
                return default
            return resolved

        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def _tags_of(entry: dict[str, Any], key: str) -> tuple[str, ...]:
    """Read a tag list that may be written as a bare string."""
    value = entry.get(key) or ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


@dataclass(frozen=True)
class ScorerConfig:
    """One scorer applied to every case in the dataset.

    ``weight`` controls the scorer's share of the overall suite score.
    ``threshold`` is the per-case pass mark: a case scoring below it is a
    failure for this scorer, which is what ``must_pass`` cases are judged on.
    """

    name: str
    type: str
    weight: float = 1.0
    threshold: float = 0.5
    #: Restrict this scorer to cases carrying one of these tags.
    only_tags: tuple[str, ...] = ()
    #: Exclude cases carrying any of these tags.
    skip_tags: tuple[str, ...] = ()
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ConfigError(f"scorer {self.name!r}: weight must be >= 0")
        if not 0.0 <= self.threshold <= 1.0:
            raise ConfigError(f"scorer {self.name!r}: threshold must be within 0..1")
        overlap = set(self.only_tags) & set(self.skip_tags)
        if overlap:
            raise ConfigError(
                f"scorer {self.name!r}: {', '.join(sorted(overlap))} appears in both "
                f"only_tags and skip_tags"
            )

    def applies_to(self, tags: tuple[str, ...]) -> bool:
        """Whether this scorer should judge a case with these tags.

        Not every metric is meaningful for every case. Faithfulness against
        retrieved context is the clearest example: a policy refusal is correct
        precisely because it ignores the context, so scoring it for groundedness
        measures the wrong thing and drags the suite average down for a reason
        nobody can act on.
        """
        case_tags = set(tags)
        included = not self.only_tags or bool(case_tags & set(self.only_tags))
        excluded = bool(self.skip_tags) and bool(case_tags & set(self.skip_tags))
        return included and not excluded


@dataclass(frozen=True)
class GateConfig:
    """Rules that decide whether a run may be merged.

    The defaults encode the behaviour most teams actually want: a small amount
    of noise is tolerated, but a case that was passing and is now failing stops
    the pipeline regardless of what the averages did.
    """

    min_score: float = 0.0
    max_regression: float = 0.02
    fail_on_must_pass: bool = True
    fail_on_new_case_failure: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_score <= 1.0:
            raise ConfigError("gate.min_score must be within 0..1")
        if self.max_regression < 0:
            raise ConfigError("gate.max_regression must be >= 0")


@dataclass(frozen=True)
class ProviderConfig:
    name: str = "fake"
    model: str = "fake-1"
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SuiteConfig:
    name: str
    dataset: Path
    prompt: Path | None
    provider: ProviderConfig
    judge: ProviderConfig
    scorers: tuple[ScorerConfig, ...]
    gate: GateConfig
    baseline: Path
    report_dir: Path
    cache_dir: Path | None
    concurrency: int
    root: Path

    @property
    def total_weight(self) -> float:
        return sum(s.weight for s in self.scorers)

    @classmethod
    def load(cls, path: str | Path) -> SuiteConfig:
        path = Path(path).resolve()
        if not path.is_file():
            raise ConfigError(f"suite file not found: {path}")
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ConfigError(f"{path}: top level of a suite file must be a mapping")
        return cls.from_dict(raw, root=path.parent)

    @classmethod
    def from_dict(cls, raw: dict[str, Any], root: Path) -> SuiteConfig:
        # Expand here rather than in load(), so a suite built programmatically
        # resolves ${VAR} the same way one read from disk does.
        raw = _expand_env(raw)
        root = Path(root).resolve()

        def resolve(value: str) -> Path:
            candidate = Path(value)
            return candidate if candidate.is_absolute() else (root / candidate)

        if "dataset" not in raw:
            raise ConfigError("suite is missing the required 'dataset' key")

        scorers_raw = raw.get("scorers") or []
        if not scorers_raw:
            raise ConfigError("suite declares no scorers; at least one is required")

        scorers: list[ScorerConfig] = []
        seen: set[str] = set()
        for entry in scorers_raw:
            if not isinstance(entry, dict):
                raise ConfigError("each scorer must be a mapping")
            if "type" not in entry:
                raise ConfigError("each scorer needs a 'type'")
            name = str(entry.get("name") or entry["type"])
            if name in seen:
                raise ConfigError(f"duplicate scorer name {name!r}")
            seen.add(name)
            known = {"name", "type", "weight", "threshold", "only_tags", "skip_tags"}
            scorers.append(
                ScorerConfig(
                    name=name,
                    type=str(entry["type"]),
                    weight=float(entry.get("weight", 1.0)),
                    threshold=float(entry.get("threshold", 0.5)),
                    only_tags=_tags_of(entry, "only_tags"),
                    skip_tags=_tags_of(entry, "skip_tags"),
                    options={k: v for k, v in entry.items() if k not in known},
                )
            )

        if sum(s.weight for s in scorers) <= 0:
            raise ConfigError("the combined weight of all scorers must be > 0")

        def provider_of(key: str, default_model: str) -> ProviderConfig:
            block = raw.get(key) or {}
            if isinstance(block, str):
                block = {"name": block}
            if not isinstance(block, dict):
                raise ConfigError(f"'{key}' must be a mapping or a provider name")
            known = {"name", "model"}
            return ProviderConfig(
                name=str(block.get("name", "fake")),
                model=str(block.get("model", default_model)),
                options={k: v for k, v in block.items() if k not in known},
            )

        gate_raw = raw.get("gate") or {}
        if not isinstance(gate_raw, dict):
            raise ConfigError("'gate' must be a mapping")

        concurrency = int(raw.get("concurrency", 4))
        if concurrency < 1:
            raise ConfigError("concurrency must be >= 1")

        cache_raw = raw.get("cache_dir", ".evalgate/cache")
        prompt_raw = raw.get("prompt")

        return cls(
            name=str(raw.get("name", "eval")),
            dataset=resolve(str(raw["dataset"])),
            prompt=resolve(str(prompt_raw)) if prompt_raw else None,
            provider=provider_of("provider", "fake-1"),
            judge=provider_of("judge", "fake-judge-1"),
            scorers=tuple(scorers),
            gate=GateConfig(
                min_score=float(gate_raw.get("min_score", 0.0)),
                max_regression=float(gate_raw.get("max_regression", 0.02)),
                fail_on_must_pass=bool(gate_raw.get("fail_on_must_pass", True)),
                fail_on_new_case_failure=bool(
                    gate_raw.get("fail_on_new_case_failure", True)
                ),
            ),
            baseline=resolve(str(raw.get("baseline", "baseline.json"))),
            report_dir=resolve(str(raw.get("report_dir", ".evalgate/reports"))),
            cache_dir=resolve(str(cache_raw)) if cache_raw else None,
            concurrency=concurrency,
            root=root,
        )
