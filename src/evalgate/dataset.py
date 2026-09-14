"""Golden dataset loading.

A golden dataset is a JSONL file — one case per line — so that adding a case
shows up as a one-line diff in review, and so that a dataset of any size can be
streamed rather than held in memory twice.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _as_tuple(value: Any) -> tuple[str, ...]:
    """Accept either a bare string or a list for fields that allow both."""
    if not value:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


class DatasetError(ValueError):
    """Raised when a dataset file cannot be parsed."""


@dataclass(frozen=True)
class Case:
    """A single evaluation case.

    ``must_pass`` marks a case the product cannot ship without — a refusal that
    has to stay a refusal, a regulated disclaimer, a known customer bug. These
    are gated individually rather than being averaged away.
    """

    id: str
    input: str
    expected: str | None = None
    context: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    must_pass: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, index: int, source: Path) -> Case:
        if not isinstance(raw, dict):
            raise DatasetError(f"{source}:{index}: each line must be a JSON object")
        if "input" not in raw:
            raise DatasetError(f"{source}:{index}: case is missing 'input'")

        context = _as_tuple(raw.get("context"))
        tags = _as_tuple(raw.get("tags"))

        known = {"id", "input", "expected", "context", "tags", "must_pass"}
        return cls(
            id=str(raw.get("id") or f"case-{index}"),
            input=str(raw["input"]),
            expected=None if raw.get("expected") is None else str(raw["expected"]),
            context=context,
            tags=tags,
            must_pass=bool(raw.get("must_pass", False)),
            metadata={k: v for k, v in raw.items() if k not in known},
        )


def load_dataset(path: str | Path) -> list[Case]:
    """Read a JSONL golden dataset, rejecting duplicate ids."""
    path = Path(path)
    if not path.is_file():
        raise DatasetError(f"dataset not found: {path}")

    cases: list[Case] = []
    seen: set[str] = set()
    for index, line in _iter_json_lines(path):
        case = Case.from_dict(line, index=index, source=path)
        if case.id in seen:
            raise DatasetError(f"{path}:{index}: duplicate case id {case.id!r}")
        seen.add(case.id)
        cases.append(case)

    if not cases:
        raise DatasetError(f"{path}: dataset is empty")
    return cases


def _iter_json_lines(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for index, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("//"):
                continue
            try:
                yield index, json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path}:{index}: invalid JSON — {exc.msg}") from exc


def filter_cases(
    cases: list[Case],
    *,
    tags: tuple[str, ...] = (),
    ids: tuple[str, ...] = (),
) -> list[Case]:
    """Narrow a dataset to the cases worth re-running while iterating."""
    selected = cases
    if tags:
        wanted = set(tags)
        selected = [c for c in selected if wanted & set(c.tags)]
    if ids:
        wanted_ids = set(ids)
        selected = [c for c in selected if c.id in wanted_ids]
    return selected
