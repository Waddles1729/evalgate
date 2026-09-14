"""Scorers that need no model call.

These run in microseconds and cost nothing, so they should carry as much of a
suite as they can. Reach for the judge only for what genuinely needs judgement.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from ..dataset import Case
from ..providers import Provider
from .base import Scorer, ScorerError, ScoreResult

_WORD = re.compile(r"[\w']+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class ExactMatchScorer(Scorer):
    """Whole-answer equality, optionally after normalisation."""

    type = "exact_match"

    def score(self, case: Case, answer: str, *, judge: Provider | None) -> ScoreResult:
        if case.expected is None:
            raise ScorerError(f"case {case.id!r}: exact_match needs an 'expected' value")
        left, right = answer, case.expected
        if self.option("strip", True):
            left, right = left.strip(), right.strip()
        if self.option("ignore_case", True):
            left, right = left.lower(), right.lower()
        if self.option("collapse_whitespace", True):
            left = " ".join(left.split())
            right = " ".join(right.split())
        hit = left == right
        return ScoreResult(
            score=1.0 if hit else 0.0,
            reason="exact match" if hit else "answer differs from the expected value",
        )


class ContainsScorer(Scorer):
    """Require phrases to be present, and others to be absent.

    This is the scorer that encodes policy: a disclaimer that must appear, a
    competitor name that must not, a link that has to be the real one.
    """

    type = "contains"

    def validate(self) -> None:
        if not self.option("any_of") and not self.option("all_of") and not self.option("none_of"):
            raise ScorerError(
                f"scorer {self.name!r}: set at least one of any_of / all_of / none_of"
            )

    def score(self, case: Case, answer: str, *, judge: Provider | None) -> ScoreResult:
        haystack = answer if self.option("case_sensitive", False) else answer.lower()

        def normalise(values: Any) -> list[str]:
            if not values:
                return []
            if isinstance(values, str):
                values = [values]
            return [
                str(v) if self.option("case_sensitive", False) else str(v).lower()
                for v in values
            ]

        all_of = normalise(self.option("all_of"))
        any_of = normalise(self.option("any_of"))
        none_of = normalise(self.option("none_of"))

        missing = [phrase for phrase in all_of if phrase not in haystack]
        forbidden = [phrase for phrase in none_of if phrase in haystack]
        any_hit = not any_of or any(phrase in haystack for phrase in any_of)

        if forbidden:
            return ScoreResult(
                score=0.0,
                reason=f"contains forbidden text: {', '.join(sorted(forbidden))}",
                details={"forbidden": forbidden},
            )
        if not any_hit:
            return ScoreResult(
                score=0.0,
                reason=f"none of the required phrases appeared: {', '.join(any_of)}",
                details={"any_of": any_of},
            )
        if missing:
            satisfied = len(all_of) - len(missing)
            return ScoreResult(
                score=satisfied / len(all_of) if all_of else 0.0,
                reason=f"missing required text: {', '.join(sorted(missing))}",
                details={"missing": missing},
            )
        return ScoreResult(score=1.0, reason="all phrase constraints satisfied")


class RegexScorer(Scorer):
    """Match the answer against a pattern."""

    type = "regex"

    def validate(self) -> None:
        pattern = self.option("pattern")
        if not pattern:
            raise ScorerError(f"scorer {self.name!r}: 'pattern' is required")
        flags = re.MULTILINE | re.DOTALL
        if self.option("ignore_case", True):
            flags |= re.IGNORECASE
        try:
            self._regex = re.compile(str(pattern), flags)
        except re.error as exc:
            raise ScorerError(f"scorer {self.name!r}: invalid pattern — {exc}") from exc
        self._negate = bool(self.option("negate", False))

    def score(self, case: Case, answer: str, *, judge: Provider | None) -> ScoreResult:
        found = self._regex.search(answer) is not None
        ok = found != self._negate
        if ok:
            reason = "pattern did not match, as required" if self._negate else "pattern matched"
        else:
            reason = "pattern matched but should not have" if self._negate else "pattern did not match"
        return ScoreResult(score=1.0 if ok else 0.0, reason=reason)


class JSONValidScorer(Scorer):
    """Parse the answer as JSON and optionally check its shape.

    Structured output is the single most common place an LLM feature breaks
    silently after a model upgrade, and it is fully checkable without a judge.
    """

    type = "json_valid"

    def score(self, case: Case, answer: str, *, judge: Provider | None) -> ScoreResult:
        payload = answer.strip()
        if self.option("allow_fenced", True):
            payload = _strip_code_fence(payload)

        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            return ScoreResult(score=0.0, reason=f"not valid JSON — {exc.msg}")

        expected_type = self.option("expect_type")
        if expected_type:
            types = {"object": dict, "array": list, "string": str, "number": (int, float)}
            wanted = types.get(str(expected_type))
            if wanted and not isinstance(parsed, wanted):
                return ScoreResult(
                    score=0.0,
                    reason=f"expected a JSON {expected_type}, got {type(parsed).__name__}",
                )

        required = self.option("required_keys") or []
        if required:
            if not isinstance(parsed, dict):
                return ScoreResult(
                    score=0.0, reason="required_keys is set but the JSON is not an object"
                )
            missing = [key for key in required if key not in parsed]
            if missing:
                return ScoreResult(
                    score=1.0 - len(missing) / len(required),
                    reason=f"missing keys: {', '.join(missing)}",
                    details={"missing": missing},
                )

        return ScoreResult(score=1.0, reason="valid JSON matching the expected shape")


class SimilarityScorer(Scorer):
    """Semantic closeness to the reference answer.

    Uses provider embeddings when available and falls back to a token F1, so a
    suite still runs when the provider has no embedding endpoint.
    """

    type = "similarity"

    def score(self, case: Case, answer: str, *, judge: Provider | None) -> ScoreResult:
        if case.expected is None:
            raise ScorerError(f"case {case.id!r}: similarity needs an 'expected' value")

        if judge is not None and self.option("use_embeddings", True):
            try:
                left = judge.embed(answer)
                right = judge.embed(case.expected)
                score = max(0.0, min(1.0, _cosine(left, right)))
                return ScoreResult(
                    score=score,
                    reason=f"embedding cosine similarity {score:.2f}",
                    details={"method": "embedding"},
                )
            except NotImplementedError:
                pass

        score = _token_f1(answer, case.expected)
        return ScoreResult(
            score=score,
            reason=f"token F1 against the reference is {score:.2f}",
            details={"method": "token_f1"},
        )


class LengthScorer(Scorer):
    """Keep answers inside a word budget.

    Verbosity creep after a prompt change is real and nobody notices it until a
    support agent complains, so it is worth a cheap scorer.
    """

    type = "length"

    def score(self, case: Case, answer: str, *, judge: Provider | None) -> ScoreResult:
        count = len(_tokens(answer))
        minimum = int(self.option("min_words", 0))
        maximum = int(self.option("max_words", 10_000))
        if minimum > maximum:
            raise ScorerError(f"scorer {self.name!r}: min_words exceeds max_words")

        if count < minimum:
            shortfall = (minimum - count) / max(minimum, 1)
            return ScoreResult(
                score=max(0.0, 1.0 - shortfall),
                reason=f"{count} words, below the minimum of {minimum}",
            )
        if count > maximum:
            excess = (count - maximum) / max(maximum, 1)
            return ScoreResult(
                score=max(0.0, 1.0 - excess),
                reason=f"{count} words, above the maximum of {maximum}",
            )
        return ScoreResult(score=1.0, reason=f"{count} words, within budget")


def _strip_code_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) < 2:
        return text
    body = lines[1:]
    if body and body[-1].strip().startswith("```"):
        body = body[:-1]
    return "\n".join(body).strip()


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ScorerError("embeddings of different lengths cannot be compared")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    if norm_left == 0.0 or norm_right == 0.0:
        return 0.0
    return dot / (norm_left * norm_right)


def _token_f1(answer: str, reference: str) -> float:
    answer_tokens = set(_tokens(answer))
    reference_tokens = set(_tokens(reference))
    if not answer_tokens or not reference_tokens:
        return 0.0
    overlap = len(answer_tokens & reference_tokens)
    if overlap == 0:
        return 0.0
    precision = overlap / len(answer_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)
