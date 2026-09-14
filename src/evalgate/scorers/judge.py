"""LLM-as-judge scorers, including the RAG triad.

Two rules keep judge scores usable rather than decorative:

1. The judge is asked for a bounded integer on a named rubric, not a float.
   Models are far more consistent picking from five labelled options than
   emitting 0.73, and the mapping back to 0..1 stays ours.
2. Every judge call is deterministic-by-default (temperature 0) and cached on a
   hash of its full prompt, so re-running a suite after changing one case costs
   one call, not the whole dataset.
"""

from __future__ import annotations

import json
import re

from ..dataset import Case
from ..providers import Provider
from .base import Scorer, ScorerError, ScoreResult

_JUDGE_SYSTEM = (
    "You are a strict evaluator. You must reply with a single JSON object and "
    "nothing else, in the form {\"score\": <integer>, \"reason\": \"<one sentence>\"}. "
    "Do not wrap it in a code fence. Do not explain outside the JSON."
)

_DEFAULT_RUBRIC = """\
5 - Fully correct and complete; a domain expert would send this as-is.
4 - Correct, but omits a detail the reference includes.
3 - Partially correct; one clear error or a significant omission.
2 - Mostly wrong, though it touches the right topic.
1 - Wrong, or does not answer the question at all."""

_SCALE_MAX = 5


class JudgeScorer(Scorer):
    """Grade an answer against the reference on a 1..5 rubric."""

    type = "judge"
    uses_judge = True

    def score(self, case: Case, answer: str, *, judge: Provider | None) -> ScoreResult:
        if judge is None:
            raise ScorerError(f"scorer {self.name!r}: a judge provider is required")

        criteria = str(self.option("criteria", "factual correctness and completeness"))
        rubric = str(self.option("rubric", _DEFAULT_RUBRIC))

        parts = [
            f"Evaluate the answer on: {criteria}.",
            "",
            "RUBRIC:",
            rubric,
            "",
            "QUESTION:",
            case.input,
        ]
        if case.expected:
            parts += ["", "REFERENCE:", case.expected]
        if case.context:
            parts += ["", "CONTEXT:", "\n\n".join(case.context)]
        parts += ["", "ANSWER:", answer]

        completion = judge.complete("\n".join(parts), system=_JUDGE_SYSTEM)
        raw, reason = _parse_verdict(completion.text)
        score = _normalise(raw)
        return ScoreResult(
            score=score,
            reason=reason or f"judge scored {raw}",
            details={"raw_score": raw, "judge_model": completion.model},
        )


class FaithfulnessScorer(Scorer):
    """Is every claim in the answer supported by the retrieved context?

    This is the metric that catches hallucination in a RAG system, and it is
    the one worth gating on, because a confident unsupported answer is worse
    than a refusal.
    """

    type = "faithfulness"
    uses_judge = True

    def score(self, case: Case, answer: str, *, judge: Provider | None) -> ScoreResult:
        if judge is None:
            raise ScorerError(f"scorer {self.name!r}: a judge provider is required")
        if not case.context:
            raise ScorerError(
                f"case {case.id!r}: faithfulness needs 'context' — the passages the "
                f"system actually retrieved"
            )

        prompt = "\n".join(
            [
                "Decide how much of the ANSWER is supported by the CONTEXT alone.",
                "Ignore whether the answer is otherwise true; only the context counts.",
                "",
                "RUBRIC:",
                "5 - Every claim is directly supported by the context.",
                "4 - All important claims are supported; a minor aside is not.",
                "3 - About half the claims are supported.",
                "2 - Most claims are not supported by the context.",
                "1 - The answer contradicts the context or is entirely unsupported.",
                "",
                "CONTEXT:",
                "\n\n".join(case.context),
                "",
                "ANSWER:",
                answer,
            ]
        )
        completion = judge.complete(prompt, system=_JUDGE_SYSTEM)
        raw, reason = _parse_verdict(completion.text)
        return ScoreResult(
            score=_normalise(raw),
            reason=reason or f"faithfulness scored {raw}",
            details={"raw_score": raw},
        )


class AnswerRelevancyScorer(Scorer):
    """Does the answer address the question that was actually asked?"""

    type = "answer_relevancy"
    uses_judge = True

    def score(self, case: Case, answer: str, *, judge: Provider | None) -> ScoreResult:
        if judge is None:
            raise ScorerError(f"scorer {self.name!r}: a judge provider is required")

        prompt = "\n".join(
            [
                "Decide how directly the ANSWER addresses the QUESTION.",
                "A correct answer to a different question scores low.",
                "",
                "RUBRIC:",
                "5 - Answers exactly what was asked, with no padding.",
                "4 - Answers the question, with some irrelevant material.",
                "3 - Partially addresses the question.",
                "2 - Mostly talks about something else.",
                "1 - Does not address the question.",
                "",
                "QUESTION:",
                case.input,
                "",
                "ANSWER:",
                answer,
            ]
        )
        completion = judge.complete(prompt, system=_JUDGE_SYSTEM)
        raw, reason = _parse_verdict(completion.text)
        return ScoreResult(
            score=_normalise(raw),
            reason=reason or f"relevancy scored {raw}",
            details={"raw_score": raw},
        )


def _normalise(raw: float) -> float:
    """Map a 1..5 rubric score onto 0..1."""
    clamped = max(1.0, min(float(_SCALE_MAX), raw))
    return (clamped - 1.0) / (_SCALE_MAX - 1.0)


def _parse_verdict(text: str) -> tuple[float, str]:
    """Pull a score and reason out of a judge reply.

    Judges wander — a code fence here, a sentence before the JSON there. Rather
    than failing the run over formatting, fall back progressively and only give
    up when there is no number at all.
    """
    payload = text.strip()
    if payload.startswith("```"):
        lines = payload.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        payload = "\n".join(lines).strip()

    try:
        parsed = json.loads(payload)
        if isinstance(parsed, dict) and "score" in parsed:
            return float(parsed["score"]), str(parsed.get("reason", "")).strip()
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # A JSON object embedded in prose.
    brace = re.search(r"\{.*\}", payload, re.DOTALL)
    if brace:
        try:
            parsed = json.loads(brace.group(0))
            if isinstance(parsed, dict) and "score" in parsed:
                return float(parsed["score"]), str(parsed.get("reason", "")).strip()
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    # Last resort: the first number in the reply.
    number = re.search(r"-?\d+(?:\.\d+)?", payload)
    if number:
        return float(number.group(0)), "recovered a score from an unstructured reply"

    raise ScorerError(f"judge returned no parsable score: {text[:200]!r}")
