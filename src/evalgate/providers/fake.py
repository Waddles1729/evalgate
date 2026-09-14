"""Deterministic offline provider.

This exists so the test suite, the example, and CI all run with no API key and
no network. It is not a mock in the testing sense — it is a real provider that
happens to be rule-based, which means the demo in the README produces the
numbers the README claims, on anyone's machine, forever.

The rules are deliberately imperfect: the shipped example has the v1 prompt
losing on a couple of cases so that the baseline diff and the gate have
something real to catch.
"""

from __future__ import annotations

import hashlib
import math
import re

from .base import Completion, Provider

_WORD = re.compile(r"[a-z0-9']+")

# Canned answers keyed by a phrase that appears in the question. The second
# element is the "good" answer; the third is the degraded answer a weaker
# prompt produces.
_KNOWLEDGE: tuple[tuple[str, str, str], ...] = (
    (
        "refund",
        "Refunds are issued to the original payment method within 5 business days "
        "of approval. You can request one from Settings > Billing > Request refund.",
        "You should be able to get a refund somewhere in settings.",
    ),
    (
        "password",
        "Use the Forgot password link on the sign-in screen. The reset link is "
        "valid for 30 minutes and can only be used once.",
        "Try the forgot password link.",
    ),
    (
        "cancel",
        "You can cancel at any time from Settings > Billing > Cancel plan. Your "
        "plan stays active until the end of the current billing period.",
        "Go to billing and cancel. It stops right away.",
    ),
    (
        "invoice",
        "Invoices are available under Settings > Billing > Invoices, and are also "
        "emailed to the account owner on the first of each month.",
        "Invoices are in the billing page.",
    ),
    (
        "medical",
        "I can't advise on medical questions. Please speak to a qualified "
        "healthcare professional.",
        "I'd guess it's probably nothing serious, but you could see a doctor.",
    ),
    (
        "legal",
        "I can't give legal advice. Please consult a qualified lawyer for your "
        "situation.",
        "Generally you would win that case, but consult a lawyer.",
    ),
)

_STRICT_MARKERS = ("cite", "only use", "do not", "refuse", "must", "step-by-step")


class FakeProvider(Provider):
    """Rule-based provider whose quality depends on the prompt it is given."""

    name = "fake"

    def complete(self, prompt: str, *, system: str | None = None) -> Completion:
        instructions = (system or "").lower()
        # A prompt that actually constrains the model gets the good answers.
        strict = sum(marker in instructions for marker in _STRICT_MARKERS) >= 2
        # Match on the question alone. Retrieved context routinely mentions
        # neighbouring topics, and keying off the whole prompt would answer the
        # wrong question whenever it did.
        question = (_section(prompt, "QUESTION") or prompt).lower()

        text: str | None = None
        for keyword, good, weak in _KNOWLEDGE:
            if keyword in question:
                text = good if strict else weak
                break

        if text is None:
            text = (
                "I don't have that in the help centre. A support agent can pick "
                "this up for you."
                if strict
                else "I'm not sure about that one."
            )

        if strict and "json" in instructions:
            escaped = text.replace('"', '\\"')
            text = f'{{"answer": "{escaped}", "confident": true}}'

        return Completion(
            text=text,
            model=self.model,
            prompt_tokens=_count_tokens(prompt) + _count_tokens(system or ""),
            completion_tokens=_count_tokens(text),
        )

    def embed(self, text: str) -> list[float]:
        """A hashed bag-of-words embedding.

        Deterministic and dependency-free, and close enough to a real embedding
        that cosine similarity behaves sensibly on short answers.
        """
        dimensions = 64
        vector = [0.0] * dimensions
        for word in _WORD.findall(text.lower()):
            digest = hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            return vector
        return [v / norm for v in vector]


class FakeJudgeProvider(Provider):
    """Offline stand-in for an LLM judge.

    Scores on token overlap with the reference answer, then reports the result
    in the same JSON shape a real judge is asked for — so the parsing path in
    :mod:`evalgate.scorers.judge` is exercised identically offline and online.
    """

    name = "fake-judge"

    def complete(self, prompt: str, *, system: str | None = None) -> Completion:
        answer = _section(prompt, "ANSWER")
        reference = _section(prompt, "REFERENCE")
        context = _section(prompt, "CONTEXT")

        if reference:
            # Comparing against a reference answer is symmetric: an answer that
            # omits half the reference, or pads it with unrelated text, is worse.
            quality = _f1(answer, reference)
            reason = f"token overlap with the reference is {quality:.2f}"
        elif context:
            # Faithfulness is directional. The question is whether the answer's
            # claims are supported by the context — not whether the answer
            # repeats all of it. A context that covers more ground than the
            # answer uses is normal, and must not be penalised.
            quality = _coverage(answer, context)
            reason = f"{quality:.0%} of the answer is supported by the context"
        else:
            quality = 1.0 if answer.strip() else 0.0
            reason = "nothing to compare against; scored on non-emptiness"

        # Report on the same 1..5 rubric a real judge is asked for, so the
        # normalisation path is identical online and offline.
        verdict = 1 + round(quality * 4)
        text = f'{{"score": {verdict}, "reason": "{reason}"}}'
        return Completion(
            text=text,
            model=self.model,
            prompt_tokens=_count_tokens(prompt),
            completion_tokens=_count_tokens(text),
        )


def _section(prompt: str, header: str) -> str:
    """Pull one ``<HEADER>: ...`` block out of the judge prompt."""
    match = re.search(
        rf"^{header}:\s*\n(.*?)(?=\n[A-Z][A-Z ]+:|\Z)",
        prompt,
        re.MULTILINE | re.DOTALL,
    )
    return match.group(1).strip() if match else ""


#: Function words carry no factual content, so counting them inflates every
#: score and compresses the range the gate has to work with.
_STOPWORDS = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from", "has", "have", "i", "in", "is", "it", "its", "me", "my", "of", "on", "or", "our", "that", "the", "their", "them", "they", "this", "to", "was", "we", "were", "what", "when", "where", "which", "who", "will", "with", "you", "your"]
)


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS}


def _f1(answer: str, reference: str) -> float:
    """Symmetric agreement between an answer and a reference answer."""
    answer_words = _content_words(answer)
    reference_words = _content_words(reference)
    if not answer_words or not reference_words:
        return 0.0
    overlap = len(answer_words & reference_words)
    if overlap == 0:
        return 0.0
    recall = overlap / len(reference_words)
    precision = overlap / len(answer_words)
    return 2 * precision * recall / (precision + recall)


def _coverage(answer: str, context: str) -> float:
    """How much of the answer the context accounts for — the faithfulness question."""
    answer_words = _content_words(answer)
    if not answer_words:
        return 0.0
    context_words = _content_words(context)
    return len(answer_words & context_words) / len(answer_words)


def _count_tokens(text: str) -> int:
    """A rough token count; good enough for cost reporting in the demo."""
    return max(1, len(text) // 4) if text else 0
