from __future__ import annotations

import pytest

from evalgate.config import ScorerConfig
from evalgate.dataset import Case
from evalgate.providers.fake import FakeJudgeProvider, FakeProvider
from evalgate.scorers import build
from evalgate.scorers.base import ScorerError


def make(type_: str, **options) -> object:
    return build(ScorerConfig(name=type_, type=type_, options=options))


def case(**overrides) -> Case:
    defaults = {"id": "c1", "input": "How long do refunds take?"}
    defaults.update(overrides)
    return Case(**defaults)


class TestExactMatch:
    def test_normalises_case_and_whitespace_by_default(self):
        scorer = make("exact_match")
        result = scorer.score(case(expected="Five days"), "  five   DAYS ", judge=None)
        assert result.score == 1.0

    def test_respects_strict_options(self):
        scorer = make("exact_match", ignore_case=False)
        assert scorer.score(case(expected="Five"), "five", judge=None).score == 0.0

    def test_requires_an_expected_value(self):
        with pytest.raises(ScorerError, match="needs an 'expected'"):
            make("exact_match").score(case(), "anything", judge=None)


class TestContains:
    def test_forbidden_phrase_zeroes_the_score(self):
        scorer = make("contains", none_of=["I'd guess"])
        result = scorer.score(case(), "Well, I'd guess it's fine.", judge=None)
        assert result.score == 0.0
        assert "forbidden" in result.reason

    def test_partial_credit_for_missing_required_phrases(self):
        scorer = make("contains", all_of=["settings", "billing", "refund"])
        result = scorer.score(case(), "Go to Settings > Billing.", judge=None)
        assert result.score == pytest.approx(2 / 3)
        assert result.details["missing"] == ["refund"]

    def test_any_of_is_satisfied_by_one_hit(self):
        scorer = make("contains", any_of=["refund", "reimbursement"])
        assert scorer.score(case(), "A refund takes 5 days.", judge=None).score == 1.0

    def test_rejects_an_empty_configuration(self):
        with pytest.raises(ScorerError, match="at least one of"):
            make("contains")


class TestRegex:
    def test_matches(self):
        scorer = make("regex", pattern=r"\d+ business days")
        assert scorer.score(case(), "Within 5 business days.", judge=None).score == 1.0

    def test_negate_inverts_the_result(self):
        scorer = make("regex", pattern="sorry", negate=True)
        assert scorer.score(case(), "Sorry about that.", judge=None).score == 0.0
        assert scorer.score(case(), "Here you go.", judge=None).score == 1.0

    def test_invalid_pattern_fails_at_build_time(self):
        with pytest.raises(ScorerError, match="invalid pattern"):
            make("regex", pattern="(unclosed")


class TestJSONValid:
    def test_accepts_a_fenced_object(self):
        scorer = make("json_valid", required_keys=["answer"])
        answer = '```json\n{"answer": "yes"}\n```'
        assert scorer.score(case(), answer, judge=None).score == 1.0

    def test_reports_invalid_json(self):
        result = make("json_valid").score(case(), "{not json", judge=None)
        assert result.score == 0.0
        assert "not valid JSON" in result.reason

    def test_partial_credit_for_missing_keys(self):
        scorer = make("json_valid", required_keys=["a", "b"])
        result = scorer.score(case(), '{"a": 1}', judge=None)
        assert result.score == pytest.approx(0.5)

    def test_type_mismatch(self):
        scorer = make("json_valid", expect_type="array")
        assert scorer.score(case(), '{"a": 1}', judge=None).score == 0.0


class TestLength:
    def test_within_budget(self):
        scorer = make("length", max_words=10)
        assert scorer.score(case(), "one two three", judge=None).score == 1.0

    def test_degrades_as_the_answer_overruns(self):
        scorer = make("length", max_words=2)
        short = scorer.score(case(), "one two three", judge=None).score
        long = scorer.score(case(), "one two three four five six", judge=None).score
        assert 0.0 <= long < short < 1.0

    def test_rejects_an_impossible_range(self):
        with pytest.raises(ScorerError, match="min_words exceeds"):
            make("length", min_words=10, max_words=2).score(case(), "hi", judge=None)


class TestSimilarity:
    def test_uses_embeddings_when_the_judge_offers_them(self):
        scorer = make("similarity")
        judge = FakeProvider("fake-1")
        result = scorer.score(
            case(expected="Refunds take five business days"),
            "Refunds take five business days",
            judge=judge,
        )
        assert result.details["method"] == "embedding"
        assert result.score > 0.99

    def test_falls_back_to_token_f1_without_embeddings(self):
        scorer = make("similarity")
        judge = FakeJudgeProvider("fake-judge-1")  # no embed()
        result = scorer.score(
            case(expected="refunds take five days"), "refunds take five days", judge=judge
        )
        assert result.details["method"] == "token_f1"
        assert result.score == pytest.approx(1.0)

    def test_unrelated_answers_score_low(self):
        scorer = make("similarity")
        result = scorer.score(
            case(expected="Refunds take five business days"),
            "The weather in Osaka is mild.",
            judge=FakeProvider("fake-1"),
        )
        assert result.score < 0.5


class TestJudge:
    def test_scores_on_the_rubric_and_normalises_to_unit_range(self):
        scorer = make("judge")
        judge = FakeJudgeProvider("fake-judge-1")
        good = scorer.score(
            case(expected="Refunds take five business days"),
            "Refunds take five business days",
            judge=judge,
        )
        bad = scorer.score(
            case(expected="Refunds take five business days"),
            "No idea, sorry.",
            judge=judge,
        )
        assert 0.0 <= bad.score < good.score <= 1.0

    def test_requires_a_judge_provider(self):
        with pytest.raises(ScorerError, match="judge provider is required"):
            make("judge").score(case(expected="x"), "y", judge=None)


class TestFaithfulness:
    def test_requires_context(self):
        with pytest.raises(ScorerError, match="needs 'context'"):
            make("faithfulness").score(
                case(), "anything", judge=FakeJudgeProvider("fake-judge-1")
            )


class TestVerdictParsing:
    """The judge output parser is the most failure-prone path in the system."""

    @pytest.mark.parametrize(
        "reply",
        [
            '{"score": 4, "reason": "good"}',
            '```json\n{"score": 4, "reason": "good"}\n```',
            'Sure! Here is my verdict: {"score": 4, "reason": "good"}',
            "I would give this a 4 out of 5.",
        ],
    )
    def test_recovers_a_score_from_messy_replies(self, reply):
        from evalgate.scorers.judge import _parse_verdict

        score, _ = _parse_verdict(reply)
        assert score == 4.0

    def test_raises_when_there_is_no_number(self):
        from evalgate.scorers.judge import _parse_verdict

        with pytest.raises(ScorerError, match="no parsable score"):
            _parse_verdict("I cannot evaluate this.")

    def test_rubric_endpoints_map_to_zero_and_one(self):
        from evalgate.scorers.judge import _normalise

        assert _normalise(1) == 0.0
        assert _normalise(5) == 1.0
        assert _normalise(3) == pytest.approx(0.5)

    def test_out_of_range_scores_are_clamped(self):
        from evalgate.scorers.judge import _normalise

        assert _normalise(0) == 0.0
        assert _normalise(99) == 1.0
