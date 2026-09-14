"""End-to-end behaviour: config → run → baseline → diff → gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evalgate.baseline import Baseline, diff_run
from evalgate.config import ConfigError, SuiteConfig
from evalgate.dataset import Case, DatasetError, filter_cases, load_dataset
from evalgate.gate import evaluate_gate
from evalgate.report import render_markdown, render_terminal, write_reports
from evalgate.runner import run_suite

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "support-bot"


def suite(tmp_path: Path, **overrides) -> SuiteConfig:
    """A small in-memory suite pointed at a temporary baseline."""
    raw = {
        "name": "test",
        "dataset": str(EXAMPLE / "dataset.jsonl"),
        "prompt": str(EXAMPLE / "prompts" / "v2.txt"),
        "baseline": str(tmp_path / "baseline.json"),
        "report_dir": str(tmp_path / "reports"),
        "cache_dir": str(tmp_path / "cache"),
        "provider": {"name": "fake", "model": "fake-1"},
        "judge": {"name": "fake-judge", "model": "fake-judge-1"},
        "scorers": [
            {"name": "closeness", "type": "similarity", "threshold": 0.4},
            {
                "name": "policy",
                "type": "contains",
                "threshold": 1.0,
                "none_of": ["I'd guess", "you would win"],
            },
        ],
        "concurrency": 1,
    }
    raw.update(overrides)
    return SuiteConfig.from_dict(raw, root=EXAMPLE)


class TestDataset:
    def test_loads_the_example(self):
        cases = load_dataset(EXAMPLE / "dataset.jsonl")
        assert len(cases) == 7
        assert {c.id for c in cases} >= {"refund-window", "medical-refusal"}
        assert sum(1 for c in cases if c.must_pass) == 3

    def test_rejects_duplicate_ids(self, tmp_path):
        path = tmp_path / "dupes.jsonl"
        path.write_text(
            '{"id": "a", "input": "x"}\n{"id": "a", "input": "y"}\n', encoding="utf-8"
        )
        with pytest.raises(DatasetError, match="duplicate case id"):
            load_dataset(path)

    def test_reports_the_offending_line_number(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text('{"id": "a", "input": "x"}\n{oops\n', encoding="utf-8")
        with pytest.raises(DatasetError, match=r":2: invalid JSON"):
            load_dataset(path)

    def test_rejects_an_empty_dataset(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("\n// only a comment\n", encoding="utf-8")
        with pytest.raises(DatasetError, match="empty"):
            load_dataset(path)

    def test_filters_by_tag_and_id(self):
        cases = load_dataset(EXAMPLE / "dataset.jsonl")
        assert {c.id for c in filter_cases(cases, tags=("refusal",))} == {
            "medical-refusal",
            "legal-refusal",
        }
        assert len(filter_cases(cases, ids=("refund-window",))) == 1


class TestConfig:
    def test_loads_the_example_suite(self):
        config = SuiteConfig.load(EXAMPLE / "evalgate.yaml")
        assert config.name == "support-bot"
        assert config.gate.min_score == 0.60
        assert {s.name for s in config.scorers} >= {"policy", "correctness"}
        # Relative paths resolve against the suite file, not the cwd.
        assert config.dataset == (EXAMPLE / "dataset.jsonl")

    def test_rejects_a_suite_with_no_scorers(self):
        with pytest.raises(ConfigError, match="no scorers"):
            SuiteConfig.from_dict({"dataset": "d.jsonl", "scorers": []}, root=EXAMPLE)

    def test_rejects_duplicate_scorer_names(self):
        with pytest.raises(ConfigError, match="duplicate scorer"):
            SuiteConfig.from_dict(
                {
                    "dataset": "d.jsonl",
                    "scorers": [
                        {"name": "a", "type": "length"},
                        {"name": "a", "type": "length"},
                    ],
                },
                root=EXAMPLE,
            )

    def test_expands_environment_variables(self, monkeypatch):
        monkeypatch.setenv("EVALGATE_TEST_MODEL", "gpt-test")
        config = SuiteConfig.from_dict(
            {
                "dataset": "dataset.jsonl",
                "provider": {"name": "fake", "model": "${EVALGATE_TEST_MODEL}"},
                "scorers": [{"type": "length"}],
            },
            root=EXAMPLE,
        )
        assert config.provider.model == "gpt-test"

    def test_environment_defaults_are_honoured(self):
        config = SuiteConfig.from_dict(
            {
                "dataset": "dataset.jsonl",
                "provider": {"name": "fake", "model": "${EVALGATE_MISSING:-fallback}"},
                "scorers": [{"type": "length"}],
            },
            root=EXAMPLE,
        )
        assert config.provider.model == "fallback"


class TestRun:
    def test_the_stricter_prompt_scores_higher(self, tmp_path):
        cases = load_dataset(EXAMPLE / "dataset.jsonl")
        weak = run_suite(
            suite(tmp_path, prompt=str(EXAMPLE / "prompts" / "v1.txt")), cases
        )
        strong = run_suite(suite(tmp_path), cases)
        assert strong.score > weak.score
        # The v1 prompt speculates on the safety cases; v2 refuses.
        assert weak.failed_must_pass
        assert not strong.failed_must_pass

    def test_a_custom_answer_fn_is_used_instead_of_the_provider(self, tmp_path):
        cases = load_dataset(EXAMPLE / "dataset.jsonl")

        def always(case: Case) -> str:
            return case.expected or ""

        run = run_suite(suite(tmp_path), cases, answer_fn=always)
        assert run.score > 0.9
        assert run.passed_cases == len(cases)

    def test_results_keep_dataset_order_under_concurrency(self, tmp_path):
        cases = load_dataset(EXAMPLE / "dataset.jsonl")
        run = run_suite(suite(tmp_path, concurrency=4), cases)
        assert [c.id for c in run.cases] == [c.id for c in cases]

    def test_a_failing_answer_fn_is_recorded_not_raised(self, tmp_path):
        from evalgate.providers import ProviderError

        def broken(case: Case) -> str:
            raise ProviderError("upstream is down")

        run = run_suite(suite(tmp_path), load_dataset(EXAMPLE / "dataset.jsonl"), answer_fn=broken)
        assert run.errors == len(run.cases)
        assert all(not c.passed for c in run.cases)

    def test_the_cache_serves_a_repeated_run(self, tmp_path):
        cases = load_dataset(EXAMPLE / "dataset.jsonl")
        config = suite(tmp_path)
        first = run_suite(config, cases)
        second = run_suite(config, cases)
        assert first.cache_hits == 0
        assert second.cache_hits > 0
        assert second.score == pytest.approx(first.score)


class TestBaselineAndGate:
    def test_round_trips_through_disk(self, tmp_path):
        cases = load_dataset(EXAMPLE / "dataset.jsonl")
        run = run_suite(suite(tmp_path), cases)
        path = tmp_path / "baseline.json"
        Baseline.from_run(run, commit="abc1234").save(path)

        loaded = Baseline.load(path)
        assert loaded is not None
        assert loaded.score == pytest.approx(run.score, abs=1e-4)
        assert loaded.commit == "abc1234"
        assert set(loaded.case_scores) == {c.id for c in cases}

    def test_an_identical_rerun_produces_an_empty_diff(self, tmp_path):
        cases = load_dataset(EXAMPLE / "dataset.jsonl")
        config = suite(tmp_path)
        run = run_suite(config, cases)
        baseline = Baseline.from_run(run)
        diff = diff_run(run_suite(config, cases), baseline)
        assert not diff.has_changes
        assert diff.score_delta == pytest.approx(0.0)

    def test_a_prompt_regression_is_caught(self, tmp_path):
        cases = load_dataset(EXAMPLE / "dataset.jsonl")
        good = run_suite(suite(tmp_path), cases)
        baseline = Baseline.from_run(good)

        bad = run_suite(
            suite(tmp_path, prompt=str(EXAMPLE / "prompts" / "v1.txt")), cases
        )
        diff = diff_run(bad, baseline)
        verdict = evaluate_gate(bad, diff, baseline, suite(tmp_path).gate)

        assert diff.score_delta < 0
        assert diff.newly_failing
        assert not verdict.passed
        assert any("now fail" in f or "fell by" in f for f in verdict.failures)

    def test_must_pass_failure_blocks_even_with_a_high_average(self, tmp_path):
        from evalgate.config import GateConfig

        cases = load_dataset(EXAMPLE / "dataset.jsonl")
        run = run_suite(
            suite(tmp_path, prompt=str(EXAMPLE / "prompts" / "v1.txt")), cases
        )
        verdict = evaluate_gate(
            run, diff_run(run, None), None, GateConfig(min_score=0.0, max_regression=1.0)
        )
        assert not verdict.passed
        assert any("must-pass" in f for f in verdict.failures)

    def test_a_first_run_passes_but_warns(self, tmp_path):
        cases = load_dataset(EXAMPLE / "dataset.jsonl")
        run = run_suite(suite(tmp_path), cases)
        verdict = evaluate_gate(run, diff_run(run, None), None, suite(tmp_path).gate)
        assert verdict.passed
        assert any("no baseline" in w for w in verdict.warnings)

    def test_a_model_change_is_flagged_as_a_confound(self, tmp_path):
        cases = load_dataset(EXAMPLE / "dataset.jsonl")
        run = run_suite(suite(tmp_path), cases)
        stale = Baseline.from_run(run)
        stale = Baseline(**{**stale.__dict__, "model": "some-other-model"})
        verdict = evaluate_gate(run, diff_run(run, stale), stale, suite(tmp_path).gate)
        assert any("mixes two variables" in w for w in verdict.warnings)

    def test_added_and_removed_cases_are_reported(self, tmp_path):
        cases = load_dataset(EXAMPLE / "dataset.jsonl")
        run = run_suite(suite(tmp_path), cases)
        baseline = Baseline.from_run(run)

        subset = run_suite(suite(tmp_path), cases[:3])
        diff = diff_run(subset, baseline)
        assert len(diff.removed_cases) == len(cases) - 3
        assert not diff.added_cases


class TestReports:
    def test_markdown_leads_with_the_verdict(self, tmp_path):
        cases = load_dataset(EXAMPLE / "dataset.jsonl")
        run = run_suite(suite(tmp_path), cases)
        diff = diff_run(run, None)
        verdict = evaluate_gate(run, diff, None, suite(tmp_path).gate)
        markdown = render_markdown(run, diff, verdict, None)
        assert markdown.startswith("## ✅") or markdown.startswith("## ❌")
        assert "cases passed" in markdown

    def test_terminal_output_is_plain_without_colour(self, tmp_path):
        cases = load_dataset(EXAMPLE / "dataset.jsonl")
        run = run_suite(suite(tmp_path), cases)
        diff = diff_run(run, None)
        verdict = evaluate_gate(run, diff, None, suite(tmp_path).gate)
        text = render_terminal(run, diff, verdict, None, colour=False)
        assert "\033[" not in text
        assert "gate:" in text

    def test_written_json_is_machine_readable(self, tmp_path):
        cases = load_dataset(EXAMPLE / "dataset.jsonl")
        config = suite(tmp_path)
        run = run_suite(config, cases)
        diff = diff_run(run, None)
        verdict = evaluate_gate(run, diff, None, config.gate)
        paths = write_reports(config.report_dir, run, diff, verdict, None)

        payload = json.loads(paths["json"].read_text(encoding="utf-8"))
        assert payload["total_cases"] == len(cases)
        assert set(payload["gate"]) == {"passed", "failures", "warnings"}
        assert len(payload["cases"]) == len(cases)
