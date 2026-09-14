"""Command line interface."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from . import __version__
from .baseline import Baseline, diff_run
from .cache import CompletionCache
from .config import ConfigError, SuiteConfig
from .dataset import DatasetError, filter_cases, load_dataset
from .gate import evaluate_gate
from .providers import ProviderError
from .report import render_terminal, write_reports
from .runner import resolve_answer_fn, run_suite
from .scorers import ScorerError


def _use_colour(stream: object) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return bool(getattr(stream, "isatty", lambda: False)())


def _git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


def _load(args: argparse.Namespace) -> tuple[SuiteConfig, list]:
    config = SuiteConfig.load(args.suite)
    cases = load_dataset(config.dataset)
    selected = filter_cases(
        cases,
        tags=tuple(args.tag or ()),
        ids=tuple(args.case or ()),
    )
    if not selected:
        raise DatasetError("no cases matched the given --tag/--case filters")
    return config, selected


def _run(args: argparse.Namespace) -> tuple:
    config, cases = _load(args)
    answer_fn = resolve_answer_fn(getattr(args, "answer_fn", None), config.root)
    run = run_suite(config, cases, answer_fn=answer_fn)
    baseline = Baseline.load(config.baseline)
    diff = diff_run(run, baseline)
    verdict = evaluate_gate(run, diff, baseline, config.gate)
    return config, run, diff, verdict, baseline


def cmd_run(args: argparse.Namespace) -> int:
    config, run, diff, verdict, baseline = _run(args)

    print(
        render_terminal(
            run, diff, verdict, baseline, colour=_use_colour(sys.stdout)
        )
    )

    if not args.no_report:
        paths = write_reports(config.report_dir, run, diff, verdict, baseline)
        print(f"\nreports: {paths['markdown']}  {paths['json']}")

    if args.gate and not verdict.passed:
        return 1
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    args.gate = True
    return cmd_run(args)


def cmd_baseline(args: argparse.Namespace) -> int:
    config, cases = _load(args)
    answer_fn = resolve_answer_fn(getattr(args, "answer_fn", None), config.root)
    run = run_suite(config, cases, answer_fn=answer_fn)

    existing = Baseline.load(config.baseline)
    if existing is not None and not args.force:
        print(
            f"a baseline already exists at {config.baseline} "
            f"(score {existing.score:.3f}, recorded {existing.created_at}).\n"
            f"re-run with --force to overwrite it.",
            file=sys.stderr,
        )
        return 1

    baseline = Baseline.from_run(run, commit=_git_commit(config.root))
    baseline.save(config.baseline)
    print(
        f"recorded baseline: score {run.score:.3f} across {len(run.cases)} cases "
        f"→ {config.baseline}"
    )
    print("commit this file so future runs are measured against it.")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    config = SuiteConfig.load(args.suite)
    cases = load_dataset(config.dataset)
    baseline = Baseline.load(config.baseline)

    print(f"suite     {config.name}")
    print(f"dataset   {config.dataset}  ({len(cases)} cases, "
          f"{sum(1 for c in cases if c.must_pass)} must-pass)")
    print(f"provider  {config.provider.name}:{config.provider.model}")
    print(f"judge     {config.judge.name}:{config.judge.model}")
    print(f"scorers   {', '.join(f'{s.name}({s.type}) w={s.weight}' for s in config.scorers)}")
    print(
        f"gate      min_score={config.gate.min_score} "
        f"max_regression={config.gate.max_regression} "
        f"must_pass={config.gate.fail_on_must_pass}"
    )
    if baseline is None:
        print("baseline  none recorded")
    else:
        print(
            f"baseline  score {baseline.score:.3f} on "
            f"{baseline.provider}:{baseline.model} at {baseline.created_at}"
            + (f" ({baseline.commit})" if baseline.commit else "")
        )

    tags: dict[str, int] = {}
    for case in cases:
        for tag in case.tags:
            tags[tag] = tags.get(tag, 0) + 1
    if tags:
        print("tags      " + ", ".join(f"{k}={v}" for k, v in sorted(tags.items())))
    return 0


def cmd_clear_cache(args: argparse.Namespace) -> int:
    config = SuiteConfig.load(args.suite)
    removed = CompletionCache(config.cache_dir).clear()
    print(f"removed {removed} cached completion(s) from {config.cache_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evalgate",
        description="Regression gate for LLM applications.",
    )
    parser.add_argument("--version", action="version", version=f"evalgate {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "-s", "--suite", default="evalgate.yaml", help="path to the suite file"
        )
        sub.add_argument(
            "--tag", action="append", help="only run cases with this tag (repeatable)"
        )
        sub.add_argument(
            "--case", action="append", help="only run this case id (repeatable)"
        )
        sub.add_argument(
            "--answer-fn",
            dest="answer_fn",
            help="'module:function' that produces an answer for a case, so the "
            "suite measures your application rather than a bare prompt",
        )

    run_parser = subparsers.add_parser("run", help="run the suite and print a report")
    add_common(run_parser)
    run_parser.add_argument(
        "--gate", action="store_true", help="exit non-zero if the gate fails"
    )
    run_parser.add_argument(
        "--no-report", action="store_true", help="skip writing report files"
    )
    run_parser.set_defaults(func=cmd_run)

    gate_parser = subparsers.add_parser(
        "gate", help="run the suite and fail the build on a regression"
    )
    add_common(gate_parser)
    gate_parser.add_argument("--no-report", action="store_true")
    gate_parser.set_defaults(func=cmd_gate)

    baseline_parser = subparsers.add_parser(
        "baseline", help="record the current scores as the baseline"
    )
    add_common(baseline_parser)
    baseline_parser.add_argument(
        "--force", action="store_true", help="overwrite an existing baseline"
    )
    baseline_parser.set_defaults(func=cmd_baseline)

    show_parser = subparsers.add_parser("show", help="describe the suite")
    show_parser.add_argument("-s", "--suite", default="evalgate.yaml")
    show_parser.set_defaults(func=cmd_show)

    cache_parser = subparsers.add_parser("clear-cache", help="drop cached model calls")
    cache_parser.add_argument("-s", "--suite", default="evalgate.yaml")
    cache_parser.set_defaults(func=cmd_clear_cache)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ConfigError, DatasetError, ScorerError, ProviderError, ValueError) as exc:
        print(f"evalgate: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
