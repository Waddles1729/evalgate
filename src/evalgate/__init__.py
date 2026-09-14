"""evalgate — a regression gate for LLM applications.

Score an LLM feature against a golden dataset, compare it to a committed
baseline, and fail the build when it gets worse.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .baseline import Baseline, Diff, diff_run
from .config import SuiteConfig
from .dataset import Case, load_dataset
from .gate import Verdict, evaluate_gate
from .runner import CaseResult, RunResult, run_suite

__all__ = [
    "Baseline",
    "Case",
    "CaseResult",
    "Diff",
    "RunResult",
    "SuiteConfig",
    "Verdict",
    "__version__",
    "diff_run",
    "evaluate_gate",
    "load_dataset",
    "run_suite",
]
