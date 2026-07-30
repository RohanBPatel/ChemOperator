"""Reusable Python API for reactor operator benchmarks.

The executable, user-editable benchmark configuration lives in
``scripts/benchmark.py``.  This package contains the library implementation
used by that script and by the individual reactor scripts.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from chem_operator._benchmark.artifacts import (
    ArtifactManifest,
    CheckpointBundle,
    make_run_id,
)
from chem_operator._benchmark.registry import (
    get_model_spec,
    get_problem_spec,
    list_models,
    list_problems,
)
from chem_operator._benchmark.metrics import MetricSet, TimingRecord
from chem_operator._benchmark.reporting import build_report
from chem_operator._benchmark.runner import (
    BenchmarkRunner,
    LoadedBenchmark,
    PreparedProblem,
    load_benchmark as _load_benchmark,
)
from chem_operator._benchmark.specs import (
    BenchmarkConfig,
    BenchmarkResult,
    ModelSpec,
    ProblemSpec,
)
from chem_operator._benchmark.trainers.base import ModelBackend


def run_benchmark(
    config: BenchmarkConfig | dict[str, Any],
) -> BenchmarkResult:
    """Tune, train, evaluate, and save one problem/model combination."""

    if isinstance(config, dict):
        config = BenchmarkConfig(**config)
    return BenchmarkRunner().run(config)


def run_matrix(
    base_config: BenchmarkConfig | dict[str, Any],
    *,
    problems: list[str] | tuple[str, ...] | None = None,
    models: list[str] | tuple[str, ...] | None = None,
) -> list[BenchmarkResult]:
    """Run a requested benchmark matrix and generate aggregate reports."""

    if isinstance(base_config, dict):
        base_config = BenchmarkConfig(**base_config)
    selected_problems = tuple(problems or list_problems())
    selected_models = tuple(models or list_models())
    run_id = base_config.run_id or make_run_id()
    shared_config = replace(base_config, run_id=run_id)
    runner = BenchmarkRunner()
    results = [
        runner.run(shared_config.for_cell(problem, model))
        for problem in selected_problems
        for model in selected_models
    ]
    build_report(
        results,
        output_dir=Path(shared_config.output_root) / run_id / "summary",
        expected_problems=selected_problems,
        expected_models=selected_models,
    )
    return results


def load_benchmark(checkpoint_directory: str) -> LoadedBenchmark:
    """Restore a saved model and preprocessing bundle."""

    return _load_benchmark(checkpoint_directory)


__all__ = [
    "ArtifactManifest",
    "BenchmarkConfig",
    "BenchmarkResult",
    "BenchmarkRunner",
    "CheckpointBundle",
    "LoadedBenchmark",
    "MetricSet",
    "ModelBackend",
    "ModelSpec",
    "PreparedProblem",
    "ProblemSpec",
    "TimingRecord",
    "build_report",
    "get_model_spec",
    "get_problem_spec",
    "list_models",
    "list_problems",
    "load_benchmark",
    "run_benchmark",
    "run_matrix",
]
