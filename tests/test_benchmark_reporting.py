"""Tests for benchmark CSV and chart generation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd

from chem_operator._benchmark.metrics import MetricSet, TimingRecord
from chem_operator._benchmark.reporting import (
    build_report,
    collect_result_rows,
)


@dataclass
class _Result:
    run_id: str
    problem: str
    model: str
    status: str
    metrics: MetricSet | None
    timings: TimingRecord | None
    parameter_count: int | None = None
    error: str | None = None


def _metric(value: float) -> MetricSet:
    return MetricSet(
        valid_normalized_rmse_macro=value,
        per_field={
            "temperature": {
                "rmse": value * 10.0,
                "relative_l2": value,
                "max_abs": value * 20.0,
                "normalized_rmse": value,
            }
        },
    )


def _timing(value: float) -> TimingRecord:
    return TimingRecord(
        tune_wall_seconds=value * 100.0,
        final_training_seconds=value * 10.0,
        inference_latency_median_seconds=value / 100.0,
        inference_latency_p95_seconds=value / 80.0,
    )


def test_build_report_includes_success_failed_and_missing_cells(
    tmp_path: Path,
) -> None:
    results = [
        _Result(
            "run-1",
            "cstr",
            "fno",
            "success",
            _metric(0.1),
            _timing(0.1),
            parameter_count=100,
        ),
        _Result(
            "run-1",
            "pfr",
            "deeponet",
            "failed",
            None,
            None,
            error="out of memory",
        ),
    ]

    artifacts = build_report(
        results,
        tmp_path / "summary",
        expected_problems=("cstr", "pfr"),
        expected_models=("fno", "deeponet"),
    )

    frame = pd.read_csv(artifacts["results_csv"])
    assert len(frame) == 4
    assert set(frame["status"]) == {"success", "failed", "missing"}
    successful = frame[
        (frame["problem"] == "cstr") & (frame["model"] == "fno")
    ].iloc[0]
    assert successful["valid_normalized_rmse_macro"] == 0.1
    assert successful["field_temperature_rmse"] == 1.0

    for name in (
        "accuracy_heatmap",
        "tune_time_heatmap",
        "training_time_heatmap",
        "inference_latency_heatmap",
        "accuracy_vs_latency",
    ):
        assert artifacts[name].is_file()
        assert artifacts[name].stat().st_size > 0
    assert len(artifacts["per_problem"]) == 2
    assert all(path.is_file() for path in artifacts["per_problem"])


def test_collect_result_rows_from_run_directory(tmp_path: Path) -> None:
    run_directory = tmp_path / "run-1"
    success_directory = run_directory / "cstr" / "fno"
    success_directory.mkdir(parents=True)
    (success_directory / "resolved_config.json").write_text(
        json.dumps({"problem": "cstr", "model": "fno", "run_id": "run-1"}),
        encoding="utf-8",
    )
    (success_directory / "metrics.json").write_text(
        json.dumps(_metric(0.25).to_dict()),
        encoding="utf-8",
    )
    (success_directory / "timings.json").write_text(
        json.dumps(_timing(0.25).to_dict()),
        encoding="utf-8",
    )

    failed_directory = run_directory / "pfr" / "deeponet"
    failed_directory.mkdir(parents=True)
    (failed_directory / "manifest.json").write_text(
        json.dumps(
            {
                "problem": "pfr",
                "model": "deeponet",
                "status": "failed",
                "error": "trial failed",
            }
        ),
        encoding="utf-8",
    )

    # Ray Tune uses result.json files below its own directory; they are not
    # benchmark-cell artifacts and must not become duplicate report rows.
    ray_trial = success_directory / "ray" / "trial-001"
    ray_trial.mkdir(parents=True)
    (ray_trial / "result.json").write_text(
        json.dumps({"training_iteration": 1}),
        encoding="utf-8",
    )

    rows = collect_result_rows(run_directory)

    assert len(rows) == 2
    by_cell = {(row["problem"], row["model"]): row for row in rows}
    assert by_cell[("cstr", "fno")][
        "valid_normalized_rmse_macro"
    ] == 0.25
    assert by_cell[("pfr", "deeponet")]["status"] == "failed"
