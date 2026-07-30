"""Small end-to-end benchmark and opt-in Ray smoke tests."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from chem_operator._benchmark import (
    BenchmarkConfig,
    load_benchmark,
    run_benchmark,
    run_matrix,
)
from chem_operator.datasets import (
    CaseParameters,
    SimulationDatasetGenerator,
    SimulationRecord,
)


class _Simulator:
    name = "hagen_poiseuille_pipe_flow"
    parameter_space = {}

    def make_case(self, params):
        return CaseParameters()

    def run_case(self, case):  # pragma: no cover - generator is not used here
        raise NotImplementedError


def _records(count: int, split: str) -> list[SimulationRecord]:
    radius_coordinate = np.linspace(0.0, 1.0e-3, 8)
    records = []
    for index in range(count):
        radius = 0.9e-3 + index * 0.02e-3
        pressure_drop = 40.0 + index
        velocity = (
            pressure_drop
            * (radius**2 - radius_coordinate**2)
            / (4.0 * 1.0e-3)
        )
        params = {
            "radius": radius,
            "length": 1.0,
            "dynamic_viscosity": 1.0e-3,
            "pressure_drop": pressure_drop,
            "density": 1000.0,
        }
        records.append(
            SimulationRecord(
                coordinates={"r": radius_coordinate},
                fields={"velocity": velocity},
                constants={},
                metadata={
                    "params": params,
                    "split": split,
                    "wall_time": 1.0e-3,
                },
            )
        )
    return records


@pytest.fixture(name="small_data_root")
def fixture_small_data_root(tmp_path: Path) -> Path:
    directory = tmp_path / "pipe_flow"
    generator = SimulationDatasetGenerator(_Simulator(), directory)
    for split, count in (("train", 4), ("valid", 2), ("test", 2)):
        generator.save_split(split, _records(count, split))
    return tmp_path


def _config(
    tmp_path: Path,
    data_root: Path,
    *,
    use_ray: bool,
) -> BenchmarkConfig:
    return BenchmarkConfig(
        problem="pipe_flow",
        model="fno",
        data_root=data_root,
        output_root=tmp_path / "artifacts",
        run_id="smoke",
        use_ray=use_ray,
        tune_num_samples=1,
        tune_time_budget_s=60,
        tune_epochs=1,
        final_epochs=1,
        early_stopping_patience=1,
        cpus_per_trial=1,
        gpus_per_trial=0,
        inference_warmups=0,
        inference_repeats=1,
        fail_fast=True,
    )


def test_one_epoch_writes_and_reloads_all_core_artifacts(
    tmp_path,
    small_data_root,
) -> None:
    config = _config(tmp_path, small_data_root, use_ray=False)
    result = run_matrix(
        config,
        problems=("pipe_flow",),
        models=("fno",),
    )[0]
    assert result.status == "completed"
    directory = Path(result.run_directory)
    for name in (
        "resolved_config.json",
        "best_config.json",
        "model.pt",
        "preprocessing.pt",
        "metrics.json",
        "timings.json",
        "history.csv",
        "manifest.json",
        "result.json",
    ):
        assert (directory / name).is_file()
    loaded = load_benchmark(directory)
    assert loaded.problem.name == "pipe_flow"
    summary = Path(config.output_root) / "smoke" / "summary"
    assert (summary / "results.csv").is_file()
    assert (summary / "accuracy_heatmap.png").is_file()


@pytest.mark.skipif(
    os.environ.get("CHEM_OPERATOR_RUN_RAY_SMOKE") != "1",
    reason="Set CHEM_OPERATOR_RUN_RAY_SMOKE=1 for the process-level Ray smoke.",
)
def test_one_ray_trial_reports_and_checkpoints(
    tmp_path,
    small_data_root,
) -> None:
    result = run_benchmark(
        _config(tmp_path, small_data_root, use_ray=True)
    )
    assert result.status == "completed"
    assert (Path(result.run_directory) / "ray").is_dir()
