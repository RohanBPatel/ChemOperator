"""Registry and public specification tests for the benchmark matrix."""

from __future__ import annotations

import runpy
from pathlib import Path

from chem_operator._benchmark import (
    BenchmarkConfig,
    get_model_spec,
    get_problem_spec,
    list_models,
    list_problems,
)
from chem_operator._benchmark.runner import BenchmarkRunner


def test_python_benchmark_driver_exposes_editable_selection() -> None:
    benchmark = runpy.run_path(
        Path(__file__).parents[1] / "scripts" / "benchmark.py",
        run_name="benchmark_configuration_test",
    )

    assert benchmark["CONFIG"].problem == benchmark["PROBLEM"]
    assert benchmark["CONFIG"].model == benchmark["MODEL"]
    assert benchmark["MATRIX_PROBLEMS"]
    assert benchmark["MATRIX_MODELS"]
    assert set(benchmark["MATRIX_PROBLEMS"]) <= set(list_problems())
    assert set(benchmark["MATRIX_MODELS"]) <= set(list_models())
    assert callable(benchmark["show_parameters"])


def test_default_registry_is_the_eighteen_cell_matrix() -> None:
    assert list_problems() == (
        "cstr",
        "pfr",
        "packed_bed",
        "pipe_flow",
        "pipe_flow_transient",
        "q2d",
    )
    assert list_models() == ("fno", "deeponet", "pod-deeponet")
    assert sum(
        get_model_spec(model).supports(get_problem_spec(problem))
        for problem in list_problems()
        for model in list_models()
    ) == 18
    assert all(
        channel.source in {"parameter", "constant"}
        for problem in list_problems()
        for channel in get_problem_spec(problem).input_channels
    )


def test_problem_targets_and_aliases_are_stable() -> None:
    assert get_problem_spec("q2d_cmr").name == "q2d"
    assert get_model_spec("pod_deeponet").name == "pod-deeponet"
    cstr = get_problem_spec("cstr")
    assert [channel.label for channel in cstr.output_channels] == [
        "T",
        "P",
        "X_NC7H16",
        "X_O2",
        "X_H2O",
        "X_CO",
        "X_CO2",
        "X_OH",
    ]
    selected = cstr.with_output_channels(("T", "X_O2"))
    assert [channel.label for channel in selected.output_channels] == [
        "T",
        "X_O2",
    ]


def test_benchmark_config_round_trip() -> None:
    config = BenchmarkConfig(
        problem="q2d",
        model="fno",
        data_root="custom-data",
        channel_overrides=("velocity_axial",),
        use_ray=False,
    )
    assert BenchmarkConfig.from_dict(config.to_dict()) == config


def test_cstr_full_state_override_expands_mechanism_species() -> None:
    problem, _ = BenchmarkRunner._resolved(  # pylint: disable=protected-access
        BenchmarkConfig(
            problem="cstr",
            model="deeponet",
            channel_overrides=("all",),
            use_ray=False,
        )
    )
    assert [channel.label for channel in problem.output_channels[:2]] == [
        "T",
        "P",
    ]
    assert len(problem.output_channels) > 1200
    assert any(
        channel.species == "NC7H16"
        for channel in problem.output_channels
    )
