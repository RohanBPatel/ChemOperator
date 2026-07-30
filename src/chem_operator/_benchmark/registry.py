"""Built-in reactor-problem and operator-model registries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from chem_operator._benchmark.specs import (
    BenchmarkConfig,
    ChannelSpec,
    ModelSpec,
    ProblemSpec,
)


def _parameter(
    label: str,
    key: str | None = None,
    *,
    display_name: str | None = None,
    unit: str = "-",
) -> ChannelSpec:
    return ChannelSpec(
        label,
        "parameter",
        key or label,
        display_name=display_name,
        unit=unit,
    )


def _field(
    label: str,
    key: str | None = None,
    *,
    display_name: str | None = None,
    unit: str = "-",
) -> ChannelSpec:
    return ChannelSpec(
        label,
        "field",
        key or label,
        display_name=display_name,
        unit=unit,
    )


def _species(
    field_name: str,
    species: str,
    *,
    label: str | None = None,
) -> ChannelSpec:
    return ChannelSpec(
        label or f"{field_name}_{species}",
        "species",
        field_name,
        species=species,
        display_name=f"{field_name}({species})",
    )


_PROBLEMS: dict[str, ProblemSpec] = {
    "cstr": ProblemSpec(
        name="cstr",
        display_name="Non-isothermal CSTR",
        dataset_dir="cstr",
        file_stem="cstr_non_isothermal",
        task="operator_cartesian",
        coordinate_names=("t",),
        spatial_ndim=1,
        coordinate_stride=5,
        input_channels=(
            _parameter("T0", unit="K"),
            _parameter("P0", unit="Pa"),
            _parameter("phi"),
            _parameter("reactive_fraction"),
            _parameter("reactor_volume", unit="m³"),
            _parameter("residence_time", unit="s"),
            _parameter("ambient_temperature", unit="K"),
            _parameter("wall_area", unit="m²"),
            _parameter("heat_transfer_coefficient"),
        ),
        output_channels=(
            _field("T", unit="K"),
            _field("P", unit="Pa"),
            _species("X", "NC7H16"),
            _species("X", "O2"),
            _species("X", "H2O"),
            _species("X", "CO"),
            _species("X", "CO2"),
            _species("X", "OH"),
        ),
        qoi_names=("final_temperature",),
    ),
    "pfr": ProblemSpec(
        name="pfr",
        display_name="PFR chain",
        dataset_dir="pfr",
        file_stem="pfr_chain_of_reactors",
        task="operator_cartesian",
        coordinate_names=("z",),
        spatial_ndim=1,
        coordinate_stride=4,
        input_channels=(
            _parameter("T0", unit="K"),
            _parameter("P0", unit="Pa"),
            _parameter("u0", unit="m/s"),
            _parameter("phi"),
            _parameter("ar_o2_ratio"),
            _parameter("length", unit="m"),
            _parameter("area", unit="m²"),
            _parameter("pressure_controller_K"),
        ),
        output_channels=(
            _field("T", unit="K"),
            _field("P", unit="Pa"),
            _field("velocity", unit="m/s"),
            _species("X", "H2"),
            _species("X", "O2"),
            _species("X", "H2O"),
            _species("X", "OH"),
        ),
        qoi_names=("outlet_temperature",),
    ),
    "packed_bed": ProblemSpec(
        name="packed_bed",
        display_name="Packed bed 1D",
        dataset_dir="packed_bed_1D",
        file_stem="packed_bed_1d",
        task="operator_cartesian",
        coordinate_names=("z",),
        spatial_ndim=1,
        resample_shape=(128,),
        input_channels=(
            _parameter("T0", unit="K"),
            _parameter("P0", unit="Pa"),
            _parameter("inlet_velocity", unit="m/s"),
            _parameter("wall_temperature", unit="K"),
            _parameter("inlet_nh3_mole_fraction"),
            _parameter("length", unit="m"),
            _parameter("radius", unit="m"),
            _parameter("porosity"),
            _parameter("tortuosity"),
            _parameter("particle_diameter", unit="m"),
            _parameter("specific_surface_area", unit="1/m"),
            _parameter("heat_transfer_coefficient"),
            _parameter("solve_energy"),
            _parameter("membrane_present"),
            _parameter("membrane_permeability"),
            _parameter("membrane_thickness", unit="m"),
            _parameter("sweep_pressure", unit="Pa"),
            _parameter("diluent_species_code", "diluent_species__code"),
            _parameter("membrane_species_code", "membrane_species__code"),
        ),
        output_channels=(
            _field("T", unit="K"),
            _field("P", unit="Pa"),
            _field("velocity", unit="m/s"),
            _field("rhou"),
            _species("X", "H2"),
            _species("X", "NH3"),
            _species("X", "N2"),
            _species("X", "AR"),
            _species("Z", "Ru(s)"),
            _species("Z", "N(s)"),
            _species("Z", "H(s)"),
            _species("Z", "NH(s)"),
            _species("Z", "NH2(s)"),
            _species("Z", "NH3(s)"),
        ),
        categorical_encoders={
            "diluent_species": ("AR",),
            "membrane_species": ("H2",),
        },
        qoi_names=("outlet_temperature",),
    ),
    "pipe_flow": ProblemSpec(
        name="pipe_flow",
        display_name="Steady pipe flow",
        dataset_dir="pipe_flow",
        file_stem="hagen_poiseuille_pipe_flow",
        task="operator_cartesian",
        coordinate_names=("r",),
        spatial_ndim=1,
        input_channels=(
            _parameter("radius", unit="m"),
            _parameter("length", unit="m"),
            _parameter("dynamic_viscosity", unit="Pa·s"),
            _parameter("pressure_drop", unit="Pa"),
            _parameter("density", unit="kg/m³"),
        ),
        output_channels=(_field("velocity", unit="m/s"),),
        qoi_names=("flow_rate",),
    ),
    "pipe_flow_transient": ProblemSpec(
        name="pipe_flow_transient",
        display_name="Transient pipe flow",
        dataset_dir="pipe_flow_transient",
        file_stem="transient_hagen_poiseuille_pipe_flow",
        task="operator_cartesian",
        coordinate_names=("t", "r"),
        spatial_ndim=2,
        input_channels=(
            _parameter("radius", unit="m"),
            _parameter("length", unit="m"),
            _parameter("dynamic_viscosity", unit="Pa·s"),
            _parameter("pressure_drop", unit="Pa"),
            _parameter("density", unit="kg/m³"),
        ),
        output_channels=(_field("velocity", unit="m/s"),),
        qoi_names=("flow_rate",),
    ),
    "q2d": ProblemSpec(
        name="q2d",
        display_name="Quasi-2D CMR",
        dataset_dir="q2d_cmr",
        file_stem="q2d_cmr",
        task="field_map",
        coordinate_names=("z", "r"),
        spatial_ndim=2,
        input_channels=(
            _parameter("T0", unit="K"),
            _parameter("SCCM", "sccm", unit="sccm"),
        ),
        output_channels=(
            _field("velocity_axial", unit="m/s"),
            _species("X", "CH4"),
            _species("X", "H2"),
        ),
        qoi_names=("outlet_conversion",),
    ),
}


def _fno_search_space(_: BenchmarkConfig) -> Mapping[str, Any]:
    from ray import tune

    return {
        "modes": tune.choice([8, 12, 16]),
        "hidden_channels": tune.choice([16, 32, 64]),
        "n_layers": tune.choice([2, 3, 4]),
        "learning_rate": tune.loguniform(1.0e-4, 3.0e-3),
        "weight_decay": tune.loguniform(1.0e-8, 1.0e-4),
        "batch_size": tune.choice([2, 4, 8, 16]),
    }


def _deeponet_search_space(_: BenchmarkConfig) -> Mapping[str, Any]:
    from ray import tune

    return {
        "loss": "mse",
        "width": tune.choice([128, 256, 512]),
        "latent_width": tune.choice([16, 32, 64]),
        "branch_hidden_layers": tune.choice([2, 3, 4]),
        "trunk_hidden_layers": tune.choice([2, 3, 4]),
        "activation": tune.choice(["relu", "gelu", "tanh"]),
        "learning_rate": tune.loguniform(1.0e-4, 3.0e-3),
        "weight_decay": tune.loguniform(1.0e-8, 1.0e-4),
        "batch_size": tune.choice([4, 8, 16, 32]),
    }


def _pod_search_space(_: BenchmarkConfig) -> Mapping[str, Any]:
    from ray import tune

    return {
        "loss": "mse",
        "width": tune.choice([128, 256, 512]),
        "latent_width": 1,
        "branch_hidden_layers": tune.choice([2, 3, 4]),
        "trunk_hidden_layers": 0,
        "activation": tune.choice(["relu", "gelu", "tanh"]),
        "learning_rate": tune.loguniform(1.0e-4, 3.0e-3),
        "weight_decay": tune.loguniform(1.0e-8, 1.0e-4),
        "batch_size": tune.choice([4, 8, 16, 32]),
        "variance_threshold": tune.choice([0.99, 0.999, 0.9995]),
        "max_pod_components": tune.choice([32, 64, 128]),
    }


def _make_backend(name: str):
    if name == "fno":
        from chem_operator._benchmark.trainers.fno import FNOBackend

        return FNOBackend()
    if name in {"deeponet", "pod-deeponet"}:
        from chem_operator._benchmark.trainers.deeponet import DeepONetBackend

        return DeepONetBackend(use_pod=name == "pod-deeponet")
    raise KeyError(f"Unknown benchmark backend {name!r}.")


def _make_adapter(dataset: Any, problem: ProblemSpec, *, backend: str):
    if backend == "fno":
        return dataset
    from chem_operator._benchmark.trainers.base import GridDeepONetDataset

    return GridDeepONetDataset(
        dataset,
        coordinate_names=problem.coordinate_names,
        output_labels=[
            channel.label for channel in problem.output_channels
        ],
    )


_MODELS: dict[str, ModelSpec] = {
    "fno": ModelSpec(
        name="fno",
        display_name="FNO",
        backend="fno",
        search_space_factory=_fno_search_space,
        backend_factory=_make_backend,
        adapter_factory=lambda dataset, problem: _make_adapter(
            dataset, problem, backend="fno"
        ),
        default_hyperparameters={
            "modes": 8,
            "hidden_channels": 16,
            "n_layers": 2,
            "learning_rate": 1.0e-3,
            "weight_decay": 1.0e-6,
            "batch_size": 4,
        },
    ),
    "deeponet": ModelSpec(
        name="deeponet",
        display_name="DeepONet",
        backend="deeponet",
        search_space_factory=_deeponet_search_space,
        backend_factory=_make_backend,
        adapter_factory=lambda dataset, problem: _make_adapter(
            dataset, problem, backend="deeponet"
        ),
        default_hyperparameters={
            "loss": "mse",
            "width": 128,
            "latent_width": 16,
            "branch_hidden_layers": 2,
            "trunk_hidden_layers": 2,
            "activation": "gelu",
            "learning_rate": 1.0e-3,
            "weight_decay": 1.0e-6,
            "batch_size": 8,
        },
    ),
    "pod-deeponet": ModelSpec(
        name="pod-deeponet",
        display_name="POD-DeepONet",
        backend="pod-deeponet",
        search_space_factory=_pod_search_space,
        backend_factory=_make_backend,
        adapter_factory=lambda dataset, problem: _make_adapter(
            dataset, problem, backend="pod-deeponet"
        ),
        default_hyperparameters={
            "loss": "mse",
            "width": 128,
            "latent_width": 1,
            "branch_hidden_layers": 2,
            "trunk_hidden_layers": 0,
            "activation": "gelu",
            "learning_rate": 1.0e-3,
            "weight_decay": 1.0e-6,
            "batch_size": 8,
            "variance_threshold": 0.999,
            "max_pod_components": 64,
        },
    ),
}

_PROBLEM_ALIASES = {
    "non_isothermal_cstr": "cstr",
    "pfr_chain": "pfr",
    "packed_bed_1d": "packed_bed",
    "steady_pipe_flow": "pipe_flow",
    "transient_pipe_flow": "pipe_flow_transient",
    "q2d_cmr": "q2d",
}
_MODEL_ALIASES = {
    "pod_deeponet": "pod-deeponet",
    "pod": "pod-deeponet",
    "deep_onet": "deeponet",
}


def list_problems() -> tuple[str, ...]:
    return tuple(_PROBLEMS)


def list_models() -> tuple[str, ...]:
    return tuple(_MODELS)


def get_problem_spec(name: str) -> ProblemSpec:
    key = _PROBLEM_ALIASES.get(name, name)
    try:
        return _PROBLEMS[key]
    except KeyError as exc:
        raise KeyError(
            f"Unknown problem {name!r}; choose from {', '.join(list_problems())}."
        ) from exc


def get_model_spec(name: str) -> ModelSpec:
    key = _MODEL_ALIASES.get(name, name)
    try:
        return _MODELS[key]
    except KeyError as exc:
        raise KeyError(
            f"Unknown model {name!r}; choose from {', '.join(list_models())}."
        ) from exc


__all__ = [
    "get_model_spec",
    "get_problem_spec",
    "list_models",
    "list_problems",
]
