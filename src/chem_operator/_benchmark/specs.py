"""Serializable specifications used by the reactor benchmark runner."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal


ChannelSource = Literal["parameter", "constant", "field", "species"]
BenchmarkStatus = Literal["completed", "failed", "unsupported"]


@dataclass(frozen=True)
class ChannelSpec:
    """One scalar control or spatial model-output channel."""

    label: str
    source: ChannelSource
    key: str
    species: str | None = None
    display_name: str | None = None
    unit: str = "-"

    def __post_init__(self) -> None:
        if not self.label or not self.key:
            raise ValueError("Channel labels and keys must be non-empty.")
        if self.source == "species" and not self.species:
            raise ValueError(f"Species channel {self.label!r} needs a species.")
        if self.source != "species" and self.species is not None:
            raise ValueError("Only species channels may set species.")

    @property
    def title(self) -> str:
        return self.display_name or self.label

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ChannelSpec":
        return cls(**dict(values))


@dataclass(frozen=True)
class ProblemSpec:
    """Describe a reactor dataset and its common operator-learning contract."""

    name: str
    display_name: str
    dataset_dir: str
    file_stem: str
    task: str
    coordinate_names: tuple[str, ...]
    input_channels: tuple[ChannelSpec, ...]
    output_channels: tuple[ChannelSpec, ...]
    spatial_ndim: int
    coordinate_stride: int = 1
    resample_shape: tuple[int, ...] | None = None
    categorical_encoders: Mapping[str, tuple[str, ...]] = field(
        default_factory=dict
    )
    qoi_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.spatial_ndim not in {1, 2}:
            raise ValueError("The benchmark currently supports 1D and 2D problems.")
        if len(self.coordinate_names) != self.spatial_ndim:
            raise ValueError(
                "coordinate_names must have one entry per spatial dimension."
            )
        if self.coordinate_stride < 1:
            raise ValueError("coordinate_stride must be positive.")
        if not self.input_channels or not self.output_channels:
            raise ValueError("Problems need at least one input and output channel.")
        labels = [
            item.label for item in self.input_channels + self.output_channels
        ]
        if len(labels) != len(set(labels)):
            raise ValueError(f"Problem {self.name!r} has duplicate channel labels.")
        if self.resample_shape is not None:
            if (
                len(self.resample_shape) != self.spatial_ndim
                or min(self.resample_shape) < 2
            ):
                raise ValueError(
                    "resample_shape must contain at least two points per dimension."
                )

    @property
    def coordinate_name(self) -> str:
        """Primary coordinate used by ``ChemOperatorDataset`` windowing."""

        return self.coordinate_names[0]

    @property
    def required_fields(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                channel.key
                for channel in self.input_channels + self.output_channels
                if channel.source in {"field", "species"}
            )
        )

    @property
    def required_constants(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                channel.key
                for channel in self.input_channels + self.output_channels
                if channel.source == "constant"
            )
        )

    def split_path(
        self,
        split: str,
        *,
        data_root: str | Path | None = None,
    ) -> Path:
        if split not in {"train", "valid", "test"}:
            raise ValueError(f"Unsupported dataset split {split!r}.")
        if data_root is None:
            root = Path(__file__).resolve().parents[3] / "datasets"
        else:
            root = Path(data_root)
        return root / self.dataset_dir / f"{self.file_stem}_{split}.h5"

    def with_output_channels(self, labels: Sequence[str]) -> "ProblemSpec":
        requested = tuple(labels)
        by_name = {channel.label: channel for channel in self.output_channels}
        missing = sorted(set(requested) - set(by_name))
        if missing:
            raise KeyError(
                f"Unknown output channels for {self.name}: {', '.join(missing)}"
            )
        if not requested:
            raise ValueError("At least one output channel must be selected.")
        return replace(
            self,
            output_channels=tuple(by_name[name] for name in requested),
        )

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["categorical_encoders"] = {
            name: list(categories)
            for name, categories in self.categorical_encoders.items()
        }
        return values

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ProblemSpec":
        data = dict(values)
        data["coordinate_names"] = tuple(data["coordinate_names"])
        data["input_channels"] = tuple(
            ChannelSpec.from_dict(item) for item in data["input_channels"]
        )
        data["output_channels"] = tuple(
            ChannelSpec.from_dict(item) for item in data["output_channels"]
        )
        if data.get("resample_shape") is not None:
            data["resample_shape"] = tuple(data["resample_shape"])
        data["categorical_encoders"] = {
            str(name): tuple(categories)
            for name, categories in data.get("categorical_encoders", {}).items()
        }
        data["qoi_names"] = tuple(data.get("qoi_names", ()))
        return cls(**data)


SearchSpaceFactory = Callable[["BenchmarkConfig"], Mapping[str, Any]]
BackendFactory = Callable[[str], Any]
AdapterFactory = Callable[[Any, ProblemSpec], Any]


@dataclass(frozen=True)
class ModelSpec:
    """Describe one model family and its benchmark factories."""

    name: str
    display_name: str
    backend: str
    supported_dimensions: tuple[int, ...] = (1, 2)
    checkpoint_schema: str = "chem-operator-benchmark-model-v1"
    default_hyperparameters: Mapping[str, Any] = field(default_factory=dict)
    search_space_factory: SearchSpaceFactory | None = field(
        default=None, repr=False, compare=False
    )
    backend_factory: BackendFactory | None = field(
        default=None, repr=False, compare=False
    )
    adapter_factory: AdapterFactory | None = field(
        default=None, repr=False, compare=False
    )

    def supports(self, problem: ProblemSpec) -> bool:
        return problem.spatial_ndim in self.supported_dimensions

    def search_space(self, config: "BenchmarkConfig") -> Mapping[str, Any]:
        if self.search_space_factory is None:
            raise RuntimeError(f"Model {self.name!r} has no search-space factory.")
        return self.search_space_factory(config)

    def make_backend(self) -> Any:
        if self.backend_factory is None:
            raise RuntimeError(f"Model {self.name!r} has no backend factory.")
        return self.backend_factory(self.backend)

    def make_adapter(self, dataset: Any, problem: ProblemSpec) -> Any:
        if self.adapter_factory is None:
            raise RuntimeError(f"Model {self.name!r} has no adapter factory.")
        return self.adapter_factory(dataset, problem)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "backend": self.backend,
            "supported_dimensions": list(self.supported_dimensions),
            "checkpoint_schema": self.checkpoint_schema,
            "default_hyperparameters": dict(self.default_hyperparameters),
        }


@dataclass(frozen=True)
class BenchmarkConfig:
    """Runtime, tuning, resource, and artifact settings for one matrix cell."""

    problem: str
    model: str
    seed: int = 42
    output_root: str | Path = "artifacts/benchmarks"
    run_id: str | None = None
    data_root: str | Path | None = None
    tune_time_budget_s: float = 3600.0
    tune_num_samples: int = 20
    tune_epochs: int = 50
    final_epochs: int = 100
    early_stopping_patience: int = 15
    checkpoint_interval: int = 10
    max_concurrent_trials: int = 1
    cpus_per_trial: float = 2.0
    gpus_per_trial: float | None = None
    num_workers: int = 0
    max_train_cases: int | None = None
    max_valid_cases: int | None = None
    max_test_cases: int | None = None
    channel_overrides: tuple[str, ...] | None = None
    device: str = "auto"
    use_ray: bool = True
    resume: bool = True
    inference_warmups: int = 10
    inference_repeats: int = 50
    fail_fast: bool = False

    def __post_init__(self) -> None:
        if not self.problem or not self.model:
            raise ValueError("problem and model must be non-empty.")
        positive = {
            "tune_time_budget_s": self.tune_time_budget_s,
            "tune_num_samples": self.tune_num_samples,
            "tune_epochs": self.tune_epochs,
            "final_epochs": self.final_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "checkpoint_interval": self.checkpoint_interval,
            "max_concurrent_trials": self.max_concurrent_trials,
            "cpus_per_trial": self.cpus_per_trial,
            "inference_repeats": self.inference_repeats,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(
                "Benchmark settings must be positive: " + ", ".join(invalid)
            )
        if self.num_workers < 0 or self.inference_warmups < 0:
            raise ValueError("Worker and warm-up counts cannot be negative.")
        for name in ("max_train_cases", "max_valid_cases", "max_test_cases"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be positive when supplied.")
        if self.gpus_per_trial is not None and self.gpus_per_trial < 0:
            raise ValueError("gpus_per_trial cannot be negative.")
        if self.channel_overrides is not None:
            object.__setattr__(
                self, "channel_overrides", tuple(self.channel_overrides)
            )

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["output_root"] = str(self.output_root)
        if self.data_root is not None:
            values["data_root"] = str(self.data_root)
        return values

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "BenchmarkConfig":
        data = dict(values)
        if data.get("channel_overrides") is not None:
            data["channel_overrides"] = tuple(data["channel_overrides"])
        return cls(**data)

    def for_cell(self, problem: str, model: str) -> "BenchmarkConfig":
        return replace(self, problem=problem, model=model)


@dataclass
class BenchmarkResult:
    """Serializable outcome for one model/problem benchmark cell."""

    status: BenchmarkStatus
    problem: str
    model: str
    run_id: str
    run_directory: str
    metrics: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, Any] = field(default_factory=dict)
    parameter_count: int = 0
    checkpoint_size_bytes: int = 0
    artifacts: dict[str, str] = field(default_factory=dict)
    best_config: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "BenchmarkResult":
        return cls(**dict(values))


__all__ = [
    "BenchmarkConfig",
    "BenchmarkResult",
    "BenchmarkStatus",
    "ChannelSpec",
    "ModelSpec",
    "ProblemSpec",
]
