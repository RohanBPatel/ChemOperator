"""Ray Tune orchestration for the reactor-operator benchmark matrix."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
import math
from pathlib import Path
import tempfile
import time
import traceback
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from chem_operator._benchmark.artifacts import (
    ArtifactManifest,
    CheckpointBundle,
    SCHEMA_VERSION,
    load_torch,
    make_run_id,
    save_pod,
    write_history,
    write_json,
)
from chem_operator._benchmark.metrics import (
    compute_metrics,
    summarize_timings,
)
from chem_operator._benchmark.registry import (
    get_model_spec,
    get_problem_spec,
)
from chem_operator._benchmark.specs import (
    BenchmarkConfig,
    BenchmarkResult,
    ChannelSpec,
    ModelSpec,
    ProblemSpec,
)
from chem_operator._benchmark.trainers.base import (
    GridDeepONetDataset,
    ResampledOperatorDataset,
    select_device,
)
from chem_operator.datasets import ChemOperatorDataset
from chem_operator.models import (
    FNOAdapter,
    FNOChannel,
    fit_fno_zscore_normalizer,
)
from chem_operator.normalization import (
    Normalizer,
    ZScoreNormalizer,
    normalizer_from_state_dict,
)


OBJECTIVE = "valid_normalized_rmse_macro"


class _LimitedDataset(Dataset):
    def __init__(self, dataset: Dataset, maximum: int | None) -> None:
        self.dataset = dataset
        self.maximum = maximum

    def __len__(self) -> int:
        size = len(self.dataset)
        return size if self.maximum is None else min(size, self.maximum)

    def __getitem__(self, index: int) -> Any:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return self.dataset[index]


class _MetadataDataset(Dataset):
    """Add resolved species and encoded categorical controls lazily."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        species_by_field: Mapping[str, Sequence[str]],
        categorical_encoders: Mapping[str, Sequence[str]],
    ) -> None:
        self.dataset = dataset
        self.species_by_field = {
            str(field): tuple(names)
            for field, names in species_by_field.items()
        }
        self.categorical_encoders = {
            str(name): tuple(categories)
            for name, categories in categorical_encoders.items()
        }

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        original = self.dataset[index]
        sample = dict(original)
        metadata = dict(original.get("metadata", {}))
        params = dict(metadata.get("params", {}))
        for name, categories in self.categorical_encoders.items():
            if name not in params:
                raise KeyError(f"Categorical parameter {name!r} is unavailable.")
            value = str(params[name])
            try:
                category_index = categories.index(value)
            except ValueError as exc:
                raise ValueError(
                    f"Unknown {name} category {value!r}; expected {categories}."
                ) from exc
            denominator = max(len(categories) - 1, 1)
            params[f"{name}__code"] = category_index / denominator
        metadata["params"] = params
        existing = dict(metadata.get("field_species", {}))
        existing.update(self.species_by_field)
        metadata["field_species"] = existing
        sample["metadata"] = metadata
        return sample


def _fno_channel(spec: ChannelSpec) -> FNOChannel:
    return FNOChannel(
        label=spec.label,
        source=spec.source,
        key=spec.key,
        species=spec.species,
        display_name=spec.display_name,
        unit=spec.unit,
    )


def _fallback_species(dataset: ChemOperatorDataset) -> dict[str, tuple[str, ...]]:
    """Resolve mechanism species if older records omit field metadata."""

    result = {
        field: dataset.species_names(field)
        for field in dataset.field_names
        if dataset.species_names(field)
    }
    required = [
        field
        for field in dataset.field_names
        if field.rsplit("/", 1)[-1].casefold() in {"x", "y"}
    ]
    if required and any(field not in result for field in required):
        sample = dataset[0]
        mechanism = sample.get("metadata", {}).get("mechanism")
        if mechanism:
            import cantera as ct

            gas = ct.Solution(str(mechanism))
            for field in required:
                descriptor = dataset.inspect_fields()[field]
                if descriptor["channel_count"] == gas.n_species:
                    result[field] = tuple(gas.species_names)
    return result


class PreparedProblem:
    """Open split-local readers and expose one common normalized channel set."""

    def __init__(
        self,
        spec: ProblemSpec,
        config: BenchmarkConfig,
        *,
        preprocessing_state: Mapping[str, Any] | None = None,
        splits: Sequence[str] = ("train", "valid", "test"),
    ) -> None:
        self.spec = spec
        self.config = config
        self.raw: dict[str, ChemOperatorDataset] = {}
        self.operator: dict[str, ResampledOperatorDataset] = {}
        self.deeponet: dict[str, GridDeepONetDataset] = {}
        self.preprocessing_seconds = 0.0
        self.input_channels = tuple(
            _fno_channel(channel) for channel in spec.input_channels
        )
        self.output_channels = tuple(
            _fno_channel(channel) for channel in spec.output_channels
        )

        try:
            for split in splits:
                self.raw[split] = self._open_raw(split)
            discovery = self.raw.get("train") or next(iter(self.raw.values()))
            species_by_field = _fallback_species(discovery)
            wrapped: dict[str, Dataset] = {}
            for split, raw in self.raw.items():
                limit = getattr(config, f"max_{split}_cases")
                wrapped[split] = _MetadataDataset(
                    _LimitedDataset(raw, limit),
                    species_by_field=species_by_field,
                    categorical_encoders=spec.categorical_encoders,
                )
            tic = time.perf_counter()
            if preprocessing_state is None:
                if "train" not in wrapped:
                    raise ValueError(
                        "Training data is required to fit preprocessing."
                    )
                self.normalizer = fit_fno_zscore_normalizer(
                    wrapped["train"],
                    self.input_channels,
                    self.output_channels,
                )
            else:
                restored = normalizer_from_state_dict(preprocessing_state)
                if not isinstance(restored, ZScoreNormalizer):
                    raise TypeError("Benchmarks require a Z-score normalizer.")
                self.normalizer = restored
            self.preprocessing_seconds = time.perf_counter() - tic

            for split, dataset in wrapped.items():
                fno = FNOAdapter(
                    dataset,
                    self.normalizer,
                    input_channels=self.input_channels,
                    output_channels=self.output_channels,
                    coordinate_names=spec.coordinate_names,
                )
                common = ResampledOperatorDataset(
                    fno,
                    coordinate_names=spec.coordinate_names,
                    output_labels=[
                        channel.label for channel in spec.output_channels
                    ],
                    resample_shape=spec.resample_shape,
                )
                self.operator[split] = common
                self.deeponet[split] = GridDeepONetDataset(
                    common,
                    coordinate_names=spec.coordinate_names,
                    output_labels=[
                        channel.label for channel in spec.output_channels
                    ],
                )
            self.dataset_fingerprint = discovery.fingerprint()
        except BaseException:
            self.close()
            raise

    def _open_raw(self, split: str) -> ChemOperatorDataset:
        return ChemOperatorDataset(
            self.spec.split_path(split, data_root=self.config.data_root),
            task=self.spec.task,
            coordinate_name=self.spec.coordinate_name,
            input_fields=self.spec.required_fields,
            output_fields=self.spec.required_fields,
            constant_inputs=self.spec.required_constants,
            n_steps_input=1,
            n_steps_output=1,
            index_stride=self.spec.coordinate_stride,
            dtype=torch.float32,
        )

    @property
    def preprocessing_state(self) -> dict[str, Any]:
        return self.normalizer.state_dict()

    @property
    def train_scales(self) -> dict[str, float]:
        return {
            channel.label: float(self.normalizer.stds[channel.label])
            for channel in self.spec.output_channels
        }

    def model_dataset(self, split: str, model: str) -> Dataset:
        if model == "fno":
            return self.operator[split]
        return self.deeponet[split]

    def normalized_targets(self, split: str) -> torch.Tensor:
        return torch.stack(
            [self.operator[split][index]["y"] for index in range(len(self.operator[split]))]
        )

    def physical_targets(self, split: str) -> torch.Tensor:
        return self.operator[split].denormalize_output(
            self.normalized_targets(split)
        )

    def mean_solver_wall_time(self, split: str = "test") -> float | None:
        values: list[float] = []
        for index in range(len(self.operator[split])):
            metadata = self.operator[split].physical_item(index).get(
                "metadata", {}
            )
            value = metadata.get("wall_time")
            if value is not None and math.isfinite(float(value)):
                values.append(float(value))
        return None if not values else float(np.mean(values))

    def close(self) -> None:
        for dataset in getattr(self, "raw", {}).values():
            dataset.close()

    def __enter__(self) -> "PreparedProblem":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class _RayEpochReporter:
    def __init__(self, checkpoint_interval: int) -> None:
        self.checkpoint_interval = checkpoint_interval

    def __call__(
        self,
        metrics: Mapping[str, float | int],
        model: torch.nn.Module,
        extra: Mapping[str, Any],
    ) -> None:
        from ray import tune

        epoch = int(metrics["epoch"])
        should_checkpoint = bool(metrics.get("improved", 0)) or (
            epoch % self.checkpoint_interval == 0
        )
        report = dict(metrics)
        if not should_checkpoint:
            tune.report(report)
            return
        with tempfile.TemporaryDirectory(prefix="chem-operator-ray-") as root:
            state = {
                "epoch": epoch,
                "state_dict": {
                    name: value.detach().cpu()
                    for name, value in model.state_dict().items()
                    if isinstance(value, torch.Tensor)
                },
                **dict(extra),
            }
            torch.save(state, Path(root) / "trial_state.pt")
            tune.report(
                report,
                checkpoint=tune.Checkpoint.from_directory(root),
            )


def _load_ray_resume_state() -> Mapping[str, Any] | None:
    from ray import tune

    checkpoint = tune.get_checkpoint()
    if checkpoint is None:
        return None
    with checkpoint.as_directory() as directory:
        path = Path(directory) / "trial_state.pt"
        if not path.is_file():
            return None
        return torch.load(path, map_location="cpu", weights_only=True)


def _ray_trial(
    trial_config: Mapping[str, Any],
    *,
    benchmark_config: Mapping[str, Any],
    problem_spec: Mapping[str, Any],
    preprocessing_state: Mapping[str, Any],
) -> None:
    """Top-level Ray trainable: all HDF5 readers are process-local."""

    config = BenchmarkConfig.from_dict(benchmark_config)
    problem = ProblemSpec.from_dict(problem_spec)
    model_spec = get_model_spec(config.model)
    torch.set_num_threads(max(1, int(config.cpus_per_trial)))
    with PreparedProblem(
        problem,
        config,
        preprocessing_state=preprocessing_state,
        splits=("train", "valid"),
    ) as prepared:
        backend = model_spec.make_backend()
        resume = _load_ray_resume_state()
        set_resume = getattr(backend, "set_resume_state", None)
        if set_resume is not None:
            set_resume(resume)
        device = select_device(config.device, config.gpus_per_trial)
        backend.fit(
            prepared.model_dataset("train", model_spec.name),
            prepared.model_dataset("valid", model_spec.name),
            config=trial_config,
            epochs=config.tune_epochs,
            patience=config.early_stopping_patience,
            seed=config.seed,
            device=device,
            reporter=_RayEpochReporter(config.checkpoint_interval),
        )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _inference_timings(
    backend: Any,
    prepared: PreparedProblem,
    model_name: str,
    *,
    device: torch.device,
    warmups: int,
    repeats: int,
) -> list[float]:
    if model_name == "fno":
        sample = prepared.operator["test"][0]
        inputs: Any = sample["x"].unsqueeze(0)
    else:
        sample = prepared.deeponet["test"][0]
        scaler = backend.coordinate_scaler
        if scaler is None:
            raise RuntimeError("DeepONet coordinate scaler is unavailable.")
        inputs = (
            sample["branch"].unsqueeze(0),
            scaler.transform_tensor(sample["trunk"]),
        )
    for _ in range(warmups):
        backend.predict_tensor(inputs, device=device)
    _synchronize(device)
    values: list[float] = []
    for _ in range(repeats):
        tic = time.perf_counter()
        backend.predict_tensor(inputs, device=device)
        _synchronize(device)
        values.append(time.perf_counter() - tic)
    return values


def _relative_qoi(reference: torch.Tensor, prediction: torch.Tensor) -> float:
    error = torch.sqrt(torch.sum((prediction - reference).reshape(-1) ** 2))
    scale = torch.sqrt(torch.sum(reference.reshape(-1) ** 2)).clamp_min(
        1.0e-12
    )
    return float(error / scale)


def _qoi_metrics(
    prepared: PreparedProblem,
    reference: torch.Tensor,
    prediction: torch.Tensor,
) -> dict[str, float]:
    problem = prepared.spec.name
    labels = [
        channel.label for channel in prepared.spec.output_channels
    ]
    qoi: dict[str, float] = {}
    if problem in {"cstr", "pfr", "packed_bed"} and "T" in labels:
        index = labels.index("T")
        name = (
            "final_temperature"
            if problem == "cstr"
            else "outlet_temperature"
        )
        qoi[f"{name}_relative_l2"] = _relative_qoi(
            reference[:, index, -1],
            prediction[:, index, -1],
        )
    if problem in {"pipe_flow", "pipe_flow_transient"}:
        index = labels.index("velocity")
        true_flow: list[torch.Tensor] = []
        predicted_flow: list[torch.Tensor] = []
        for case in range(reference.shape[0]):
            radius = prepared.operator["test"][case]["r"]
            factor = 2.0 * math.pi * radius
            true_flow.append(
                torch.trapezoid(
                    reference[case, index] * factor,
                    radius,
                    dim=-1,
                )
            )
            predicted_flow.append(
                torch.trapezoid(
                    prediction[case, index] * factor,
                    radius,
                    dim=-1,
                )
            )
        qoi["flow_rate_relative_l2"] = _relative_qoi(
            torch.stack(true_flow), torch.stack(predicted_flow)
        )
    if problem == "q2d" and "X_CH4" in labels:
        index = labels.index("X_CH4")
        true_inlet = reference[:, index, 0].mean(dim=-1).clamp_min(1.0e-12)
        pred_inlet = prediction[:, index, 0].mean(dim=-1).clamp_min(1.0e-12)
        true_conversion = 1.0 - reference[:, index, -1].mean(dim=-1) / true_inlet
        pred_conversion = (
            1.0 - prediction[:, index, -1].mean(dim=-1) / pred_inlet
        )
        qoi["outlet_conversion_relative_l2"] = _relative_qoi(
            true_conversion, pred_conversion
        )
    return qoi


class BenchmarkRunner:
    """Run cells while caching train-only preprocessing across model families."""

    def __init__(self) -> None:
        self._preprocessing_cache: dict[
            tuple[str, tuple[str, ...], str | None], Mapping[str, Any]
        ] = {}

    @staticmethod
    def _resolved(
        config: BenchmarkConfig,
    ) -> tuple[ProblemSpec, ModelSpec]:
        problem = get_problem_spec(config.problem)
        if config.channel_overrides is not None:
            requested = tuple(config.channel_overrides)
            full_cstr = problem.name == "cstr" and any(
                name in {"all", "X:*"} for name in requested
            )
            if full_cstr:
                import cantera as ct

                species = ct.Solution(
                    "n-heptane-NUIG-2016.yaml"
                ).species_names
                expanded = tuple(
                    ChannelSpec(
                        label=f"X_{name}",
                        source="species",
                        key="X",
                        species=name,
                        display_name=f"X({name})",
                    )
                    for name in species
                )
                if "all" in requested:
                    scalar = tuple(
                        channel
                        for channel in problem.output_channels
                        if channel.source == "field"
                    )
                    problem = replace(
                        problem, output_channels=scalar + expanded
                    )
                else:
                    explicit = tuple(
                        name for name in requested if name != "X:*"
                    )
                    scalar = (
                        problem.with_output_channels(explicit).output_channels
                        if explicit
                        else ()
                    )
                    existing_labels = {channel.label for channel in scalar}
                    problem = replace(
                        problem,
                        output_channels=scalar
                        + tuple(
                            channel
                            for channel in expanded
                            if channel.label not in existing_labels
                        ),
                    )
            else:
                problem = problem.with_output_channels(requested)
        model = get_model_spec(config.model)
        return problem, model

    @staticmethod
    def _run_id(config: BenchmarkConfig) -> str:
        return config.run_id or make_run_id()

    def _tune(
        self,
        config: BenchmarkConfig,
        problem: ProblemSpec,
        model: ModelSpec,
        preprocessing_state: Mapping[str, Any],
        run_directory: Path,
    ) -> tuple[dict[str, Any], float]:
        if not config.use_ray:
            return dict(model.default_hyperparameters), 0.0

        import optuna
        import ray
        from ray import tune
        from ray.tune.schedulers import ASHAScheduler
        from ray.tune.search.optuna import OptunaSearch

        started_ray = False
        if not ray.is_initialized():
            ray.init(include_dashboard=False, ignore_reinit_error=True)
            started_ray = True
        search = OptunaSearch(
            metric=OBJECTIVE,
            mode="min",
            sampler=optuna.samplers.TPESampler(
                seed=config.seed,
                n_startup_trials=min(3, config.tune_num_samples),
            ),
        )
        scheduler = ASHAScheduler(
            metric=OBJECTIVE,
            mode="min",
            time_attr="training_iteration",
            max_t=config.tune_epochs,
            grace_period=max(1, config.tune_epochs // 4),
            reduction_factor=2,
        )
        parameterized = tune.with_parameters(
            _ray_trial,
            benchmark_config=config.to_dict(),
            problem_spec=problem.to_dict(),
            preprocessing_state=dict(preprocessing_state),
        )
        trainable = tune.with_resources(
            parameterized,
            resources={
                "cpu": config.cpus_per_trial,
                "gpu": (
                    float(torch.cuda.is_available())
                    if config.gpus_per_trial is None
                    else config.gpus_per_trial
                ),
            },
        )
        ray_root = (run_directory / "ray").resolve()
        experiment_name = "tune"
        restore_path = ray_root / experiment_name
        tic = time.perf_counter()
        try:
            if config.resume and tune.Tuner.can_restore(str(restore_path)):
                tuner = tune.Tuner.restore(
                    str(restore_path),
                    trainable=trainable,
                    resume_unfinished=True,
                    resume_errored=True,
                )
            else:
                tuner = tune.Tuner(
                    trainable,
                    param_space=dict(model.search_space(config)),
                    tune_config=tune.TuneConfig(
                        search_alg=search,
                        scheduler=scheduler,
                        num_samples=config.tune_num_samples,
                        max_concurrent_trials=config.max_concurrent_trials,
                        time_budget_s=config.tune_time_budget_s,
                        reuse_actors=False,
                    ),
                    run_config=tune.RunConfig(
                        name=experiment_name,
                        storage_path=str(ray_root),
                        verbose=1,
                    ),
                )
            results = tuner.fit()
            best = results.get_best_result(
                metric=OBJECTIVE,
                mode="min",
                scope="all",
            )
            objective_value = best.metrics.get(OBJECTIVE)
            if objective_value is None or not math.isfinite(
                float(objective_value)
            ):
                raise RuntimeError(
                    "Ray Tune produced no successful trial with the "
                    f"{OBJECTIVE!r} objective."
                )
            best_config = {
                key: (
                    value.item()
                    if isinstance(value, np.generic)
                    else value
                )
                for key, value in best.config.items()
            }
        finally:
            elapsed = time.perf_counter() - tic
            if started_ray:
                ray.shutdown()
        return best_config, elapsed

    def run(self, config: BenchmarkConfig) -> BenchmarkResult:
        problem, model = self._resolved(config)
        run_id = self._run_id(config)
        config = replace(config, run_id=run_id)
        run_directory = (
            Path(config.output_root) / run_id / problem.name / model.name
        ).resolve()
        run_directory.mkdir(parents=True, exist_ok=True)
        if not model.supports(problem):
            result = BenchmarkResult(
                status="unsupported",
                problem=problem.name,
                model=model.name,
                run_id=run_id,
                run_directory=str(run_directory),
                error=(
                    f"{model.display_name} does not support "
                    f"{problem.spatial_ndim}D data."
                ),
            )
            write_json(run_directory / "result.json", result.to_dict())
            return result

        write_json(
            run_directory / "resolved_config.json",
            {
                "benchmark": config.to_dict(),
                "problem": problem.to_dict(),
                "model": model.to_dict(),
            },
        )
        key = (
            problem.name,
            tuple(channel.label for channel in problem.output_channels),
            None if config.data_root is None else str(config.data_root),
        )
        try:
            cached = self._preprocessing_cache.get(key)
            with PreparedProblem(
                problem,
                config,
                preprocessing_state=cached,
            ) as prepared:
                self._preprocessing_cache[key] = prepared.preprocessing_state
                best_config, tune_seconds = self._tune(
                    config,
                    problem,
                    model,
                    prepared.preprocessing_state,
                    run_directory,
                )
                write_json(run_directory / "best_config.json", best_config)

                backend = model.make_backend()
                device = select_device(config.device, config.gpus_per_trial)
                tic = time.perf_counter()
                outcome = backend.fit(
                    prepared.model_dataset("train", model.name),
                    prepared.model_dataset("valid", model.name),
                    config=best_config,
                    epochs=config.final_epochs,
                    patience=config.early_stopping_patience,
                    seed=config.seed,
                    device=device,
                )
                final_training_seconds = time.perf_counter() - tic

                batch_size = int(best_config.get("batch_size", 1))
                predicted_normalized = backend.predict(
                    prepared.model_dataset("test", model.name),
                    batch_size=batch_size,
                    device=device,
                )
                reference_normalized = prepared.normalized_targets("test")
                prediction = prepared.operator["test"].denormalize_output(
                    predicted_normalized
                )
                reference = prepared.operator["test"].denormalize_output(
                    reference_normalized
                )
                labels = [
                    channel.label for channel in problem.output_channels
                ]
                mass_fraction_channels = (
                    [label for label in labels if label.startswith("X_")]
                    if problem.name == "packed_bed"
                    else None
                )
                metric_set = compute_metrics(
                    reference,
                    prediction,
                    labels,
                    prepared.train_scales,
                    channel_axis=1,
                    mass_fraction_channels=mass_fraction_channels,
                )
                metrics = metric_set.to_dict()
                metrics["qoi"] = _qoi_metrics(
                    prepared, reference, prediction
                )
                metrics["best_validation_normalized_rmse_macro"] = (
                    outcome.best_valid_loss
                )
                metrics["best_epoch"] = outcome.best_epoch

                latencies = _inference_timings(
                    backend,
                    prepared,
                    model.name,
                    device=device,
                    warmups=config.inference_warmups,
                    repeats=config.inference_repeats,
                )
                timing = summarize_timings(
                    latencies,
                    preprocessing_seconds=prepared.preprocessing_seconds,
                    tune_wall_seconds=tune_seconds,
                    final_training_seconds=final_training_seconds,
                    solver_wall_seconds=prepared.mean_solver_wall_time(),
                    warmup_repeats=config.inference_warmups,
                    extra={"test_cases": len(prepared.operator["test"])},
                )

                checkpoint = CheckpointBundle(
                    model_state=backend.checkpoint_state(),
                    preprocessing_state={
                        "normalizer": prepared.preprocessing_state,
                        "problem_spec": problem.to_dict(),
                        "dataset_fingerprint": prepared.dataset_fingerprint,
                    },
                    problem_spec=problem.to_dict(),
                    model_spec=model.to_dict(),
                    best_config=best_config,
                    dataset_fingerprint=prepared.dataset_fingerprint,
                )
                artifact_paths = checkpoint.save(run_directory)
                artifact_paths["resolved_config"] = (
                    run_directory / "resolved_config.json"
                )
                artifact_paths["best_config"] = (
                    run_directory / "best_config.json"
                )
                if (run_directory / "ray").is_dir():
                    artifact_paths["ray"] = run_directory / "ray"
                if getattr(backend, "pod", None) is not None:
                    artifact_paths["pod"] = save_pod(
                        run_directory / "pod.npz", backend.pod
                    )
                artifact_paths["metrics"] = write_json(
                    run_directory / "metrics.json", metrics
                )
                artifact_paths["timings"] = write_json(
                    run_directory / "timings.json", timing.to_dict()
                )
                artifact_paths["history"] = write_history(
                    run_directory / "history.csv", outcome.history
                )
                checkpoint_size = (run_directory / "model.pt").stat().st_size

                # A saved model is only considered successful if it reloads.
                reloaded = model.make_backend()
                reloaded.load_checkpoint_state(
                    load_torch(run_directory / "model.pt"),
                    device=device,
                )
                verification = reloaded.predict(
                    prepared.model_dataset("test", model.name),
                    batch_size=batch_size,
                    device=device,
                )
                if verification.shape != predicted_normalized.shape or not torch.allclose(
                    verification,
                    predicted_normalized,
                    rtol=1.0e-5,
                    atol=1.0e-6,
                ):
                    raise RuntimeError(
                        "Reloaded checkpoint did not reproduce final predictions."
                    )

                manifest = ArtifactManifest.create(
                    run_id=run_id,
                    problem=problem.name,
                    model=model.name,
                    seed=config.seed,
                    dataset_fingerprint=prepared.dataset_fingerprint,
                )
                manifest.files = {
                    name: str(path.relative_to(run_directory))
                    for name, path in artifact_paths.items()
                }
                manifest.files["result"] = "result.json"
                artifact_paths["manifest"] = manifest.save(
                    run_directory / "manifest.json"
                )
                result = BenchmarkResult(
                    status="completed",
                    problem=problem.name,
                    model=model.name,
                    run_id=run_id,
                    run_directory=str(run_directory),
                    metrics=metrics,
                    timings=timing.to_dict(),
                    parameter_count=outcome.parameter_count,
                    checkpoint_size_bytes=checkpoint_size,
                    artifacts={
                        name: str(path) for name, path in artifact_paths.items()
                    },
                    best_config=best_config,
                )
                write_json(run_directory / "result.json", result.to_dict())
                return result
        except Exception as error:
            result = BenchmarkResult(
                status="failed",
                problem=problem.name,
                model=model.name,
                run_id=run_id,
                run_directory=str(run_directory),
                error=f"{type(error).__name__}: {error}",
                artifacts={
                    "traceback": str(run_directory / "traceback.txt")
                },
            )
            (run_directory / "traceback.txt").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
            write_json(run_directory / "result.json", result.to_dict())
            if config.fail_fast:
                raise
            return result


class LoadedBenchmark:
    """Reloaded benchmark model with normalized-tensor and split prediction."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory).resolve()
        resolved = load_torch(self.directory / "preprocessing.pt")
        if resolved.get("schema") != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported benchmark checkpoint schema "
                f"{resolved.get('schema')!r}."
            )
        self.problem = ProblemSpec.from_dict(resolved["problem_spec"])
        self.dataset_fingerprint = str(resolved["dataset_fingerprint"])
        self.normalizer: Normalizer = normalizer_from_state_dict(
            resolved["normalizer"]
        )
        model_state = load_torch(self.directory / "model.pt")
        self.model_spec = get_model_spec(str(model_state["backend"]))
        self.backend = self.model_spec.make_backend()
        self.device = select_device("auto")
        self.backend.load_checkpoint_state(model_state, device=self.device)

    def predict(
        self,
        inputs: Any,
        *,
        device: str | torch.device | None = None,
        denormalize: bool = True,
    ) -> torch.Tensor:
        """Predict from normalized FNO tensors or normalized branch/trunk inputs."""

        selected = self.device if device is None else torch.device(device)
        values = self.backend.predict_tensor(inputs, device=selected)
        if not denormalize:
            return values
        labels = [channel.label for channel in self.problem.output_channels]
        if self.model_spec.name == "fno":
            fields = [
                self.normalizer.denormalize(values[:, index], label)
                for index, label in enumerate(labels)
            ]
            return torch.stack(fields, dim=1)
        fields = [
            self.normalizer.denormalize(values[..., index], label)
            for index, label in enumerate(labels)
        ]
        return torch.stack(fields, dim=-1)


def load_benchmark(path: str | Path) -> LoadedBenchmark:
    return LoadedBenchmark(path)


__all__ = [
    "BenchmarkRunner",
    "LoadedBenchmark",
    "PreparedProblem",
    "load_benchmark",
]
