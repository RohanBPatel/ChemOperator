"""Tune, train, and benchmark a Fourier neural operator for Q2D CMR data.

The current operator maps the two independently varied case parameters
``T0`` and ``SCCM`` to axial velocity and the CH4/H2 mole-fraction fields.
Channel extraction is declarative: extending ``INPUT_CHANNELS`` or
``OUTPUT_CHANNELS`` is sufficient to expose another parameter, constant,
scalar field, or species field to the rest of the workflow.

Converged solution fields should only be added to ``INPUT_CHANNELS`` when
they are genuinely known at inference time. Otherwise, doing so leaks the
prediction target into the model input.
"""

# Environment variables must be set before importing plotting and Ray.
# pylint: disable=wrong-import-position

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import re
import statistics
import time
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("RAY_memory_monitor_refresh_ms", "0")

import matplotlib.pyplot as plt
from neuralop.losses import LpLoss
from neuralop.models import FNO
import numpy as np
import optuna
import pandas as pd
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
import torch
from torch import nn
from torch.nn import functional as torch_functional
from torch.utils.data import DataLoader

from chem_operator.datasets import CanteraDataset
from chem_operator.models import (
    FNOAdapter,
    FNOChannel,
    fit_fno_zscore_normalizer,
)
from chem_operator.normalization import ZScoreNormalizer


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datasets" / "q2d_cmr"
OUTPUT_DIR = ROOT / "scripts" / "q2d_fno_results"
RAY_TEMP_DIR = ROOT / ".ray"

FILE_STEM = "q2d_cmr"
SEED = 42
METRIC = "best_valid_loss"

# Run-mode flags. Leave both false for tuning, training, and evaluation.
TRAIN_BEST_CONFIG_ONLY = False
PLOT_SAVED_MODEL_ONLY = False

TUNE_SAMPLES = 25
TUNE_EPOCHS = 30
FINAL_EPOCHS = 75
EVALUATION_BATCH_SIZE = 2

CPUS_PER_TRIAL = 2
GPUS_PER_TRIAL = 1 if torch.cuda.is_available() else 0
MAX_CONCURRENT_TRIALS = 1

LATENCY_WARMUPS = 5
LATENCY_REPEATS = 20
RELATIVE_L2_EPS = 1.0e-8

MESH_FILE_PATTERN = re.compile(r"q2d_cmr_(\d+)_(\d+)_test\.h5")

INPUT_CHANNELS = (
    FNOChannel(
        "T0",
        "parameter",
        "T0",
        display_name="Inlet temperature",
        unit="K",
    ),
    FNOChannel(
        "SCCM",
        "constant",
        "SCCM",
        display_name="Inlet flow rate",
        unit="sccm",
    ),
)

OUTPUT_CHANNELS = (
    FNOChannel(
        "velocity_axial",
        "field",
        "velocity_axial",
        display_name="Axial velocity",
        unit="m/s",
    ),
    FNOChannel(
        "X_CH4",
        "species",
        "X",
        species="CH4",
        display_name="Methane",
        unit="-",
    ),
    FNOChannel(
        "X_H2",
        "species",
        "X",
        species="H2",
        display_name="Hydrogen",
        unit="-",
    ),
    # FNOChannel(
    #     "theta_C(s)",
    #     "species",
    #     "theta",
    #     species="C(s)",
    #     display_name="Carbon Accumulation",
    #     unit="-",
    # ),
)


@dataclass(frozen=True)
class MeshFile:
    """One resolution-sweep HDF5 file."""

    path: Path
    n_z: int
    n_r: int

    @property
    def mesh_points(self) -> int:
        return self.n_z * self.n_r


def raw_dataset(path: Path) -> CanteraDataset:
    """Open one complete steady Q2D field per HDF5 case."""
    field_names = FNOAdapter.required_field_names(
        INPUT_CHANNELS,
        OUTPUT_CHANNELS,
    )
    if not field_names:
        raise ValueError("At least one configured channel must read an HDF5 field.")
    return CanteraDataset(
        path,
        task="field_map",
        coordinate_name="z",
        input_fields=field_names,
        output_fields=field_names,
        constant_inputs=FNOAdapter.required_constant_names(
            INPUT_CHANNELS,
            OUTPUT_CHANNELS,
        ),
        n_steps_input=1,
        dtype=torch.float32,
    )


def _case_geometry(metadata: Mapping[str, Any]) -> tuple[float, float]:
    """Return physical axial length and lumen radius for domain validation."""
    inputs = metadata.get("input_values", {})
    try:
        return float(inputs["LENGTH_LUMEN"]), float(inputs["CHANNELRAD_LUMEN"])
    except KeyError as exc:
        raise KeyError("Q2D metadata is missing its physical domain geometry.") from exc


def adapter_spatial_shape(
    dataset: FNOAdapter,
    dataset_name: str,
) -> tuple[int, int]:
    """Return and validate the common spatial shape stored by an adapter."""
    if len(dataset) == 0:
        raise RuntimeError(f"{dataset_name} contains no cases.")
    expected: tuple[int, int] | None = None
    for index in range(len(dataset)):
        physical = dataset.physical_item(index)
        input_shape = tuple(int(value) for value in physical["x"].shape[-2:])
        output_shape = tuple(int(value) for value in physical["y"].shape[-2:])
        if input_shape != output_shape:
            raise ValueError(
                f"{dataset_name} case {index} has input shape {input_shape} "
                f"and output shape {output_shape}."
            )
        if expected is None:
            expected = output_shape
        elif output_shape != expected:
            raise ValueError(
                f"{dataset_name} mixes spatial shapes {expected} and "
                f"{output_shape}."
            )
    assert expected is not None
    return expected


def fit_normalizer(
    path: Path,
) -> tuple[ZScoreNormalizer, tuple[float, float], tuple[int, int]]:
    """Fit training-only statistics and discover the training mesh shape."""
    dataset = raw_dataset(path)
    geometry: tuple[float, float] | None = None
    try:
        for index in range(len(dataset)):
            sample = dataset[index]
            current_geometry = _case_geometry(sample["metadata"])
            if geometry is None:
                geometry = current_geometry
            elif not np.allclose(
                current_geometry,
                geometry,
                rtol=1.0e-10,
                atol=1.0e-12,
            ):
                raise ValueError("Training cases do not share one physical domain.")
        normalizer = fit_fno_zscore_normalizer(
            dataset,
            INPUT_CHANNELS,
            OUTPUT_CHANNELS,
        )
        training_adapter = FNOAdapter(
            dataset,
            normalizer,
            input_channels=INPUT_CHANNELS,
            output_channels=OUTPUT_CHANNELS,
            coordinate_names=("z", "r"),
        )
        training_shape = adapter_spatial_shape(
            training_adapter,
            path.name,
        )
    finally:
        dataset.close()

    if geometry is None:
        raise RuntimeError("The training dataset contains no cases.")
    return normalizer, geometry, training_shape


def normalizer_state(
    normalizer: ZScoreNormalizer,
) -> dict[str, dict[str, torch.Tensor]]:
    """Return a CPU-only serializable normalizer state."""
    return {
        "mean": {
            name: value.detach().cpu() for name, value in normalizer.means.items()
        },
        "std": {
            name: value.detach().cpu() for name, value in normalizer.stds.items()
        },
        "mean_delta": {
            name: value.detach().cpu()
            for name, value in normalizer.delta_means.items()
        },
        "std_delta": {
            name: value.detach().cpu()
            for name, value in normalizer.delta_stds.items()
        },
    }


def normalizer_from_state(
    state: Mapping[str, Mapping[str, torch.Tensor]],
) -> ZScoreNormalizer:
    """Reconstruct the configured ChemOperator normalizer."""
    return ZScoreNormalizer(
        state,
        variable_field_order=tuple(spec.label for spec in OUTPUT_CHANNELS),
        constant_field_order=tuple(spec.label for spec in INPUT_CHANNELS),
    )


def make_adapter(
    path: Path,
    normalizer: ZScoreNormalizer,
    geometry: tuple[float, float],
) -> tuple[CanteraDataset, FNOAdapter]:
    """Open a raw dataset and its normalized Q2D adapter."""
    raw = raw_dataset(path)
    current_geometry = _case_geometry(raw[0]["metadata"])
    if not np.allclose(
        current_geometry,
        geometry,
        rtol=1.0e-10,
        atol=1.0e-12,
    ):
        raw.close()
        raise ValueError(
            f"Dataset geometry {current_geometry} differs from training "
            f"geometry {geometry}."
        )
    return raw, FNOAdapter(
        raw,
        normalizer,
        input_channels=INPUT_CHANNELS,
        output_channels=OUTPUT_CHANNELS,
        coordinate_names=("z", "r"),
    )


def model_from_config(
    config: Mapping[str, Any],
    device: torch.device,
) -> FNO:
    """Construct the configured two-dimensional NeuralOperator FNO."""
    return FNO(
        n_modes=(int(config["modes_z"]), int(config["modes_r"])),
        in_channels=len(INPUT_CHANNELS),
        out_channels=len(OUTPUT_CHANNELS),
        hidden_channels=int(config["hidden_channels"]),
        n_layers=int(config["n_layers"]),
        positional_embedding="grid",
        domain_padding=float(config.get("domain_padding", 0.0)),
    ).to(device)


def count_parameters(model: nn.Module) -> int:
    """Return the number of model parameters."""
    return sum(parameter.numel() for parameter in model.parameters())


def validation_loss(
    model: nn.Module,
    dataset: FNOAdapter,
    *,
    batch_size: int,
    device: torch.device,
) -> float:
    """Return mean global normalized relative L2 over a complete split."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    loss_function = LpLoss(d=3, p=2, reduction="mean", eps=RELATIVE_L2_EPS)
    model.eval()
    total = 0.0
    samples = 0
    with torch.no_grad():
        for batch in loader:
            target = batch["y"].to(device)
            prediction = model(batch["x"].to(device))
            loss = loss_function(prediction, target)
            count = target.shape[0]
            total += float(loss) * count
            samples += count
    return total / max(samples, 1)


def train_model(  # pylint: disable=too-many-arguments,too-many-locals
    config: Mapping[str, Any],
    train_data: FNOAdapter,
    valid_data: FNOAdapter,
    *,
    epochs: int,
    device: torch.device,
    report_to_ray: bool = False,
    print_epochs: bool = False,
) -> tuple[FNO, dict[str, list[float]], float]:
    """Train one FNO and restore its best validation checkpoint."""
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    model = model_from_config(config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    loss_function = LpLoss(d=3, p=2, reduction="mean", eps=RELATIVE_L2_EPS)
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    loader = DataLoader(
        train_data,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        generator=generator,
    )
    history = {"train_loss": [], "valid_loss": []}
    best_loss = float("inf")
    best_state = deepcopy(model.state_dict())
    tic = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        train_total = 0.0
        samples = 0
        for batch in loader:
            model_input = batch["x"].to(device)
            target = batch["y"].to(device)
            prediction = model(model_input)
            loss = loss_function(prediction, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            count = target.shape[0]
            train_total += float(loss.detach()) * count
            samples += count

        train_value = train_total / max(samples, 1)
        valid_value = validation_loss(
            model,
            valid_data,
            batch_size=int(config["batch_size"]),
            device=device,
        )
        history["train_loss"].append(train_value)
        history["valid_loss"].append(valid_value)
        if valid_value < best_loss:
            best_loss = valid_value
            best_state = deepcopy(model.state_dict())
        if print_epochs:
            print(
                f"Epoch {epoch:03d}/{epochs}: "
                f"train={train_value:.6e}, valid={valid_value:.6e}"
            )
        if report_to_ray:
            tune.report(
                {
                    "train_loss": train_value,
                    "valid_loss": valid_value,
                    METRIC: best_loss,
                    "n_params": count_parameters(model),
                }
            )

    elapsed = time.perf_counter() - tic
    model.load_state_dict(best_state)
    return model.eval(), history, elapsed


def ray_trial(
    config: Mapping[str, Any],
    *,
    data_dir: str,
    normalization: Mapping[str, Mapping[str, torch.Tensor]],
    geometry: tuple[float, float],
) -> None:
    """Ray trainable that creates process-local lazy HDF5 readers."""
    torch.set_num_threads(max(1, CPUS_PER_TRIAL))
    normalizer = normalizer_from_state(normalization)
    train_raw, train_data = make_adapter(
        Path(data_dir) / f"{FILE_STEM}_train.h5",
        normalizer,
        geometry,
    )
    valid_raw, valid_data = make_adapter(
        Path(data_dir) / f"{FILE_STEM}_valid.h5",
        normalizer,
        geometry,
    )
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_model(
            config,
            train_data,
            valid_data,
            epochs=TUNE_EPOCHS,
            device=device,
            report_to_ray=True,
        )
    finally:
        train_raw.close()
        valid_raw.close()


def tune_hyperparameters(
    data_dir: Path,
    output_dir: Path,
    normalizer: ZScoreNormalizer,
    geometry: tuple[float, float],
) -> dict[str, Any]:
    """Run Optuna/ASHA tuning and return the best final configuration."""
    search = OptunaSearch(
        metric=METRIC,
        mode="min",
        sampler=optuna.samplers.TPESampler(
            seed=SEED,
            n_startup_trials=4,
            multivariate=True,
        ),
    )
    scheduler = ASHAScheduler(
        metric=METRIC,
        mode="min",
        time_attr="training_iteration",
        max_t=TUNE_EPOCHS,
        grace_period=min(8, TUNE_EPOCHS),
        reduction_factor=2,
    )
    parameterized = tune.with_parameters(
        ray_trial,
        data_dir=str(data_dir.resolve()),
        normalization=normalizer_state(normalizer),
        geometry=geometry,
    )
    trainable = tune.with_resources(
        parameterized,
        resources={"cpu": CPUS_PER_TRIAL, "gpu": GPUS_PER_TRIAL},
    )
    run_name = time.strftime("q2d_fno_%Y%m%d_%H%M%S")
    tuner = tune.Tuner(
        trainable,
        param_space={
            "modes_z": tune.choice([4, 5, 6]),
            "modes_r": tune.choice([2, 3, 4]),
            "hidden_channels": tune.choice([16, 20, 24, 28, 32]),
            "n_layers": tune.choice([6, 7, 8, 9]),
            "learning_rate": tune.loguniform(1.0e-4, 4.0e-3),
            "weight_decay": tune.loguniform(1.0e-8, 1.0e-4),
            "batch_size": 2, #tune.choice([2, 4]),
            "domain_padding": tune.choice([0.0, 0.05, 0.1, 0.15, 0.2]),
        },
        tune_config=tune.TuneConfig(
            search_alg=search,
            scheduler=scheduler,
            num_samples=TUNE_SAMPLES,
            max_concurrent_trials=MAX_CONCURRENT_TRIALS,
            reuse_actors=False,
        ),
        run_config=tune.RunConfig(
            name=run_name,
            storage_path=str((output_dir / "ray_results").resolve()),
            verbose=1,
        ),
    )
    results = tuner.fit()
    best = results.get_best_result(metric=METRIC, mode="min", scope="last")
    return {
        key: value.item() if hasattr(value, "item") else value
        for key, value in best.config.items()
    }


def save_checkpoint(
    path: Path,
    model: FNO,
    config: Mapping[str, Any],
    normalizer: ZScoreNormalizer,
    geometry: tuple[float, float],
    training_shape: tuple[int, int],
) -> None:
    """Save model tensors and all data-interface configuration."""
    tensor_state = {
        name: value
        for name, value in model.state_dict().items()
        if isinstance(value, torch.Tensor)
    }
    torch.save(
        {
            "state_dict": tensor_state,
            "model_config": dict(config),
            "input_channels": [asdict(spec) for spec in INPUT_CHANNELS],
            "output_channels": [asdict(spec) for spec in OUTPUT_CHANNELS],
            "normalization": normalizer_state(normalizer),
            "geometry": tuple(geometry),
            "training_shape": tuple(training_shape),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    device: torch.device,
) -> tuple[FNO, ZScoreNormalizer, tuple[float, float], tuple[int, int]]:
    """Load and validate a serialized Q2D FNO."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    expected_inputs = [asdict(spec) for spec in INPUT_CHANNELS]
    expected_outputs = [asdict(spec) for spec in OUTPUT_CHANNELS]
    if checkpoint["input_channels"] != expected_inputs:
        raise RuntimeError("Checkpoint input-channel configuration has changed.")
    if checkpoint["output_channels"] != expected_outputs:
        raise RuntimeError("Checkpoint output-channel configuration has changed.")
    training_shape = tuple(int(value) for value in checkpoint["training_shape"])
    if len(training_shape) != 2 or any(value < 2 for value in training_shape):
        raise RuntimeError(
            f"Checkpoint training mesh {training_shape} is invalid."
        )
    model = model_from_config(checkpoint["model_config"], device)
    incompatible = model.load_state_dict(checkpoint["state_dict"], strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "FNO checkpoint tensors are incompatible: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}."
        )
    geometry = tuple(float(value) for value in checkpoint["geometry"])
    return (
        model.eval(),
        normalizer_from_state(checkpoint["normalization"]),
        geometry,
        training_shape,
    )


def _relative_l2_per_sample(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    difference = (prediction - target).flatten(start_dim=1)
    reference = target.flatten(start_dim=1)
    return torch.linalg.vector_norm(difference, dim=1) / torch.linalg.vector_norm(
        reference, dim=1
    ).clamp_min(RELATIVE_L2_EPS)


def _relative_l2_per_field(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    difference = (prediction - target).flatten(start_dim=2)
    reference = target.flatten(start_dim=2)
    return torch.linalg.vector_norm(difference, dim=2) / torch.linalg.vector_norm(
        reference, dim=2
    ).clamp_min(RELATIVE_L2_EPS)


def evaluate(
    model: FNO,
    dataset: FNOAdapter,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate mean per-case normalized relative L2."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    global_values: list[torch.Tensor] = []
    field_values: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            target = batch["y"].to(device)
            prediction = model(batch["x"].to(device))
            global_values.append(
                _relative_l2_per_sample(prediction, target).detach().cpu()
            )
            field_values.append(
                _relative_l2_per_field(prediction, target).detach().cpu()
            )
    global_tensor = torch.cat(global_values)
    fields_tensor = torch.cat(field_values)
    metrics = {"normalized_relative_l2": float(global_tensor.mean())}
    for index, spec in enumerate(OUTPUT_CHANNELS):
        metrics[f"normalized_relative_l2_{spec.label}"] = float(
            fields_tensor[:, index].mean()
        )
    return metrics


def resize_model_input(
    model_input: torch.Tensor,
    shape: tuple[int, int],
) -> torch.Tensor:
    """Resize channel-first inputs to a requested FNO evaluation mesh.

    Broadcast scalar channels remain exactly constant. This interpolation also
    makes future continuous physical input fields configuration-compatible
    with zero-shot superresolution. Evaluating directly on the requested mesh
    also keeps NeuralOperator domain padding and unpadding resolution-consistent.
    """
    if tuple(model_input.shape[-2:]) == tuple(shape):
        return model_input
    return torch_functional.interpolate(
        model_input.unsqueeze(0),
        size=shape,
        mode="bilinear",
        align_corners=True,
    )[0]


def write_history(
    path: Path,
    history: Mapping[str, list[float]],
) -> None:
    """Write per-epoch training and validation losses."""
    frame = pd.DataFrame(
        {
            "epoch": np.arange(1, len(history["train_loss"]) + 1),
            "train_loss": history["train_loss"],
            "valid_loss": history["valid_loss"],
        }
    )
    frame.to_csv(path, index=False)


def read_history(path: Path) -> dict[str, list[float]]:
    """Load and validate a previously saved loss history."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Training history not found at {path}. Train the model first."
        )
    frame = pd.read_csv(path)
    required = {"train_loss", "valid_loss"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"Training history {path} is missing columns {sorted(missing)}."
        )
    history = {
        name: frame[name].astype(float).tolist()
        for name in ("train_loss", "valid_loss")
    }
    if not history["train_loss"] or any(
        not math.isfinite(value)
        for values in history.values()
        for value in values
    ):
        raise ValueError(f"Training history {path} is empty or non-finite.")
    return history


def plot_history(
    path: Path,
    history: Mapping[str, list[float]],
) -> None:
    """Plot normalized relative-L2 histories."""
    epochs = range(1, len(history["train_loss"]) + 1)
    figure, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    axis.semilogy(epochs, history["train_loss"], label="Training")
    axis.semilogy(epochs, history["valid_loss"], label="Validation")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Normalized relative L2")
    axis.set_title("catalytic membrane reactor FNO")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_field_comparison(  # pylint: disable=too-many-arguments
    path: Path,
    truth: torch.Tensor,
    prediction: torch.Tensor,
    z: torch.Tensor,
    r: torch.Tensor,
    *,
    title: str,
) -> None:
    """Plot solver truth, FNO output, and absolute error for three fields."""
    exact = truth.detach().cpu().numpy()
    reconstructed = prediction.detach().cpu().numpy()
    error = np.abs(reconstructed - exact)
    z_mm = 1.0e3 * z.detach().cpu().numpy()
    r_mm = 1.0e3 * r.detach().cpu().numpy()
    figure, axes = plt.subplots(
        len(OUTPUT_CHANNELS),
        3,
        figsize=(13, 10),
        squeeze=False,
        constrained_layout=True,
    )
    for row, spec in enumerate(OUTPUT_CHANNELS):
        low = min(float(exact[row].min()), float(reconstructed[row].min()))
        high = max(float(exact[row].max()), float(reconstructed[row].max()))
        images = (
            axes[row, 0].pcolormesh(
                z_mm,
                r_mm,
                exact[row].T,
                shading="auto",
                vmin=low,
                vmax=high,
            ),
            axes[row, 1].pcolormesh(
                z_mm,
                r_mm,
                reconstructed[row].T,
                shading="auto",
                vmin=low,
                vmax=high,
            ),
            axes[row, 2].pcolormesh(
                z_mm,
                r_mm,
                error[row].T,
                shading="auto",
                cmap="magma",
            ),
        )
        axes[row, 0].set_ylabel(f"{spec.title}\nRadius [mm]")
        colorbar_label = spec.unit if spec.unit != "-" else "Mole fraction [-]"
        for column, column_title in enumerate(
            ("Solver truth", "FNO", "Absolute error")
        ):
            axes[row, column].set_title(column_title)
            axes[row, column].set_xlabel("Axial position [mm]")
            figure.colorbar(
                images[column],
                ax=axes[row, column],
                label=colorbar_label,
            )
    figure.suptitle(title)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def discover_mesh_files(data_dir: Path) -> list[MeshFile]:
    """Return all available ``q2d_cmr_{n_z}_{n_r}_test`` files."""
    files: list[MeshFile] = []
    for path in data_dir.glob("q2d_cmr_*_*_test.h5"):
        match = MESH_FILE_PATTERN.fullmatch(path.name)
        if match is not None:
            files.append(MeshFile(path, int(match.group(1)), int(match.group(2))))
    files.sort(key=lambda item: (item.n_z, item.n_r))
    discovered = {(item.n_z, item.n_r) for item in files}
    axial_points = {item.n_z for item in files}
    radial_points = {item.n_r for item in files}
    expected = {
        (n_z, n_r)
        for n_z in axial_points
        for n_r in radial_points
    }
    missing = sorted(expected - discovered)
    if missing:
        print(
            "Mesh benchmark files not present (continuing): "
            + ", ".join(f"{n_z}x{n_r}" for n_z, n_r in missing)
        )
    if not files:
        raise FileNotFoundError("No Q2D resolution-sweep test files were found.")
    return files


def _case_controls(metadata: Mapping[str, Any]) -> tuple[float, float]:
    params = metadata.get("params", {})
    try:
        return float(params["T0"]), float(params["sccm"])
    except KeyError as exc:
        raise KeyError("Case metadata is missing T0 or sccm.") from exc


def _controls_match(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> bool:
    first_t0, first_sccm = _case_controls(first)
    second_t0, second_sccm = _case_controls(second)
    return math.isclose(first_t0, second_t0, rel_tol=1.0e-10, abs_tol=1.0e-8) and (
        math.isclose(
            first_sccm,
            second_sccm,
            rel_tol=1.0e-10,
            abs_tol=1.0e-8,
        )
    )


def find_superresolution_pair(
    mesh_files: Sequence[MeshFile],
    normalizer: ZScoreNormalizer,
    geometry: tuple[float, float],
    training_shape: tuple[int, int],
) -> tuple[MeshFile, int, MeshFile, int]:
    """Find a training-mesh case with truth on the finest matching mesh."""
    try:
        base_file = next(
            item
            for item in mesh_files
            if (item.n_z, item.n_r) == training_shape
        )
    except StopIteration as exc:
        raise FileNotFoundError(
            "No mesh-sweep test file matches the discovered training mesh "
            f"{training_shape[0]}x{training_shape[1]}."
        ) from exc
    base_raw, base_data = make_adapter(base_file.path, normalizer, geometry)
    candidates = sorted(
        (
            item
            for item in mesh_files
            if item.mesh_points > math.prod(training_shape)
            and item.n_z >= training_shape[0]
            and item.n_r >= training_shape[1]
        ),
        key=lambda item: (item.mesh_points, item.n_r, item.n_z),
        reverse=True,
    )
    try:
        base_metadata = [
            base_data.physical_item(index)["metadata"]
            for index in range(len(base_data))
        ]
        for candidate in candidates:
            fine_raw, fine_data = make_adapter(
                candidate.path,
                normalizer,
                geometry,
            )
            try:
                for base_index, base_case in enumerate(base_metadata):
                    for fine_index in range(len(fine_data)):
                        fine_case = fine_data.physical_item(fine_index)["metadata"]
                        if _controls_match(base_case, fine_case):
                            return base_file, base_index, candidate, fine_index
            finally:
                fine_raw.close()
    finally:
        base_raw.close()
    raise RuntimeError("No matched coarse/fine case exists for superresolution.")


def make_reconstruction_figures(  # pylint: disable=too-many-locals
    model: FNO,
    normalizer: ZScoreNormalizer,
    geometry: tuple[float, float],
    mesh_files: Sequence[MeshFile],
    training_shape: tuple[int, int],
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Create matched coarse and solver-truth superresolution figures."""
    base_file, base_index, fine_file, fine_index = find_superresolution_pair(
        mesh_files,
        normalizer,
        geometry,
        training_shape,
    )
    base_raw, base_data = make_adapter(
        base_file.path,
        normalizer,
        geometry,
    )
    fine_raw, fine_data = make_adapter(fine_file.path, normalizer, geometry)
    try:
        base = base_data[base_index]
        fine = fine_data[fine_index]
        with torch.no_grad():
            coarse_prediction = model(base["x"].unsqueeze(0).to(device)).cpu()[0]
            fine_input = resize_model_input(
                base["x"],
                (fine_file.n_z, fine_file.n_r),
            )
            fine_prediction = model(fine_input.unsqueeze(0).to(device)).cpu()[0]
        coarse_truth = base_data.denormalize_output(base["y"])
        coarse_prediction = base_data.denormalize_output(coarse_prediction)
        fine_truth = fine_data.denormalize_output(fine["y"])
        fine_prediction = fine_data.denormalize_output(fine_prediction)
        base_metadata = base_data.physical_item(base_index)["metadata"]
        t0, sccm = _case_controls(base_metadata)
        plot_field_comparison(
            OUTPUT_DIR / "test_reconstructions.png",
            coarse_truth,
            coarse_prediction,
            base["z"],
            base["r"],
            title=(
                "Test reconstruction: "
                f"{training_shape[0]}x{training_shape[1]}, "
                f"$T_0$={t0:.2f} K, $Q_s$={sccm:.2f} sccm"
            ),
        )
        plot_field_comparison(
            OUTPUT_DIR / "test_superresolution.png",
            fine_truth,
            fine_prediction,
            fine["z"],
            fine["r"],
            title=(
                "2D CMR FNO superresolution: "
                f"{training_shape[0]}x{training_shape[1]} to "
                f"{fine_file.n_z}x{fine_file.n_r}"
            ),
        )
    finally:
        base_raw.close()
        fine_raw.close()
    return {
        "training_file": base_file.path.name,
        "training_shape": list(training_shape),
        "base_case_index": base_index,
        "fine_file": fine_file.path.name,
        "fine_case_index": fine_index,
        "fine_shape": [fine_file.n_z, fine_file.n_r],
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def inference_latency(
    model: FNO,
    model_input: torch.Tensor,
    *,
    device: torch.device,
) -> float:
    """Return median synchronized single-case FNO latency in seconds."""
    model_input = model_input.unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(LATENCY_WARMUPS):
            model(model_input)
        _synchronize(device)
        durations = []
        for _ in range(LATENCY_REPEATS):
            _synchronize(device)
            tic = time.perf_counter()
            model(model_input)
            _synchronize(device)
            durations.append(time.perf_counter() - tic)
    return statistics.median(durations)


def benchmark_meshes(  # pylint: disable=too-many-locals
    model: FNO,
    normalizer: ZScoreNormalizer,
    geometry: tuple[float, float],
    mesh_files: Sequence[MeshFile],
    training_shape: tuple[int, int],
    *,
    device: torch.device,
) -> pd.DataFrame:
    """Evaluate all runs in every discovered resolution-sweep file."""
    rows: list[dict[str, Any]] = []
    model.eval()
    for mesh_file in mesh_files:
        raw, dataset = make_adapter(mesh_file.path, normalizer, geometry)
        try:
            for case_index in range(len(dataset)):
                sample = dataset[case_index]
                target = sample["y"].unsqueeze(0).to(device)
                target_shape = tuple(int(value) for value in target.shape[-2:])
                if target_shape != (mesh_file.n_z, mesh_file.n_r):
                    raise ValueError(
                        f"{mesh_file.path.name} declares "
                        f"{mesh_file.n_z}x{mesh_file.n_r} but stores {target_shape}."
                    )
                training_input = resize_model_input(sample["x"], training_shape)
                prediction_input = resize_model_input(
                    training_input,
                    target_shape,
                )
                with torch.no_grad():
                    prediction = model(prediction_input.unsqueeze(0).to(device))
                global_l2 = float(
                    _relative_l2_per_sample(prediction, target).cpu()[0]
                )
                field_l2 = _relative_l2_per_field(prediction, target).cpu()[0]
                physical = dataset.physical_item(case_index)
                metadata = physical["metadata"]
                t0, sccm = _case_controls(metadata)
                row: dict[str, Any] = {
                    "dataset_file": mesh_file.path.name,
                    "case_index": case_index,
                    "n_z": mesh_file.n_z,
                    "n_r": mesh_file.n_r,
                    "mesh_points": mesh_file.mesh_points,
                    "T0_K": t0,
                    "SCCM": sccm,
                    "normalized_relative_l2": global_l2,
                    "solver_wall_time_s": float(metadata["wall_time"]),
                    "fno_wall_time_s": inference_latency(
                        model,
                        prediction_input,
                        device=device,
                    ),
                }
                for field_index, spec in enumerate(OUTPUT_CHANNELS):
                    row[f"normalized_relative_l2_{spec.label}"] = float(
                        field_l2[field_index]
                    )
                rows.append(row)
        finally:
            raw.close()
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("The mesh benchmark produced no rows.")
    return frame.sort_values(
        ["mesh_points", "n_z", "n_r", "case_index"],
        ignore_index=True,
    )


def _grouped_summary(
    frame: pd.DataFrame,
    column: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grouped = frame.groupby("mesh_points", sort=True)[column]
    return (
        grouped.mean().index.to_numpy(dtype=float),
        grouped.mean().to_numpy(dtype=float),
        grouped.min().to_numpy(dtype=float),
        grouped.max().to_numpy(dtype=float),
    )


def _mark_training_mesh(
    axis: plt.Axes,
    training_shape: tuple[int, int],
) -> None:
    """Mark the mesh size used to train the FNO on a benchmark axis."""
    training_points = math.prod(training_shape)
    axis.axvline(
        training_points,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=(
            "FNO training mesh "
            f"({training_shape[0]}x{training_shape[1]} = {training_points})"
        ),
    )


def plot_mesh_l2(
    path: Path,
    frame: pd.DataFrame,
    training_shape: tuple[int, int],
) -> None:
    """Plot normalized relative L2 against total mesh points."""
    x, mean, minimum, maximum = _grouped_summary(
        frame,
        "normalized_relative_l2",
    )
    figure, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    axis.plot(x, mean, marker="o", label="FNO")
    axis.fill_between(x, minimum, maximum, alpha=0.2, label="Min-max")
    axis.set_xlabel("Number of mesh points")
    axis.set_ylabel("Normalized relative L2")
    axis.set_yscale("log")
    _mark_training_mesh(axis, training_shape)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_mesh_wall_time(
    path: Path,
    frame: pd.DataFrame,
    training_shape: tuple[int, int],
    *,
    ray_tuning_seconds: float,
    final_training_seconds: float,
) -> None:
    """Plot inference, solver, tuning, and final-training wall times."""
    figure, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for column, label in (
        ("solver_wall_time_s", "1 evaluation of numerical solver"),
        ("fno_wall_time_s", "1 evaluation of FNO"),
    ):
        x, mean, minimum, maximum = _grouped_summary(frame, column)
        axis.plot(x, mean, marker="o", label=label)
        axis.fill_between(x, minimum, maximum, alpha=0.15)
    for seconds, label, color, linestyle in (
        (
            ray_tuning_seconds,
            "Finding the best FNO model",
            "tab:purple",
            "-.",
        ),
        (
            final_training_seconds,
            "FNO training",
            "tab:green",
            ":",
        ),
    ):
        if not math.isfinite(seconds) or seconds <= 0.0:
            raise ValueError(f"{label} wall time must be positive and finite.")
        axis.axhline(
            seconds,
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
            label=f"{label} ({seconds:.2f} s)",
        )
    axis.set_xlabel("Number of mesh points")
    axis.set_ylabel("Wall time [s]")
    axis.set_yscale("log")
    _mark_training_mesh(axis, training_shape)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def load_json_mapping(path: Path, description: str) -> dict[str, Any]:
    """Load a required JSON object."""
    if not path.is_file():
        raise FileNotFoundError(
            f"{description} not found at {path}. Run the full workflow first."
        )
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"{description} at {path} must contain a JSON object.")
    return value


def train_best_config(  # pylint: disable=too-many-arguments
    best_config: Mapping[str, Any],
    normalizer: ZScoreNormalizer,
    geometry: tuple[float, float],
    training_shape: tuple[int, int],
    *,
    device: torch.device,
) -> tuple[dict[str, list[float]], float]:
    """Train and save one model using an already selected configuration."""
    train_raw, train_data = make_adapter(
        DATA_DIR / f"{FILE_STEM}_train.h5",
        normalizer,
        geometry,
    )
    valid_raw, valid_data = make_adapter(
        DATA_DIR / f"{FILE_STEM}_valid.h5",
        normalizer,
        geometry,
    )
    try:
        stored_training_shape = adapter_spatial_shape(
            train_data,
            f"{FILE_STEM}_train.h5",
        )
        valid_shape = adapter_spatial_shape(
            valid_data,
            f"{FILE_STEM}_valid.h5",
        )
        if stored_training_shape != training_shape:
            raise ValueError(
                f"Training mesh changed from {training_shape} to "
                f"{stored_training_shape}."
            )
        if valid_shape != training_shape:
            raise ValueError(
                f"Validation mesh {valid_shape} differs from the discovered "
                f"training mesh {training_shape}."
            )
        model, history, training_seconds = train_model(
            best_config,
            train_data,
            valid_data,
            epochs=FINAL_EPOCHS,
            device=device,
            print_epochs=True,
        )
    finally:
        train_raw.close()
        valid_raw.close()
    print(f"Final FNO training time: {training_seconds:.6f} s")

    save_checkpoint(
        OUTPUT_DIR / "fno.pt",
        model,
        best_config,
        normalizer,
        geometry,
        training_shape,
    )
    write_history(OUTPUT_DIR / "history.csv", history)
    plot_history(OUTPUT_DIR / "training_validation_loss.png", history)
    return history, training_seconds


def use_saved_model(  # pylint: disable=too-many-locals
    device: torch.device,
    *,
    calculate_metrics: bool,
    ray_tuning_seconds: float,
    final_training_seconds: float,
) -> dict[str, Any]:
    """Load the checkpoint and regenerate Q2D evaluations and figures."""
    checkpoint_path = OUTPUT_DIR / "fno.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Saved model not found at {checkpoint_path}. Train it first."
        )
    model, normalizer, geometry, training_shape = load_checkpoint(
        checkpoint_path,
        device,
    )

    training_raw, training_data = make_adapter(
        DATA_DIR / f"{FILE_STEM}_train.h5",
        normalizer,
        geometry,
    )
    try:
        stored_training_shape = adapter_spatial_shape(
            training_data,
            f"{FILE_STEM}_train.h5",
        )
    finally:
        training_raw.close()
    if stored_training_shape != training_shape:
        raise RuntimeError(
            f"Checkpoint training mesh {training_shape} differs from the "
            f"current training dataset mesh {stored_training_shape}."
        )

    test_raw, test_data = make_adapter(
        DATA_DIR / f"{FILE_STEM}_test.h5",
        normalizer,
        geometry,
    )
    try:
        test_metrics = (
            evaluate(
                model,
                test_data,
                batch_size=EVALUATION_BATCH_SIZE,
                device=device,
            )
            if calculate_metrics
            else {}
        )
    finally:
        test_raw.close()

    plot_history(
        OUTPUT_DIR / "training_validation_loss.png",
        read_history(OUTPUT_DIR / "history.csv"),
    )

    mesh_files = discover_mesh_files(DATA_DIR)
    reconstruction = make_reconstruction_figures(
        model,
        normalizer,
        geometry,
        mesh_files,
        training_shape,
        device=device,
    )
    benchmark_path = OUTPUT_DIR / "mesh_benchmark.csv"
    if calculate_metrics:
        benchmark = benchmark_meshes(
            model,
            normalizer,
            geometry,
            mesh_files,
            training_shape,
            device=device,
        )
        benchmark.to_csv(benchmark_path, index=False)
    else:
        if not benchmark_path.is_file():
            raise FileNotFoundError(
                f"Mesh benchmark not found at {benchmark_path}. "
                "Run the full workflow first."
            )
        benchmark = pd.read_csv(benchmark_path)
    if calculate_metrics:
        print(benchmark.to_string(index=False))
    plot_mesh_l2(
        OUTPUT_DIR / "mesh_l2_vs_points.png",
        benchmark,
        training_shape,
    )
    plot_mesh_wall_time(
        OUTPUT_DIR / "mesh_wall_time_vs_points.png",
        benchmark,
        training_shape,
        ray_tuning_seconds=ray_tuning_seconds,
        final_training_seconds=final_training_seconds,
    )
    return {
        **test_metrics,
        "parameters": count_parameters(model),
        "training_shape": list(training_shape),
        "training_mesh_points": math.prod(training_shape),
        "mesh_benchmark_rows": int(len(benchmark)),
        "reconstruction": reconstruction,
    }


def main() -> None:  # pylint: disable=too-many-locals
    """Run the selected tuning, training, or saved-model plotting workflow."""
    if TRAIN_BEST_CONFIG_ONLY and PLOT_SAVED_MODEL_ONLY:
        raise ValueError(
            "TRAIN_BEST_CONFIG_ONLY and PLOT_SAVED_MODEL_ONLY cannot both be true."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics_path = OUTPUT_DIR / "metrics.json"
    if PLOT_SAVED_MODEL_ONLY:
        saved_metrics = load_json_mapping(metrics_path, "Saved metrics")
        use_saved_model(
            device,
            calculate_metrics=False,
            ray_tuning_seconds=float(saved_metrics["ray_tuning_seconds"]),
            final_training_seconds=float(
                saved_metrics["final_training_seconds"]
            ),
        )
        print(f"Plots written to {OUTPUT_DIR}")
        return

    print("Fitting scalar training-only normalization statistics ...")
    normalizer, geometry, training_shape = fit_normalizer(
        DATA_DIR / f"{FILE_STEM}_train.h5"
    )
    print(
        "Discovered FNO training mesh: "
        f"{training_shape[0]}x{training_shape[1]} "
        f"({math.prod(training_shape)} points)"
    )
    best_config_path = OUTPUT_DIR / "best_config.json"
    if TRAIN_BEST_CONFIG_ONLY:
        best_config = load_json_mapping(best_config_path, "Best configuration")
        history, training_seconds = train_best_config(
            best_config,
            normalizer,
            geometry,
            training_shape,
            device=device,
        )
        saved_metrics = (
            load_json_mapping(metrics_path, "Saved metrics")
            if metrics_path.is_file()
            else {}
        )
        saved_metrics.update(
            {
                "parameters": count_parameters(
                    load_checkpoint(OUTPUT_DIR / "fno.pt", device)[0]
                ),
                "final_training_seconds": training_seconds,
                "best_epoch": int(np.argmin(history["valid_loss"]) + 1),
                "best_validation_loss": float(min(history["valid_loss"])),
                "domain_padding": float(
                    best_config.get("domain_padding", 0.0)
                ),
                "training_shape": list(training_shape),
                "training_mesh_points": math.prod(training_shape),
            }
        )
        with metrics_path.open("w", encoding="utf-8") as file:
            json.dump(saved_metrics, file, indent=2)
        print(f"Model and loss history written to {OUTPUT_DIR}")
        return

    (OUTPUT_DIR / "ray_results").mkdir(parents=True, exist_ok=True)
    RAY_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    worker_pythonpath = os.pathsep.join(
        filter(
            None,
            (
                str((ROOT / "scripts").resolve()),
                str((ROOT / "src").resolve()),
                os.environ.get("PYTHONPATH"),
            ),
        )
    )
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        _temp_dir=str(RAY_TEMP_DIR.resolve()),
        runtime_env={"env_vars": {"PYTHONPATH": worker_pythonpath}},
    )
    tune_tic = time.perf_counter()
    try:
        best_config = tune_hyperparameters(
            DATA_DIR,
            OUTPUT_DIR,
            normalizer,
            geometry,
        )
    finally:
        ray.shutdown()
    tune_seconds = time.perf_counter() - tune_tic
    with best_config_path.open("w", encoding="utf-8") as file:
        json.dump(best_config, file, indent=2)

    history, training_seconds = train_best_config(
        best_config,
        normalizer,
        geometry,
        training_shape,
        device=device,
    )
    evaluation = use_saved_model(
        device,
        calculate_metrics=True,
        ray_tuning_seconds=tune_seconds,
        final_training_seconds=training_seconds,
    )

    metrics = {
        **evaluation,
        "final_training_seconds": training_seconds,
        "ray_tuning_seconds": tune_seconds,
        "best_epoch": int(np.argmin(history["valid_loss"]) + 1),
        "best_validation_loss": float(min(history["valid_loss"])),
        "domain_padding": float(best_config.get("domain_padding", 0.0)),
    }
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Results written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
