"""Shared datasets and protocols for benchmark model backends."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


EpochReporter = Callable[
    [Mapping[str, float | int], torch.nn.Module, Mapping[str, Any]], None
]


@dataclass
class TrainingOutcome:
    """Model-independent training result."""

    history: dict[str, list[float]]
    best_valid_loss: float
    best_epoch: int
    parameter_count: int
    extra: dict[str, Any] = field(default_factory=dict)


class ModelBackend(Protocol):
    """Interface implemented by all benchmark model families."""

    name: str

    def fit(
        self,
        train: Dataset,
        validation: Dataset,
        *,
        config: Mapping[str, Any],
        epochs: int,
        patience: int,
        seed: int,
        device: torch.device,
        reporter: EpochReporter | None = None,
    ) -> TrainingOutcome: ...

    def predict(
        self,
        dataset: Dataset,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor: ...

    def checkpoint_state(self) -> dict[str, Any]: ...

    def load_checkpoint_state(
        self, state: Mapping[str, Any], *, device: torch.device
    ) -> None: ...

    def predict_tensor(self, inputs: Any, *, device: torch.device) -> torch.Tensor: ...


class ResampledOperatorDataset(Dataset):
    """Optionally place channel-first operator samples on one uniform grid."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        coordinate_names: Sequence[str],
        output_labels: Sequence[str],
        resample_shape: Sequence[int] | None = None,
    ) -> None:
        self.dataset = dataset
        self.coordinate_names = tuple(coordinate_names)
        self.output_labels = tuple(output_labels)
        self.resample_shape = (
            None if resample_shape is None else tuple(int(v) for v in resample_shape)
        )
        if self.resample_shape is not None and (
            len(self.resample_shape) != len(self.coordinate_names)
            or len(self.resample_shape) not in {1, 2}
        ):
            raise ValueError("Resampling supports one- and two-dimensional grids.")

    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def _interpolate(
        value: torch.Tensor,
        shape: tuple[int, ...],
    ) -> torch.Tensor:
        if tuple(value.shape[1:]) == shape:
            return value
        mode = "linear" if len(shape) == 1 else "bilinear"
        result = F.interpolate(
            value.unsqueeze(0),
            size=shape,
            mode=mode,
            align_corners=True,
        )
        return result.squeeze(0)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(self.dataset[index])
        if self.resample_shape is None:
            return item
        item["x"] = self._interpolate(item["x"], self.resample_shape)
        item["y"] = self._interpolate(item["y"], self.resample_shape)
        for name, count in zip(self.coordinate_names, self.resample_shape):
            coordinate = item[name].reshape(-1)
            item[name] = torch.linspace(
                coordinate[0],
                coordinate[-1],
                count,
                dtype=coordinate.dtype,
                device=coordinate.device,
            )
        return item

    def denormalize_output(self, output: torch.Tensor) -> torch.Tensor:
        method = getattr(self.dataset, "denormalize_output")
        return method(output)

    def physical_item(self, index: int) -> dict[str, Any]:
        normalized = self[index]
        physical = {
            "x": normalized["x"],
            "y": self.denormalize_output(normalized["y"]),
            **{
                name: normalized[name]
                for name in self.coordinate_names
            },
        }
        original = getattr(self.dataset, "physical_item", None)
        if original is not None:
            physical["metadata"] = original(index).get("metadata", {})
        return physical


class GridDeepONetDataset(Dataset):
    """Expose common operator grids as lazy DeepONet branch/trunk samples."""

    format = "cartesian_product"

    def __init__(
        self,
        dataset: ResampledOperatorDataset,
        *,
        coordinate_names: Sequence[str],
        output_labels: Sequence[str],
    ) -> None:
        self.dataset = dataset
        self.coordinate_names = tuple(coordinate_names)
        self.output_labels = tuple(output_labels)
        if not self.coordinate_names:
            raise ValueError("DeepONet datasets need at least one coordinate.")

    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def _scaled_coordinate(values: torch.Tensor) -> torch.Tensor:
        values = values.reshape(-1)
        span = (values[-1] - values[0]).abs().clamp_min(
            torch.finfo(values.dtype).eps
        )
        return (values - values[0]) / span

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.dataset[index]
        model_input = item["x"]
        spatial_ndim = len(self.coordinate_names)
        if model_input.ndim != spatial_ndim + 1:
            raise ValueError(
                "Operator inputs must be channel-first with one axis per coordinate."
            )
        origin = (slice(None),) + (0,) * spatial_ndim
        branch = model_input[origin]
        scaled = [
            self._scaled_coordinate(item[name])
            for name in self.coordinate_names
        ]
        mesh = torch.meshgrid(*scaled, indexing="ij")
        trunk = torch.stack(mesh, dim=-1).reshape(-1, spatial_ndim)
        target = item["y"].movedim(0, -1)
        spatial_shape = tuple(int(value) for value in target.shape[:-1])
        target = target.reshape(-1, target.shape[-1])
        physical_mesh = torch.meshgrid(
            *[item[name].reshape(-1) for name in self.coordinate_names],
            indexing="ij",
        )
        coordinate = torch.stack(physical_mesh, dim=-1).reshape(
            -1, spatial_ndim
        )
        return {
            "branch": branch,
            "trunk": trunk,
            "target": target,
            "coordinate": coordinate,
            "coordinates": coordinate,
            "labels": self.output_labels,
            "spatial_shape": spatial_shape,
        }

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        return tuple(self[0]["spatial_shape"])


def count_parameters(model: torch.nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def select_device(requested: str, gpus_per_trial: float | None = None) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    allow_cuda = gpus_per_trial is None or gpus_per_trial > 0
    return torch.device(
        "cuda" if allow_cuda and torch.cuda.is_available() else "cpu"
    )


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalized_macro_rmse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    channel_axis: int = 1,
) -> torch.Tensor:
    """Mean per-channel RMSE for already normalized tensors."""

    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes differ.")
    axis = channel_axis % target.ndim
    reduction = tuple(index for index in range(target.ndim) if index != axis)
    return (prediction - target).square().mean(dim=reduction).sqrt().mean()


__all__ = [
    "EpochReporter",
    "GridDeepONetDataset",
    "ModelBackend",
    "ResampledOperatorDataset",
    "TrainingOutcome",
    "count_parameters",
    "normalized_macro_rmse",
    "seed_everything",
    "select_device",
]
