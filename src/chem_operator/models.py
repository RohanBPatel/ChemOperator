"""Framework adapters and shared neural-operator benchmark utilities.

The adapters in this module are intentionally separate from HDF5 ingestion:
``CanteraDataset`` returns structured samples, ``DataProcessor`` transforms
their fields, and an adapter supplies the shape expected by a model framework.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import partial
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from chem_operator.normalization import ZScoreNormalizer
from chem_operator.dataset_processing import DataProcessor


ArrayTransform = Callable[[np.ndarray], np.ndarray]
DeepXDEFormat = Literal["cartesian_product", "pointwise"]
ChannelAxis = Literal["first", "last"]
FNOChannelSource = Literal["parameter", "constant", "field", "species"]


@dataclass(frozen=True)
class FNOChannel:
    """Describe one input or output channel for :class:`FNOAdapter`.

    Parameters
    ----------
    label:
        Name used by the normalizer and model interface.
    source:
        ``"parameter"`` reads ``sample["metadata"]["params"][key]``;
        ``"constant"`` reads ``sample["constant_inputs"][key]``;
        ``"field"`` reconstructs a complete scalar field from the input and
        output windows; and ``"species"`` selects one species from a grouped
        field using ``sample["metadata"]["field_species"]``.
    key:
        Parameter, constant, or field name in the source sample.
    species:
        Species name required when ``source="species"``.
    display_name, unit:
        Optional presentation metadata for downstream plots and reports.

    Notes
    -----
    A converged solution field should only be configured as an input when it
    is genuinely available at inference time. Otherwise it leaks target data
    into the model input.
    """

    label: str
    source: FNOChannelSource
    key: str
    species: str | None = None
    display_name: str | None = None
    unit: str = "-"

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("FNO channel labels must be non-empty.")
        if not self.key:
            raise ValueError(f"FNO channel {self.label!r} needs a source key.")
        if self.source == "species" and not self.species:
            raise ValueError(f"Species channel {self.label!r} needs a species.")
        if self.source != "species" and self.species is not None:
            raise ValueError(
                f"Only species channels may set species; got {self.label!r}."
            )

    @property
    def title(self) -> str:
        """Return the preferred human-readable channel name."""
        return self.display_name or self.label


@dataclass(frozen=True)
class OperatorArrays:
    """Materialized operator-learning arrays and reconstruction context."""

    branch: np.ndarray
    trunk: np.ndarray
    targets: np.ndarray
    coordinates: tuple[np.ndarray, ...]
    model_inputs: np.ndarray
    labels: tuple[str, ...]
    constant_labels: tuple[str, ...]
    metadata: tuple[Mapping[str, Any], ...]
    trajectory_slices: tuple[slice, ...]

    @property
    def n_trajectories(self) -> int:
        return len(self.coordinates)


@dataclass(frozen=True)
class _Trajectory:
    branch: np.ndarray
    target: np.ndarray
    coordinate: np.ndarray
    model_input: np.ndarray
    labels: tuple[str, ...]
    constant_labels: tuple[str, ...]
    metadata: Mapping[str, Any]


class DeepXDEAdapter(Dataset):
    """Adapt processed trajectories to DeepXDE operator-data layouts.

    Parameters
    ----------
    dataset:
        A map-style dataset, normally a ``CanteraDataset`` configured with
        ``task="operator_cartesian"`` and one input step.
    processor:
        A state-target ``DataProcessor``. Its packed final input state becomes
        the branch input. Selected constants may be appended to that branch.
    format:
        ``"cartesian_product"`` retains one branch row per trajectory and a
        shared trunk grid. ``"pointwise"`` repeats each branch row at every
        coordinate and permits trajectories with different grids.
    resample_points:
        If set, interpolate every trajectory onto a shared grid. Relative
        coordinates are useful for adaptively stepped trajectories.
    """

    def __init__(
        self,
        dataset: Dataset,
        processor: DataProcessor,
        *,
        format: DeepXDEFormat = "cartesian_product",
        coordinate_name: str | None = None,
        include_constants: bool = False,
        include_initial: bool = True,
        resample_points: int | None = None,
        coordinate_mode: Literal["physical", "relative"] = "physical",
        indices: Sequence[int] | None = None,
        max_trajectories: int | None = None,
        dtype: np.dtype | type = np.float32,
    ):
        if format not in {"cartesian_product", "pointwise"}:
            raise ValueError(
                "format must be 'cartesian_product' or 'pointwise'."
            )
        if coordinate_mode not in {"physical", "relative"}:
            raise ValueError("coordinate_mode must be 'physical' or 'relative'.")
        if resample_points is not None and resample_points < 2:
            raise ValueError("resample_points must be at least two.")
        if max_trajectories is not None and max_trajectories < 1:
            raise ValueError("max_trajectories must be positive.")
        if processor.target_transform.is_delta:
            raise ValueError(
                "DeepXDEAdapter requires state targets; configure "
                "TargetTransformConfig(mode='state')."
            )

        self.dataset = dataset
        self.processor = processor
        self.format = format
        self.coordinate_name = coordinate_name
        self.include_constants = include_constants
        self.include_initial = include_initial
        self.resample_points = resample_points
        self.coordinate_mode = coordinate_mode
        if indices is None:
            selected = tuple(range(len(dataset)))
        else:
            selected = tuple(indices)
        if max_trajectories is not None:
            selected = selected[:max_trajectories]
        if not selected:
            raise ValueError("The adapter received no trajectory indices.")
        self.indices = selected
        self.dtype = np.dtype(dtype)
        self._arrays: OperatorArrays | None = None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> dict[str, Any]:
        """Load and process exactly one trajectory for a ``DataLoader``."""

        item = self._load_trajectory(self.indices[position])
        if self.resample_points is not None:
            resampled, trunk = self._resample([item])
            item = resampled[0]
        else:
            trunk_values = item.coordinate
            if self.coordinate_mode == "relative":
                span = item.coordinate[-1] - item.coordinate[0]
                if span <= 0:
                    raise ValueError("Cannot normalize a zero-width coordinate.")
                trunk_values = (item.coordinate - item.coordinate[0]) / span
            trunk = trunk_values.reshape(-1, 1)
        return {
            "branch": torch.from_numpy(item.branch),
            "trunk": torch.from_numpy(
                np.asarray(trunk, dtype=self.dtype)
            ),
            "target": torch.from_numpy(item.target),
            "coordinate": torch.from_numpy(item.coordinate),
            "labels": item.labels,
        }

    @staticmethod
    def _numpy(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy()

    def _coordinate_key(self, sample: Mapping[str, Any]) -> str:
        if self.coordinate_name is not None:
            return self.coordinate_name
        metadata = sample.get("metadata", {})
        name = metadata.get("coordinate_name")
        if name is None:
            coordinates = sample.get("output_coordinates", {})
            if len(coordinates) != 1:
                raise KeyError(
                    "coordinate_name is required when a sample has multiple "
                    "output coordinates."
                )
            return next(iter(coordinates))
        return str(name)

    def _load_trajectory(self, index: int) -> _Trajectory:
        processed = self.processor(self.dataset[index])
        packer = self.processor.field_packer
        x = self._numpy(packer.to_channel_last(processed["x"]))
        y = self._numpy(packer.to_channel_last(processed["y"]))
        constants = self._numpy(processed["constants"]).reshape(-1)
        branch = x[-1].reshape(-1)
        if self.include_constants and constants.size:
            branch = np.concatenate((branch, constants))

        coordinate_name = self._coordinate_key(processed)
        output_coordinate = self._numpy(
            processed["output_coordinates"][coordinate_name]
        ).reshape(-1)
        target = y
        coordinate = output_coordinate
        if self.include_initial:
            input_coordinate = self._numpy(
                processed["input_coordinates"][coordinate_name]
            ).reshape(-1)
            coordinate = np.concatenate((input_coordinate[-1:], coordinate))
            target = np.concatenate((x[-1:], target), axis=0)

        if coordinate.shape[0] != target.shape[0]:
            raise ValueError(
                f"Coordinate {coordinate_name!r} has {coordinate.shape[0]} "
                f"points but the target has {target.shape[0]}."
            )
        if np.any(np.diff(coordinate) <= 0):
            raise ValueError(
                f"Coordinate {coordinate_name!r} must be strictly increasing."
            )

        return _Trajectory(
            branch=branch.astype(self.dtype, copy=False),
            target=target.astype(self.dtype, copy=False),
            coordinate=coordinate.astype(self.dtype, copy=False),
            model_input=x.astype(self.dtype, copy=False),
            labels=tuple(processed["labels"]["y"]),
            constant_labels=tuple(processed["labels"]["constants"]),
            metadata=processed.get("metadata", {}),
        )

    @staticmethod
    def _interpolate(
        source_coordinate: np.ndarray,
        values: np.ndarray,
        target_coordinate: np.ndarray,
    ) -> np.ndarray:
        columns = [
            np.interp(target_coordinate, source_coordinate, values[:, channel])
            for channel in range(values.shape[-1])
        ]
        return np.stack(columns, axis=-1).astype(values.dtype, copy=False)

    def _resample(
        self, trajectories: Sequence[_Trajectory]
    ) -> tuple[list[_Trajectory], np.ndarray]:
        assert self.resample_points is not None
        if self.coordinate_mode == "relative":
            common = np.linspace(0.0, 1.0, self.resample_points, dtype=self.dtype)
        else:
            low = max(float(item.coordinate[0]) for item in trajectories)
            high = min(float(item.coordinate[-1]) for item in trajectories)
            if high <= low:
                raise ValueError("Trajectory coordinate ranges do not overlap.")
            common = np.linspace(low, high, self.resample_points, dtype=self.dtype)

        result: list[_Trajectory] = []
        for item in trajectories:
            if self.coordinate_mode == "relative":
                start = float(item.coordinate[0])
                span = float(item.coordinate[-1] - item.coordinate[0])
                if span <= 0:
                    raise ValueError("Cannot normalize a zero-width coordinate.")
                source = (item.coordinate - start) / span
                plot_coordinate = start + common * span
            else:
                source = item.coordinate
                plot_coordinate = common
            target = self._interpolate(source, item.target, common)
            result.append(
                _Trajectory(
                    branch=item.branch,
                    target=target,
                    coordinate=plot_coordinate.astype(self.dtype, copy=False),
                    model_input=item.model_input,
                    labels=item.labels,
                    constant_labels=item.constant_labels,
                    metadata=item.metadata,
                )
            )
        return result, common.reshape(-1, 1)

    def _materialize(self) -> OperatorArrays:
        trajectories = [self._load_trajectory(index) for index in self.indices]
        labels = trajectories[0].labels
        constant_labels = trajectories[0].constant_labels
        branch_width = trajectories[0].branch.size
        target_width = trajectories[0].target.shape[-1]
        for item in trajectories[1:]:
            if item.labels != labels or item.constant_labels != constant_labels:
                raise ValueError("Field labels vary between trajectories.")
            if (
                item.branch.size != branch_width
                or item.target.shape[-1] != target_width
            ):
                raise ValueError("Packed channel widths vary between trajectories.")

        if self.resample_points is not None:
            trajectories, common_trunk = self._resample(trajectories)
        else:
            common_trunk = trajectories[0].coordinate.reshape(-1, 1)

        branches = np.stack([item.branch for item in trajectories])
        model_inputs = np.stack([item.model_input for item in trajectories])
        coordinates = tuple(item.coordinate for item in trajectories)
        metadata = tuple(item.metadata for item in trajectories)

        if self.format == "cartesian_product":
            for item in trajectories[1:]:
                candidate = item.coordinate.reshape(-1, 1)
                if self.resample_points is None and (
                    candidate.shape != common_trunk.shape
                    or not np.allclose(candidate, common_trunk, rtol=1e-5, atol=1e-8)
                ):
                    raise ValueError(
                        "Cartesian-product data requires a shared coordinate "
                        "grid. Set resample_points or use format='pointwise'."
                    )
            targets = np.stack([item.target for item in trajectories])
            slices = tuple(
                slice(index * targets.shape[1], (index + 1) * targets.shape[1])
                for index in range(len(trajectories))
            )
            return OperatorArrays(
                branch=branches,
                trunk=common_trunk.astype(self.dtype, copy=False),
                targets=targets,
                coordinates=coordinates,
                model_inputs=model_inputs,
                labels=labels,
                constant_labels=constant_labels,
                metadata=metadata,
                trajectory_slices=slices,
            )

        repeated_branch: list[np.ndarray] = []
        trunks: list[np.ndarray] = []
        targets_list: list[np.ndarray] = []
        slices_list: list[slice] = []
        start = 0
        for item in trajectories:
            count = item.target.shape[0]
            repeated_branch.append(np.repeat(item.branch[None, :], count, axis=0))
            coordinate = item.coordinate
            if self.coordinate_mode == "relative" and self.resample_points is None:
                coordinate = (coordinate - coordinate[0]) / (
                    coordinate[-1] - coordinate[0]
                )
            trunks.append(coordinate.reshape(-1, 1))
            targets_list.append(item.target)
            slices_list.append(slice(start, start + count))
            start += count
        return OperatorArrays(
            branch=np.concatenate(repeated_branch).astype(self.dtype, copy=False),
            trunk=np.concatenate(trunks).astype(self.dtype, copy=False),
            targets=np.concatenate(targets_list).astype(self.dtype, copy=False),
            coordinates=coordinates,
            model_inputs=model_inputs,
            labels=labels,
            constant_labels=constant_labels,
            metadata=metadata,
            trajectory_slices=tuple(slices_list),
        )

    @property
    def arrays(self) -> OperatorArrays:
        if self._arrays is None:
            self._arrays = self._materialize()
        return self._arrays

    def to_deepxde_data(
        self,
        validation: "DeepXDEAdapter",
        *,
        train_targets: np.ndarray | None = None,
        validation_targets: np.ndarray | None = None,
        trunk_transform: ArrayTransform | None = None,
    ):
        """Create a DeepXDE ``Triple`` or ``TripleCartesianProd`` dataset."""

        if validation.format != self.format:
            raise ValueError("Training and validation adapters must share a format.")
        import deepxde as dde

        train = self.arrays
        valid = validation.arrays
        train_y = train.targets if train_targets is None else train_targets
        valid_y = valid.targets if validation_targets is None else validation_targets
        train_trunk = train.trunk
        valid_trunk = valid.trunk
        if trunk_transform is not None:
            train_trunk = trunk_transform(train_trunk)
            valid_trunk = trunk_transform(valid_trunk)
        train_x = (train.branch, train_trunk.astype(self.dtype, copy=False))
        valid_x = (valid.branch, valid_trunk.astype(self.dtype, copy=False))
        if self.format == "cartesian_product":
            return dde.data.TripleCartesianProd(train_x, train_y, valid_x, valid_y)
        return dde.data.Triple(train_x, train_y, valid_x, valid_y)


class NeuralOperatorAdapter(Dataset):
    """Return processed tensors with a neural-operator channel convention."""

    def __init__(
        self,
        dataset: Dataset,
        processor: DataProcessor,
        *,
        channel_axis: ChannelAxis = "first",
        append_constants: bool = False,
        spatial_ndim: int | None = None,
    ):
        if channel_axis not in {"first", "last"}:
            raise ValueError("channel_axis must be 'first' or 'last'.")
        if spatial_ndim is not None and spatial_ndim < 1:
            raise ValueError("spatial_ndim must be positive when supplied.")
        self.dataset = dataset
        self.processor = processor
        self.channel_axis = channel_axis
        self.append_constants = append_constants
        self.spatial_ndim = spatial_ndim

    def __len__(self) -> int:
        return len(self.dataset)

    def _axis(self, tensor: torch.Tensor) -> torch.Tensor:
        channel_last = self.processor.field_packer.to_channel_last(tensor)
        if self.channel_axis == "first":
            result = channel_last.movedim(-1, 0)
        else:
            result = channel_last
        if self.spatial_ndim is not None:
            current_spatial_ndim = result.ndim - 1
            while current_spatial_ndim < self.spatial_ndim:
                axis = -1 if self.channel_axis == "first" else -2
                result = result.unsqueeze(axis)
                current_spatial_ndim += 1
        return result

    def _with_constants(
        self, tensor: torch.Tensor, constants: torch.Tensor
    ) -> torch.Tensor:
        if not self.append_constants or constants.numel() == 0:
            return tensor
        if self.channel_axis == "first":
            shape = (constants.numel(),) + (1,) * (tensor.ndim - 1)
            expanded = constants.reshape(shape).expand(
                (constants.numel(),) + tuple(tensor.shape[1:])
            )
            return torch.cat((tensor, expanded), dim=0)
        shape = (1,) * (tensor.ndim - 1) + (constants.numel(),)
        expanded = constants.reshape(shape).expand(
            tuple(tensor.shape[:-1]) + (constants.numel(),)
        )
        return torch.cat((tensor, expanded), dim=-1)

    def __getitem__(self, index: int) -> dict[str, Any]:
        processed = self.processor(self.dataset[index])
        x = self._with_constants(
            self._axis(processed["x"]), processed["constants"]
        )
        y = self._axis(processed["y"])
        return {**processed, "x": x, "y": y}


class FNOAdapter(Dataset):
    """Adapt complete two-dimensional fields for NeuralOperator FNOs.

    The adapter supports transient ``(time, space)`` trajectories and steady
    ``(axial, radial)`` fields through configurable :class:`FNOChannel`
    definitions. Scalar parameters and constants are spatially broadcast;
    scalar fields and selected species retain their complete two-dimensional
    grids.

    The legacy ``field_names`` / ``constant_names`` interface remains a
    shorthand for constant inputs, scalar-field outputs, and coordinates
    ``(time_coordinate, space_coordinate)``.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        dataset: Dataset,
        normalizer: ZScoreNormalizer,
        *,
        input_channels: Sequence[FNOChannel] | None = None,
        output_channels: Sequence[FNOChannel] | None = None,
        coordinate_names: tuple[str, str] | None = None,
        field_names: Sequence[str] | None = None,
        constant_names: Sequence[str] | None = None,
        time_coordinate: str = "t",
        space_coordinate: str = "r",
        max_trajectories: int | None = None,
    ) -> None:
        configured_interface = (
            input_channels is not None
            or output_channels is not None
            or coordinate_names is not None
        )
        legacy_interface = field_names is not None or constant_names is not None
        if configured_interface and legacy_interface:
            raise ValueError(
                "Use either input_channels/output_channels/coordinate_names or "
                "the legacy field_names/constant_names interface, not both."
            )
        if configured_interface:
            if input_channels is None or output_channels is None:
                raise ValueError(
                    "input_channels and output_channels must be supplied together."
                )
            resolved_inputs = tuple(input_channels)
            resolved_outputs = tuple(output_channels)
            resolved_coordinates = coordinate_names or (
                time_coordinate,
                space_coordinate,
            )
        else:
            if not field_names:
                raise ValueError("field_names must contain at least one field.")
            if not constant_names:
                raise ValueError("constant_names must contain at least one constant.")
            resolved_inputs = tuple(
                FNOChannel(name, "constant", name) for name in constant_names
            )
            resolved_outputs = tuple(
                FNOChannel(name, "field", name) for name in field_names
            )
            resolved_coordinates = (time_coordinate, space_coordinate)

        if not resolved_inputs:
            raise ValueError("input_channels must contain at least one channel.")
        if not resolved_outputs:
            raise ValueError("output_channels must contain at least one channel.")
        if len(resolved_coordinates) != 2:
            raise ValueError("coordinate_names must contain exactly two names.")
        if resolved_coordinates[0] == resolved_coordinates[1]:
            raise ValueError("FNO coordinate names must be distinct.")
        labels = [
            channel.label for channel in resolved_inputs + resolved_outputs
        ]
        if len(labels) != len(set(labels)):
            raise ValueError("FNO channel labels must be globally unique.")
        if max_trajectories is not None and max_trajectories < 1:
            raise ValueError("max_trajectories must be positive.")

        self.dataset = dataset
        self.normalizer = normalizer
        self.input_channels = resolved_inputs
        self.output_channels = resolved_outputs
        self.coordinate_names = tuple(resolved_coordinates)
        # Compatibility attributes used by existing scripts and checkpoints.
        self.field_names = tuple(
            channel.label for channel in self.output_channels
        )
        self.constant_names = tuple(
            channel.label for channel in self.input_channels
        )
        self.time_coordinate, self.space_coordinate = self.coordinate_names
        self.max_trajectories = max_trajectories

    @staticmethod
    def required_field_names(
        *channel_groups: Sequence[FNOChannel],
    ) -> tuple[str, ...]:
        """Return unique raw field names required by channel definitions."""
        return tuple(
            dict.fromkeys(
                channel.key
                for channels in channel_groups
                for channel in channels
                if channel.source in {"field", "species"}
            )
        )

    @staticmethod
    def required_constant_names(
        *channel_groups: Sequence[FNOChannel],
    ) -> tuple[str, ...]:
        """Return unique raw constants required by channel definitions."""
        return tuple(
            dict.fromkeys(
                channel.key
                for channels in channel_groups
                for channel in channels
                if channel.source == "constant"
            )
        )

    def __len__(self) -> int:
        if self.max_trajectories is None:
            return len(self.dataset)
        return min(self.max_trajectories, len(self.dataset))

    @staticmethod
    def _trajectory(
        sample: Mapping[str, Any],
        field_name: str,
    ) -> torch.Tensor:
        try:
            initial = sample["input_fields"][field_name]
            future = sample["output_fields"][field_name]
        except KeyError as exc:
            raise KeyError(f"FNO field {field_name!r} is unavailable.") from exc
        return torch.cat((initial, future), dim=0)

    @classmethod
    def resolve_channel(
        cls,
        sample: Mapping[str, Any],
        channel: FNOChannel,
    ) -> torch.Tensor:
        """Resolve one physical channel before normalization or broadcasting."""
        if channel.source == "parameter":
            params = sample.get("metadata", {}).get("params", {})
            if channel.key not in params:
                raise KeyError(
                    f"FNO parameter {channel.key!r} is unavailable in "
                    "sample metadata."
                )
            value = torch.as_tensor(params[channel.key])
        elif channel.source == "constant":
            try:
                value = sample["constant_inputs"][channel.key]
            except KeyError as exc:
                raise KeyError(
                    f"FNO constant {channel.key!r} is unavailable."
                ) from exc
        elif channel.source == "field":
            value = cls._trajectory(sample, channel.key)
        elif channel.source == "species":
            grouped = cls._trajectory(sample, channel.key)
            species_by_field = sample.get("metadata", {}).get(
                "field_species",
                {},
            )
            species_names = species_by_field.get(channel.key)
            if species_names is None:
                raise KeyError(
                    f"FNO species metadata for field {channel.key!r} "
                    "is unavailable."
                )
            try:
                species_index = list(species_names).index(channel.species)
            except ValueError as exc:
                raise KeyError(
                    f"Species {channel.species!r} is absent from "
                    f"field {channel.key!r}."
                ) from exc
            if grouped.ndim != 3 or grouped.shape[-1] != len(species_names):
                raise ValueError(
                    f"FNO species field {channel.key!r} must have "
                    "(first coordinate, second coordinate, species) shape; "
                    f"received {tuple(grouped.shape)}."
                )
            value = grouped[..., species_index]
        else:
            raise ValueError(f"Unsupported FNO channel source {channel.source!r}.")

        value = torch.as_tensor(value)
        if not value.is_floating_point():
            value = value.to(torch.get_default_dtype())
        if not torch.isfinite(value).all():
            raise ValueError(
                f"FNO channel {channel.label!r} contains NaN or infinity."
            )
        return value

    @staticmethod
    def _as_grid(
        value: torch.Tensor,
        shape: tuple[int, int],
        label: str,
    ) -> torch.Tensor:
        if value.numel() == 1:
            return value.reshape(()).expand(shape)
        if tuple(value.shape) != shape:
            raise ValueError(
                f"FNO channel {label!r} must be scalar or have shape {shape}; "
                f"received {tuple(value.shape)}."
            )
        return value

    def _coordinates(
        self,
        sample: Mapping[str, Any],
        shape: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        first_name, second_name = self.coordinate_names
        first_values = torch.cat(
            (
                sample["input_coordinates"][first_name],
                sample["output_coordinates"][first_name],
            )
        ).reshape(-1)
        input_second = sample["input_coordinates"][second_name].reshape(-1)
        output_second = sample["output_coordinates"][second_name].reshape(-1)
        if input_second.shape != output_second.shape or not torch.equal(
            input_second,
            output_second,
        ):
            raise ValueError(
                f"Coordinate {second_name!r} must be unchanged "
                "across the complete trajectory."
            )
        if (first_values.numel(), output_second.numel()) != shape:
            raise ValueError(
                "Coordinate sizes do not match the FNO field grid: "
                f"coordinates {(first_values.numel(), output_second.numel())}, "
                f"field {shape}."
            )
        if first_values.numel() < 2 or output_second.numel() < 2:
            raise ValueError("FNO fields require at least a 2-by-2 grid.")
        if not torch.all(first_values[1:] > first_values[:-1]) or not torch.all(
            output_second[1:] > output_second[:-1]
        ):
            raise ValueError("FNO coordinates must be strictly increasing.")
        return first_values, output_second

    def physical_item(self, index: int) -> dict[str, Any]:
        """Return one unnormalized model input, target, and grid."""
        sample = self.dataset[index]
        resolved_outputs = [
            self.resolve_channel(sample, channel)
            for channel in self.output_channels
        ]
        field_outputs = [
            value for value in resolved_outputs if value.numel() != 1
        ]
        if not field_outputs:
            raise ValueError(
                "At least one FNO output channel must define the spatial grid."
            )
        if field_outputs[0].ndim != 2:
            raise ValueError(
                "FNO spatial channels must have two dimensions; received "
                f"{tuple(field_outputs[0].shape)}."
            )
        shape = tuple(field_outputs[0].shape)
        output = torch.stack(
            [
                self._as_grid(value, shape, channel.label)
                for value, channel in zip(
                    resolved_outputs,
                    self.output_channels,
                )
            ]
        )
        model_input = torch.stack(
            [
                self._as_grid(
                    self.resolve_channel(sample, channel),
                    shape,
                    channel.label,
                )
                for channel in self.input_channels
            ]
        )
        first_values, second_values = self._coordinates(sample, shape)
        return {
            "x": model_input,
            "y": output,
            self.coordinate_names[0]: first_values,
            self.coordinate_names[1]: second_values,
            "metadata": sample.get("metadata", {}),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        physical = self.physical_item(index)
        target = torch.stack(
            [
                self.normalizer.normalize(
                    physical["y"][index],
                    channel.label,
                )
                for index, channel in enumerate(self.output_channels)
            ],
            dim=0,
        )
        model_input = torch.stack(
            [
                self.normalizer.normalize(
                    physical["x"][index],
                    channel.label,
                )
                for index, channel in enumerate(self.input_channels)
            ],
            dim=0,
        )
        return {
            "x": model_input,
            "y": target,
            self.coordinate_names[0]: physical[self.coordinate_names[0]],
            self.coordinate_names[1]: physical[self.coordinate_names[1]],
        }

    def denormalize_output(self, output: torch.Tensor) -> torch.Tensor:
        """Convert channel-first FNO output back to physical field values."""
        if output.ndim < 3 or output.shape[-3] != len(self.output_channels):
            raise ValueError(
                "FNO output must end in (channel, first, second) with "
                f"{len(self.output_channels)} channels; received "
                f"{tuple(output.shape)}."
            )
        fields = [
            self.normalizer.denormalize(
                output.select(-3, index),
                channel.label,
            )
            for index, channel in enumerate(self.output_channels)
        ]
        return torch.stack(fields, dim=-3)


class AutoencoderAdapter(Dataset):
    """Expose complete processed trajectories as reconstruction pairs."""

    def __init__(
        self,
        dataset: Dataset,
        processor: DataProcessor,
        *,
        flatten: bool = False,
    ):
        if processor.target_transform.is_delta:
            raise ValueError(
                "AutoencoderAdapter requires state targets so inputs and "
                "reconstruction targets use one representation."
            )
        self.dataset = dataset
        self.processor = processor
        self.flatten = flatten

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        processed = self.processor(self.dataset[index])
        packer = self.processor.field_packer
        x = packer.to_channel_last(processed["x"])
        y = packer.to_channel_last(processed["y"])
        states = torch.cat((x, y), dim=-2)
        if self.flatten:
            states = states.reshape(-1)
        return {
            **processed,
            "x": states,
            "y": states.clone(),
            "target": states.clone(),
        }


@dataclass
class _RunningMoments:
    count: int = 0
    mean: torch.Tensor | None = None
    m2: torch.Tensor | None = None

    def update(self, values: torch.Tensor) -> None:
        values = values.detach().to(dtype=torch.float64, device="cpu")
        if values.ndim == 0:
            values = values.reshape(1)
        batch_count = int(values.shape[0])
        if batch_count == 0:
            return
        batch_mean = values.mean(dim=0)
        batch_m2 = torch.sum((values - batch_mean) ** 2, dim=0)
        if self.mean is None:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return
        assert self.m2 is not None
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * (batch_count / total)
        self.m2 = (
            self.m2
            + batch_m2
            + delta**2 * (self.count * batch_count / total)
        )
        self.count = total

    def result(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.count == 0 or self.mean is None or self.m2 is None:
            raise RuntimeError("No values were accumulated.")
        variance = self.m2 / self.count
        return self.mean.float(), torch.sqrt(variance).float()


def fit_fno_zscore_normalizer(
    dataset: Dataset | Iterable[Mapping[str, Any]],
    input_channels: Sequence[FNOChannel],
    output_channels: Sequence[FNOChannel],
    *,
    min_denom: float = 1e-8,
) -> ZScoreNormalizer:
    """Fit scalar, resolution-independent statistics for an FNO adapter.

    Every value of a spatial channel contributes to one global channel mean
    and standard deviation. Parameter and constant channels contribute once
    per sample. Consequently, fitted statistics broadcast on grids whose
    resolution differs from the training grid.

    Delta statistics are set to identity values because ``FNOAdapter`` models
    complete fields rather than one-step deltas.
    """

    inputs = tuple(input_channels)
    outputs = tuple(output_channels)
    if not inputs:
        raise ValueError("input_channels must contain at least one channel.")
    if not outputs:
        raise ValueError("output_channels must contain at least one channel.")
    labels = [channel.label for channel in inputs + outputs]
    if len(labels) != len(set(labels)):
        raise ValueError("FNO channel labels must be globally unique.")

    moments = {label: _RunningMoments() for label in labels}
    if isinstance(dataset, Dataset):
        samples: Iterable[Mapping[str, Any]] = (
            dataset[index] for index in range(len(dataset))
        )
    else:
        samples = dataset

    for sample in samples:
        for channel in inputs + outputs:
            value = FNOAdapter.resolve_channel(sample, channel)
            moments[channel.label].update(value.reshape(-1))

    means: dict[str, torch.Tensor] = {}
    stds: dict[str, torch.Tensor] = {}
    for label in labels:
        means[label], stds[label] = moments[label].result()
    output_labels = tuple(channel.label for channel in outputs)
    return ZScoreNormalizer(
        {
            "mean": means,
            "std": stds,
            "mean_delta": {
                label: torch.tensor(0.0, dtype=torch.float32)
                for label in output_labels
            },
            "std_delta": {
                label: torch.tensor(1.0, dtype=torch.float32)
                for label in output_labels
            },
        },
        variable_field_order=output_labels,
        constant_field_order=tuple(channel.label for channel in inputs),
        min_denom=min_denom,
    )


def fit_zscore_normalizer(
    dataset: Dataset | Iterable[Mapping[str, Any]],
    variable_field_order: Sequence[str],
    constant_field_order: Sequence[str] = (),
    *,
    min_denom: float = 1e-8,
) -> ZScoreNormalizer:
    """Fit field and one-step-delta statistics from raw trajectories.

    The function streams one trajectory at a time and never concatenates a
    complete HDF5 split in memory.
    """

    variables = tuple(variable_field_order)
    constants = tuple(constant_field_order)
    moments = {name: _RunningMoments() for name in variables}
    delta_moments = {name: _RunningMoments() for name in variables}
    constant_moments = {name: _RunningMoments() for name in constants}

    if isinstance(dataset, Dataset):
        samples: Iterable[Mapping[str, Any]] = (
            dataset[index] for index in range(len(dataset))
        )
    else:
        samples = dataset

    for sample in samples:
        for name in variables:
            trajectory = torch.cat(
                (sample["input_fields"][name], sample["output_fields"][name]),
                dim=0,
            )
            moments[name].update(trajectory)
            delta_moments[name].update(trajectory[1:] - trajectory[:-1])
        for name in constants:
            value = sample["constant_inputs"][name]
            constant_moments[name].update(value.unsqueeze(0))

    means: dict[str, torch.Tensor] = {}
    stds: dict[str, torch.Tensor] = {}
    delta_means: dict[str, torch.Tensor] = {}
    delta_stds: dict[str, torch.Tensor] = {}
    for name in variables:
        means[name], stds[name] = moments[name].result()
        delta_means[name], delta_stds[name] = delta_moments[name].result()
    for name in constants:
        means[name], stds[name] = constant_moments[name].result()

    return ZScoreNormalizer(
        {
            "mean": means,
            "std": stds,
            "mean_delta": delta_means,
            "std_delta": delta_stds,
        },
        variable_field_order=variables,
        constant_field_order=constants,
        min_denom=min_denom,
    )


@dataclass(frozen=True)
class PODTransform:
    """Trajectory-output POD transform fitted by incremental PCA."""

    mean: np.ndarray
    basis: np.ndarray
    explained_variance_ratio: np.ndarray
    cumulative_explained_variance: float
    output_shape: tuple[int, ...]

    @property
    def n_components(self) -> int:
        return self.basis.shape[1]

    def flatten(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states)
        if tuple(states.shape[-len(self.output_shape) :]) != self.output_shape:
            raise ValueError(
                f"POD states must end in {self.output_shape}, got "
                f"{tuple(states.shape)}."
            )
        leading = states.shape[: -len(self.output_shape)]
        return states.reshape(leading + (self.mean.size,))

    def flatten_tensor(self, states: torch.Tensor) -> torch.Tensor:
        if tuple(states.shape[-len(self.output_shape) :]) != self.output_shape:
            raise ValueError(
                f"POD states must end in {self.output_shape}, got "
                f"{tuple(states.shape)}."
            )
        leading = tuple(states.shape[: -len(self.output_shape)])
        return states.reshape(leading + (self.mean.size,))

    def encode(self, states: np.ndarray) -> np.ndarray:
        flattened = self.flatten(states)
        return np.einsum(
            "...d,dk->...k", flattened - self.mean, self.basis
        ).astype(np.float32, copy=False)

    def encode_tensor(self, states: torch.Tensor) -> torch.Tensor:
        flattened = self.flatten_tensor(states)
        mean = torch.tensor(
            self.mean, dtype=states.dtype, device=states.device
        )
        basis = torch.tensor(
            self.basis, dtype=states.dtype, device=states.device
        )
        return torch.einsum("...d,dk->...k", flattened - mean, basis)

    def decode(self, coefficients: np.ndarray) -> np.ndarray:
        flattened = np.einsum(
            "...k,dk->...d", coefficients, self.basis
        ) + self.mean
        return flattened.reshape(
            flattened.shape[:-1] + self.output_shape
        ).astype(np.float32, copy=False)

    def decode_tensor(self, coefficients: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(
            self.mean,
            dtype=coefficients.dtype,
            device=coefficients.device,
        )
        basis = torch.tensor(
            self.basis,
            dtype=coefficients.dtype,
            device=coefficients.device,
        )
        flattened = (
            torch.einsum("...k,dk->...d", coefficients, basis) + mean
        )
        return flattened.reshape(
            tuple(flattened.shape[:-1]) + self.output_shape
        )

    def unflatten_tensor(self, states: torch.Tensor) -> torch.Tensor:
        if states.shape[-1] != self.mean.size:
            raise ValueError(
                f"Flattened POD output has {states.shape[-1]} values; "
                f"expected {self.mean.size}."
            )
        return states.reshape(tuple(states.shape[:-1]) + self.output_shape)


def fit_incremental_pod(
    trajectory_batches: np.ndarray | Iterable[np.ndarray],
    *,
    variance_threshold: float = 0.999,
    n_components: int | None = None,
) -> PODTransform:
    """Fit trajectory-output IPCA from batches shaped ``[B, *output_shape]``.

    Components are truncated at the first cumulative explained-variance value
    meeting ``variance_threshold``.
    """

    if not 0.0 < variance_threshold <= 1.0:
        raise ValueError("variance_threshold must be in (0, 1].")
    batches = iter(trajectory_batches)
    try:
        first = np.asarray(next(batches), dtype=np.float32)
    except StopIteration:
        raise ValueError("At least one trajectory batch is required for POD fitting.")
    if first.ndim < 2:
        raise ValueError("POD batches must have shape [batch, *output_shape].")
    output_shape = tuple(first.shape[1:])
    flattened = first.reshape(first.shape[0], -1)
    n_features = flattened.shape[1]
    if n_components is None:
        n_components = min(n_features, flattened.shape[0])
    n_components = int(n_components)
    if n_components < 1:
        raise ValueError("n_components must be positive.")
    if flattened.shape[0] < n_components:
        raise ValueError("The first IPCA batch is smaller than n_components.")

    from sklearn.decomposition import IncrementalPCA

    ipca = IncrementalPCA(n_components=n_components)
    ipca.partial_fit(flattened)
    for batch in batches:
        batch = np.asarray(batch, dtype=np.float32)
        if tuple(batch.shape[1:]) != output_shape:
            raise ValueError("POD trajectory output shapes do not match.")
        flattened = batch.reshape(batch.shape[0], -1)
        if batch.shape[0] < n_components:
            raise ValueError(
                "Every IPCA trajectory batch must have at least as many "
                "trajectories as n_components."
            )
        ipca.partial_fit(flattened)

    ratios = np.nan_to_num(ipca.explained_variance_ratio_, nan=0.0)
    cumulative = np.minimum(np.cumsum(ratios), 1.0)
    if cumulative[-1] + 1e-7 < variance_threshold:
        raise ValueError(
            f"The IPCA batches explain only {cumulative[-1]:.6f} of variance; "
            "fit more components."
        )
    retained = int(np.searchsorted(cumulative, variance_threshold) + 1)
    basis = ipca.components_[:retained].T.astype(np.float32, copy=False)
    return PODTransform(
        mean=ipca.mean_.astype(np.float32, copy=False),
        basis=basis,
        explained_variance_ratio=ratios[:retained].astype(np.float32, copy=False),
        cumulative_explained_variance=float(cumulative[retained - 1]),
        output_shape=output_shape,
    )


def _collate_pod_trajectories(
    samples: Sequence[Mapping[str, Any]],
) -> torch.Tensor:
    return torch.stack([sample["target"] for sample in samples])


def fit_incremental_pod_dataset(
    dataset: DeepXDEAdapter,
    *,
    variance_threshold: float = 0.999,
    max_components: int | None = None,
    num_workers: int = 0,
) -> PODTransform:
    """Fit trajectory POD lazily, increasing IPCA rank until the threshold."""

    n_trajectories = len(dataset)
    if n_trajectories < 2:
        raise ValueError("Trajectory POD requires at least two trajectories.")
    sample = dataset[0]["target"]
    n_features = sample.numel()
    limit = min(n_trajectories, n_features)
    if max_components is not None:
        limit = min(limit, max_components)
    candidate = min(16, limit)
    while True:
        if n_trajectories < 2 * candidate:
            batches = [list(range(n_trajectories))]
        else:
            n_batches = n_trajectories // candidate
            quotient, remainder = divmod(n_trajectories, n_batches)
            sizes = [
                quotient + (index < remainder)
                for index in range(n_batches)
            ]
            batches = []
            start = 0
            for size in sizes:
                batches.append(list(range(start, start + size)))
                start += size
        loader = DataLoader(
            dataset,
            batch_sampler=batches,
            num_workers=num_workers,
            collate_fn=_collate_pod_trajectories,
        )
        try:
            return fit_incremental_pod(
                (batch.numpy() for batch in loader),
                variance_threshold=variance_threshold,
                n_components=candidate,
            )
        except ValueError as error:
            if "fit more components" not in str(error) or candidate == limit:
                raise
            candidate = min(2 * candidate, limit)


@dataclass(frozen=True)
class CoordinateScaler:
    minimum: np.ndarray
    span: np.ndarray

    @classmethod
    def fit(cls, coordinates: np.ndarray) -> "CoordinateScaler":
        minimum = np.min(coordinates, axis=0)
        maximum = np.max(coordinates, axis=0)
        span = np.maximum(maximum - minimum, 1e-12)
        return cls(minimum=minimum, span=span)

    def transform(self, coordinates: np.ndarray) -> np.ndarray:
        return (2.0 * (coordinates - self.minimum) / self.span - 1.0).astype(
            np.float32,
            copy=False,
        )

    def transform_tensor(self, coordinates: torch.Tensor) -> torch.Tensor:
        minimum = torch.as_tensor(
            self.minimum,
            dtype=coordinates.dtype,
            device=coordinates.device,
        )
        span = torch.as_tensor(
            self.span,
            dtype=coordinates.dtype,
            device=coordinates.device,
        )
        return 2.0 * (coordinates - minimum) / span - 1.0


@dataclass(frozen=True)
class DeepONetBenchmarkConfig:
    loss: Literal["relative_l2", "mse"] = "relative_l2"
    epochs: int = 2000
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 32
    width: int = 128
    latent_width: int = 8
    branch_hidden_layers: int = 2
    trunk_hidden_layers: int = 2
    activation: str = "tanh"
    display_every: int = 100
    variance_threshold: float = 0.999
    seed: int = 7
    plot_cases: int = 2


@dataclass(frozen=True)
class DeepONetBenchmarkResult:
    direct_model: Any
    pod_model: Any
    pod: PODTransform
    metrics: Mapping[str, Mapping[str, float]]


def _network(
    branch_width: int,
    trunk_width: int,
    output_width: int,
    config: DeepONetBenchmarkConfig,
):
    import deepxde as dde

    if output_width == 1:
        branch_last = config.latent_width
        strategy = None
    else:
        branch_last = config.latent_width * output_width
        strategy = "split_branch"
    branch_layers = [
        branch_width,
        *([config.width] * config.branch_hidden_layers),
        branch_last,
    ]
    trunk_layers = [
        trunk_width,
        *([config.width] * config.trunk_hidden_layers),
        config.latent_width,
    ]
    return dde.nn.DeepONetCartesianProd(
        branch_layers,
        trunk_layers,
        config.activation,
        "Glorot normal",
        num_outputs=output_width,
        multi_output_strategy=strategy,
        regularization=("l2", config.weight_decay),
    )


class _PODOnlyDeepONet(torch.nn.Module):
    """DeepXDE PODDeepONet with a fixed POD-only trunk and PCA mean shift."""

    def __init__(
        self,
        branch_width: int,
        pod: PODTransform,
        config: DeepONetBenchmarkConfig,
    ):
        super().__init__()
        import deepxde as dde

        self.network = dde.nn.PODDeepONet(
            np.array(pod.basis, dtype=np.float32, copy=True),
            [
                branch_width,
                *([config.width] * config.branch_hidden_layers),
                pod.n_components,
            ],
            config.activation,
            "Glorot normal",
            layer_sizes_trunk=None,
            regularization=("l2", config.weight_decay),
        )
        basis = self.network.pod_basis
        del self.network.pod_basis
        self.network.register_buffer("pod_basis", basis)
        self.register_buffer(
            "pod_mean",
            torch.tensor(np.array(pod.mean, copy=True), dtype=torch.float32),
        )

    @property
    def branch(self) -> torch.nn.Module:
        return self.network.branch

    @property
    def trunk(self) -> None:
        return None

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return self.network(inputs) + self.pod_mean


@dataclass
class DeepONetTrainingHistory:
    """Minimal loss history shared by lazy training and plotting."""

    steps: list[int]
    loss_train: list[list[float]]
    loss_test: list[list[float]]
    loss_name: str = "relative_l2"


TensorTransform = Callable[[torch.Tensor], torch.Tensor]
MetricReporter = Callable[[Mapping[str, float | int]], None]


def deeponet_parameter_counts(model: torch.nn.Module) -> dict[str, int]:
    """Count trainable parameters in the branch, trunk, and full network."""

    def trainable(module: torch.nn.Module) -> int:
        return sum(
            parameter.numel()
            for parameter in module.parameters()
            if parameter.requires_grad
        )

    branch = getattr(model, "branch", None)
    trunk = getattr(model, "trunk", None)
    if not isinstance(branch, torch.nn.Module):
        raise TypeError("DeepONet model does not expose a branch module.")
    return {
        "n_params": trainable(model),
        "n_params_branch": trainable(branch),
        "n_params_trunk": 0 if trunk is None else trainable(trunk),
    }


def collate_deeponet_trajectories(
    samples: Sequence[Mapping[str, Any]],
    *,
    coordinate_scaler: CoordinateScaler,
    target_transform: TensorTransform | None = None,
) -> dict[str, Any]:
    """Collate lazily loaded trajectories on one shared trunk grid."""

    if not samples:
        raise ValueError("Cannot collate an empty trajectory batch.")
    reference_trunk = samples[0]["trunk"]
    for sample in samples[1:]:
        candidate = sample["trunk"]
        if candidate.shape != reference_trunk.shape or not torch.allclose(
            candidate,
            reference_trunk,
            rtol=1e-5,
            atol=1e-8,
        ):
            raise ValueError(
                "Cartesian-product batches require a shared trunk grid."
            )
    targets = torch.stack([sample["target"] for sample in samples])
    if target_transform is not None:
        targets = target_transform(targets)
    return {
        "branch": torch.stack([sample["branch"] for sample in samples]),
        "trunk": coordinate_scaler.transform_tensor(reference_trunk),
        "target": targets,
        "coordinates": tuple(sample["coordinate"] for sample in samples),
        "labels": tuple(samples[0]["labels"]),
    }


def make_deeponet_dataloader(
    dataset: DeepXDEAdapter,
    *,
    batch_size: int,
    shuffle: bool,
    coordinate_scaler: CoordinateScaler,
    target_transform: TensorTransform | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 0,
) -> DataLoader:
    """Build a loader that reads only the trajectories in the current batch."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")
    generator = torch.Generator(
        device=torch.get_default_device()
    ).manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=pin_memory,
        generator=generator,
        collate_fn=partial(
            collate_deeponet_trajectories,
            coordinate_scaler=coordinate_scaler,
            target_transform=target_transform,
        ),
    )


def _loss_target(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if prediction.ndim + 1 == target.ndim and target.shape[-1] == 1:
        return target[..., 0]
    return target


def _with_output_channel(
    prediction: torch.Tensor,
    output_width: int,
) -> torch.Tensor:
    if output_width == 1 and prediction.ndim == 2:
        return prediction.unsqueeze(-1)
    return prediction


def relative_l2_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Mean trajectory-wise ``||prediction-target||₂ / ||target||₂``."""

    if prediction.shape != target.shape:
        raise ValueError(
            "prediction and target must have identical shapes for relative L2."
        )
    error = (prediction - target).reshape(target.shape[0], -1)
    reference = target.reshape(target.shape[0], -1)
    denominator = torch.linalg.vector_norm(reference, dim=1)
    denominator = denominator.clamp_min(torch.finfo(target.dtype).eps)
    return (
        torch.linalg.vector_norm(error, dim=1) / denominator
    ).mean()


def _loader_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    pin_memory: bool,
    loss_name: Literal["relative_l2", "mse"],
) -> float:
    training = optimizer is not None
    model.train(training)
    accumulated_loss = 0.0
    count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            branch = batch["branch"].to(
                device, non_blocking=pin_memory
            )
            trunk = batch["trunk"].to(device, non_blocking=pin_memory)
            target = batch["target"].to(device, non_blocking=pin_memory)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            prediction = model((branch, trunk))
            target = _loss_target(prediction, target)
            if loss_name == "relative_l2":
                loss = relative_l2_loss(prediction, target)
                batch_count = target.shape[0]
            elif loss_name == "mse":
                loss = torch.nn.functional.mse_loss(prediction, target)
                batch_count = target.numel()
            else:
                raise ValueError(f"Unsupported loss {loss_name!r}.")
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            accumulated_loss += float(loss.detach()) * batch_count
            count += batch_count
    return accumulated_loss / max(count, 1)


def train_deeponet_lazy(
    train: DeepXDEAdapter,
    validation: DeepXDEAdapter,
    *,
    config: DeepONetBenchmarkConfig,
    coordinate_scaler: CoordinateScaler,
    target_transform: TensorTransform | None = None,
    pod: PODTransform | None = None,
    reporter: MetricReporter | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    device: str | torch.device | None = None,
) -> tuple[torch.nn.Module, DeepONetTrainingHistory]:
    """Train a Cartesian DeepONet from lazy trajectory ``DataLoader`` batches."""

    if config.epochs < 1 or config.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive.")
    if config.branch_hidden_layers < 1:
        raise ValueError("branch_hidden_layers must be positive.")
    if pod is None and config.trunk_hidden_layers < 1:
        raise ValueError("trunk_hidden_layers must be positive for DeepONet.")
    if config.loss not in {"relative_l2", "mse"}:
        raise ValueError("loss must be 'relative_l2' or 'mse'.")
    if pod is not None and target_transform is not None:
        raise ValueError("Use pod or target_transform, not both.")
    if train.format != "cartesian_product" or validation.format != train.format:
        raise ValueError("Lazy DeepONet training requires Cartesian adapters.")
    sample = train[0]
    sample_target = sample["target"]
    if pod is not None:
        target_transform = pod.flatten_tensor
    if target_transform is not None:
        sample_target = target_transform(sample_target)
    branch_width = int(sample["branch"].shape[-1])
    trunk_width = int(sample["trunk"].shape[-1])

    import deepxde as dde

    dde.config.set_random_seed(config.seed)
    selected_device = torch.device(
        device
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if pod is None:
        output_width = int(sample_target.shape[-1])
        model = _network(
            branch_width,
            trunk_width,
            output_width,
            config,
        )
    else:
        model = _PODOnlyDeepONet(branch_width, pod, config)
    model = model.to(selected_device)
    parameter_counts = deeponet_parameter_counts(model)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    train_loader = make_deeponet_dataloader(
        train,
        batch_size=config.batch_size,
        shuffle=True,
        coordinate_scaler=coordinate_scaler,
        target_transform=target_transform,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=config.seed,
    )
    valid_loader = make_deeponet_dataloader(
        validation,
        batch_size=config.batch_size,
        shuffle=False,
        coordinate_scaler=coordinate_scaler,
        target_transform=target_transform,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=config.seed,
    )
    history = DeepONetTrainingHistory([], [], [], config.loss)
    best_valid_loss = float("inf")
    for epoch in range(1, config.epochs + 1):
        train_loss = _loader_loss(
            model,
            train_loader,
            device=selected_device,
            optimizer=optimizer,
            pin_memory=pin_memory,
            loss_name=config.loss,
        )
        valid_loss = _loader_loss(
            model,
            valid_loader,
            device=selected_device,
            optimizer=None,
            pin_memory=pin_memory,
            loss_name=config.loss,
        )
        best_valid_loss = min(best_valid_loss, valid_loss)
        history.steps.append(epoch)
        history.loss_train.append([train_loss])
        history.loss_test.append([valid_loss])
        metrics: dict[str, float | int] = {
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "best_valid_loss": best_valid_loss,
            "epoch": epoch,
            "branch_hidden_layers": config.branch_hidden_layers,
            "trunk_hidden_layers": (
                0 if pod is not None else config.trunk_hidden_layers
            ),
            **parameter_counts,
        }
        if reporter is not None:
            reporter(metrics)
        if reporter is None and (
            epoch == 1
            or epoch == config.epochs
            or epoch % max(1, config.display_every) == 0
        ):
            print(
                f"epoch={epoch:5d} train_loss={train_loss:.6e} "
                f"valid_loss={valid_loss:.6e}"
            )
    return model, history


def tune_deeponet_hyperparameters(
    config: Mapping[str, Any],
    *,
    train: DeepXDEAdapter,
    validation: DeepXDEAdapter,
    coordinate_scaler: CoordinateScaler,
    pod: PODTransform | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> None:
    """Ray trainable body backed by lazy datasets rather than NumPy arrays."""

    from ray import tune

    benchmark_config = DeepONetBenchmarkConfig(
        loss=str(config.get("loss", "relative_l2")),
        epochs=int(config["epochs"]),
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        batch_size=int(config["batch_size"]),
        width=int(config["width"]),
        latent_width=int(config["latent_width"]),
        branch_hidden_layers=int(config.get("branch_hidden_layers", 2)),
        trunk_hidden_layers=int(config.get("trunk_hidden_layers", 2)),
        activation=str(config["activation"]),
        display_every=1,
        seed=int(config["seed"]),
    )
    train_deeponet_lazy(
        train,
        validation,
        config=benchmark_config,
        coordinate_scaler=coordinate_scaler,
        pod=pod,
        reporter=tune.report,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def _loss_curve(history: Any, kind: Literal["train", "valid"]) -> np.ndarray:
    values = history.loss_train if kind == "train" else history.loss_test
    return np.asarray([np.sum(value) for value in values], dtype=float)


@dataclass
class _StreamingMetrics:
    count: int = 0
    squared_error: float = 0.0
    reference_squared: float = 0.0
    max_absolute_error: float = 0.0

    def update(self, reference: np.ndarray, prediction: np.ndarray) -> None:
        reference64 = reference.astype(np.float64, copy=False)
        error = prediction.astype(np.float64, copy=False) - reference64
        self.count += error.size
        self.squared_error += float(np.sum(error**2))
        self.reference_squared += float(np.sum(reference64**2))
        self.max_absolute_error = max(
            self.max_absolute_error,
            float(np.max(np.abs(error))),
        )

    def result(self) -> dict[str, float]:
        return {
            "rmse": float(
                np.sqrt(self.squared_error / max(self.count, 1))
            ),
            "relative_l2": float(
                np.sqrt(
                    self.squared_error
                    / max(self.reference_squared, 1e-30)
                )
            ),
            "max_absolute_error": self.max_absolute_error,
        }


def _plot_losses(histories: Mapping[str, Any], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    colors = {"DeepONet": "tab:blue", "POD-DeepONet": "tab:orange"}
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    for name, history in histories.items():
        steps = np.asarray(history.steps)
        axis.semilogy(
            steps,
            _loss_curve(history, "train"),
            color=colors[name],
            label=f"{name} train",
        )
        axis.semilogy(
            steps,
            _loss_curve(history, "valid"),
            color=colors[name],
            linestyle="--",
            label=f"{name} validation",
        )
    loss_names = {
        getattr(history, "loss_name", "loss")
        for history in histories.values()
    }
    if loss_names == {"relative_l2"}:
        loss_label = "Relative L2 loss"
    elif loss_names == {"mse"}:
        loss_label = "Mean-squared loss"
    else:
        loss_label = "Loss"
    axis.set(xlabel="Epoch / iteration", ylabel=loss_label)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(output_dir / "training_validation_loss.png", dpi=200)
    plt.close(figure)


def _plot_reconstructions(
    coordinates: Sequence[np.ndarray],
    reference: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    labels: Sequence[str],
    selected_labels: Sequence[str],
    coordinate_label: str,
    plot_cases: int,
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    label_to_index = {label: index for index, label in enumerate(labels)}
    missing = [label for label in selected_labels if label not in label_to_index]
    if missing:
        raise KeyError("Unknown plot labels: " + ", ".join(missing))
    n_cases = min(plot_cases, reference.shape[0])
    figure, axes = plt.subplots(
        len(selected_labels),
        n_cases,
        figsize=(5.2 * n_cases, 2.6 * len(selected_labels)),
        squeeze=False,
        sharex="col",
    )
    styles = {
        "DeepONet": ("tab:blue", "--"),
        "POD-DeepONet": ("tab:orange", ":"),
    }
    for column in range(n_cases):
        for row, label in enumerate(selected_labels):
            axis = axes[row, column]
            channel = label_to_index[label]
            axis.plot(
                coordinates[column],
                reference[column, :, channel],
                color="black",
                linewidth=1.8,
                label="Cantera",
            )
            for name, values in predictions.items():
                color, linestyle = styles[name]
                axis.plot(
                    coordinates[column],
                    values[column, :, channel],
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.3,
                    label=name,
                )
            axis.set_ylabel(label)
            axis.grid(alpha=0.2)
            if row == 0:
                axis.set_title(f"Test trajectory {column + 1}")
            if row == len(selected_labels) - 1:
                axis.set_xlabel(coordinate_label)
            if row == 0 and column == 0:
                axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "test_reconstructions.png", dpi=200)
    plt.close(figure)


def run_deepxde_benchmark(
    train: DeepXDEAdapter,
    validation: DeepXDEAdapter,
    test: DeepXDEAdapter,
    normalizer: ZScoreNormalizer,
    *,
    output_dir: str | Path,
    plot_labels: Sequence[str],
    coordinate_label: str,
    config: DeepONetBenchmarkConfig | None = None,
    direct_config: DeepONetBenchmarkConfig | None = None,
    pod_config: DeepONetBenchmarkConfig | None = None,
    pod: PODTransform | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DeepONetBenchmarkResult:
    """Lazily train direct and state-POD DeepONets and write test plots."""

    if any(
        adapter.format != "cartesian_product"
        for adapter in (train, validation, test)
    ):
        raise ValueError(
            "The shared benchmark runner requires Cartesian-product adapters."
        )
    if config is not None and (direct_config is not None or pod_config is not None):
        raise ValueError("Use config or the two model-specific configs, not both.")
    base_config = config or DeepONetBenchmarkConfig()
    direct_config = direct_config or base_config
    pod_config = pod_config or base_config
    for model_config in (direct_config, pod_config):
        if model_config.epochs < 1 or model_config.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive.")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    first_train = train[0]
    scaler = CoordinateScaler.fit(first_train["trunk"].numpy())
    if pod is None:
        pod = fit_incremental_pod_dataset(
            train,
            variance_threshold=pod_config.variance_threshold,
            num_workers=num_workers,
        )
    print(
        f"IPCA retained {pod.n_components} trajectory components for "
        f"{pod.cumulative_explained_variance:.6%} cumulative variance."
    )
    print("Training DeepONet ...")
    direct_model, direct_history = train_deeponet_lazy(
        train,
        validation,
        config=direct_config,
        coordinate_scaler=scaler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    # Keep only the actively trained model on the accelerator.
    direct_model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Training POD-DeepONet ...")
    pod_model, pod_history = train_deeponet_lazy(
        train,
        validation,
        config=pod_config,
        coordinate_scaler=scaler,
        pod=pod,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = make_deeponet_dataloader(
        test,
        batch_size=max(direct_config.batch_size, pod_config.batch_size),
        shuffle=False,
        coordinate_scaler=scaler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=direct_config.seed,
    )
    direct_device = next(direct_model.parameters()).device
    pod_device = next(pod_model.parameters()).device
    direct_model.eval()
    pod_model.eval()
    metric_state = {
        "DeepONet": _StreamingMetrics(),
        "POD-DeepONet": _StreamingMetrics(),
    }
    plot_limit = max(direct_config.plot_cases, pod_config.plot_cases)
    plot_coordinates: list[np.ndarray] = []
    plot_reference: list[np.ndarray] = []
    plot_direct: list[np.ndarray] = []
    plot_pod: list[np.ndarray] = []
    labels: tuple[str, ...] | None = None
    with torch.no_grad():
        for batch in test_loader:
            branch = batch["branch"]
            trunk = batch["trunk"]
            direct_normalized = direct_model(
                (branch.to(direct_device), trunk.to(direct_device))
            )
            direct_normalized = _with_output_channel(
                direct_normalized,
                batch["target"].shape[-1],
            ).cpu()
            pod_flattened = pod_model(
                (branch.to(pod_device), trunk.to(pod_device))
            )
            pod_normalized = pod.unflatten_tensor(pod_flattened).cpu()
            reference = normalizer.denormalize_flattened(
                batch["target"], "variable"
            ).numpy()
            direct_prediction = normalizer.denormalize_flattened(
                direct_normalized, "variable"
            ).numpy()
            pod_prediction = normalizer.denormalize_flattened(
                pod_normalized, "variable"
            ).numpy()
            metric_state["DeepONet"].update(
                reference, direct_prediction
            )
            metric_state["POD-DeepONet"].update(
                reference, pod_prediction
            )
            labels = batch["labels"]
            for index, coordinate in enumerate(batch["coordinates"]):
                if len(plot_reference) >= plot_limit:
                    break
                plot_coordinates.append(coordinate.numpy())
                plot_reference.append(reference[index])
                plot_direct.append(direct_prediction[index])
                plot_pod.append(pod_prediction[index])
    if labels is None or not plot_reference:
        raise RuntimeError("The test loader produced no trajectories.")
    metrics = {
        name: state.result() for name, state in metric_state.items()
    }
    reference_plot = np.stack(plot_reference)
    prediction_plots = {
        "DeepONet": np.stack(plot_direct),
        "POD-DeepONet": np.stack(plot_pod),
    }

    _plot_losses(
        {"DeepONet": direct_history, "POD-DeepONet": pod_history},
        output_path,
    )
    _plot_reconstructions(
        plot_coordinates,
        reference_plot,
        prediction_plots,
        labels,
        plot_labels,
        coordinate_label,
        max(direct_config.plot_cases, pod_config.plot_cases),
        output_path,
    )
    summary = {
        "direct_config": asdict(direct_config),
        "pod_config": asdict(pod_config),
        "parameter_counts": {
            "DeepONet": deeponet_parameter_counts(direct_model),
            "POD-DeepONet": deeponet_parameter_counts(pod_model),
        },
        "pod_components": pod.n_components,
        "pod_cumulative_explained_variance": pod.cumulative_explained_variance,
        "metrics": metrics,
    }
    np.savez(
        output_path / "ipca_pod_matrix.npz",
        mean=pod.mean,
        basis=pod.basis,
        explained_variance_ratio=pod.explained_variance_ratio,
        output_shape=np.asarray(pod.output_shape, dtype=np.int64),
    )
    with (output_path / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    return DeepONetBenchmarkResult(
        direct_model=direct_model,
        pod_model=pod_model,
        pod=pod,
        metrics=metrics,
    )


__all__ = [
    "AutoencoderAdapter",
    "CoordinateScaler",
    "DeepONetBenchmarkConfig",
    "DeepONetBenchmarkResult",
    "DeepONetTrainingHistory",
    "DeepXDEAdapter",
    "FNOAdapter",
    "FNOChannel",
    "NeuralOperatorAdapter",
    "OperatorArrays",
    "PODTransform",
    "deeponet_parameter_counts",
    "fit_incremental_pod",
    "fit_incremental_pod_dataset",
    "fit_fno_zscore_normalizer",
    "fit_zscore_normalizer",
    "make_deeponet_dataloader",
    "relative_l2_loss",
    "run_deepxde_benchmark",
    "train_deeponet_lazy",
    "tune_deeponet_hyperparameters",
]
