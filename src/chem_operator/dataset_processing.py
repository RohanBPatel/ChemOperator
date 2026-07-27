"""Dictionary-to-tensor processing shared by the reactor datasets.

The module lives beside the PFR examples for now, but none of the classes are
PFR-specific.  They operate on the structured samples returned by
``CanteraDataset`` and deliberately contain no HDF5 I/O.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch.utils.data import Dataset

from chem_operator.normalization import IdentityNormalizer, Normalizer


ChannelAxis = Literal["last", "first"]
TargetMode = Literal["state", "absolute", "identity", "delta"]
MultiStepDelta = Literal["direct", "incremental", "sequential"]


@dataclass(frozen=True)
class TargetTransformConfig:
    """Configure the representation learned by a model.

    ``direct`` deltas are measured from the final input state at every output
    step. ``incremental`` (also accepted as ``sequential``) measures the first
    delta from the final input and later deltas from the preceding output.
    """

    mode: TargetMode = "state"
    multi_step_delta: MultiStepDelta = "direct"

    def __post_init__(self) -> None:
        if self.mode not in {"state", "absolute", "identity", "delta"}:
            raise ValueError(f"Unsupported target transform mode {self.mode!r}.")
        if self.multi_step_delta not in {
            "direct",
            "incremental",
            "sequential",
        }:
            raise ValueError(
                "multi_step_delta must be 'direct' or 'incremental'."
            )

    @property
    def is_delta(self) -> bool:
        return self.mode == "delta"

    @property
    def is_direct_delta(self) -> bool:
        return self.multi_step_delta == "direct"


@dataclass(frozen=True)
class NormalizationConfig:
    """Choose which parts of a processed sample are normalized."""

    enabled: bool = True
    normalize_inputs: bool = True
    normalize_targets: bool = True
    normalize_constants: bool = True


@dataclass(frozen=True)
class PackedFieldLayout:
    """Description of fields concatenated into a packed channel dimension."""

    names: tuple[str, ...]
    feature_shapes: tuple[tuple[int, ...], ...]
    widths: tuple[int, ...]
    labels: tuple[str, ...]

    @property
    def n_channels(self) -> int:
        return sum(self.widths)

    @property
    def slices(self) -> dict[str, slice]:
        result: dict[str, slice] = {}
        start = 0
        for name, width in zip(self.names, self.widths):
            result[name] = slice(start, start + width)
            start += width
        return result


class FieldPacker:
    """Pack ordered field dictionaries into channel-based model tensors.

    A variable field's first dimension is its step dimension. All remaining
    dimensions are flattened into channels, so scalar and vector fields can be
    concatenated consistently. Constants are flattened completely.
    """

    def __init__(
        self,
        *,
        channel_axis: ChannelAxis = "last",
        variable_field_order: Sequence[str] = (),
        constant_field_order: Sequence[str] = (),
    ):
        if channel_axis not in {"last", "first"}:
            raise ValueError("channel_axis must be 'last' or 'first'.")
        self.channel_axis = channel_axis
        self.variable_field_order = tuple(variable_field_order)
        self.constant_field_order = tuple(constant_field_order)
        self._variable_layout: PackedFieldLayout | None = None
        self._constant_layout: PackedFieldLayout | None = None

    @staticmethod
    def _labels(name: str, width: int) -> tuple[str, ...]:
        if width == 1:
            return (name,)
        return tuple(f"{name}[{index}]" for index in range(width))

    @staticmethod
    def _ordered_names(
        fields: Mapping[str, torch.Tensor],
        configured_order: tuple[str, ...],
        *,
        kind: str,
    ) -> tuple[str, ...]:
        names = configured_order or tuple(fields.keys())
        missing = [name for name in names if name not in fields]
        extras = [name for name in fields if name not in names]
        if missing or extras:
            details = []
            if missing:
                details.append(f"missing {kind} fields: {', '.join(missing)}")
            if extras:
                details.append(f"unordered {kind} fields: {', '.join(extras)}")
            raise KeyError("; ".join(details))
        return names

    def variable_layout(
        self, fields: Mapping[str, torch.Tensor]
    ) -> PackedFieldLayout:
        names = self._ordered_names(
            fields, self.variable_field_order, kind="variable"
        )
        shapes: list[tuple[int, ...]] = []
        widths: list[int] = []
        labels: list[str] = []
        n_steps: int | None = None

        for name in names:
            tensor = fields[name]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Variable field {name!r} is not a tensor.")
            if tensor.ndim == 0:
                raise ValueError(
                    f"Variable field {name!r} has no leading step dimension."
                )
            if n_steps is None:
                n_steps = tensor.shape[0]
            elif tensor.shape[0] != n_steps:
                raise ValueError(
                    "Variable fields do not share a step dimension: "
                    f"{name!r} has {tensor.shape[0]}, expected {n_steps}."
                )
            feature_shape = tuple(tensor.shape[1:])
            width = tensor[0].numel() if tensor.shape[0] else 0
            if width == 0:
                raise ValueError(f"Variable field {name!r} has no channels.")
            shapes.append(feature_shape)
            widths.append(width)
            labels.extend(self._labels(name, width))

        return PackedFieldLayout(
            names=names,
            feature_shapes=tuple(shapes),
            widths=tuple(widths),
            labels=tuple(labels),
        )

    def constant_layout(
        self, fields: Mapping[str, torch.Tensor]
    ) -> PackedFieldLayout:
        names = self._ordered_names(
            fields, self.constant_field_order, kind="constant"
        )
        shapes: list[tuple[int, ...]] = []
        widths: list[int] = []
        labels: list[str] = []
        for name in names:
            tensor = fields[name]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Constant field {name!r} is not a tensor.")
            shape = tuple(tensor.shape)
            width = tensor.numel()
            if width == 0:
                raise ValueError(f"Constant field {name!r} has no values.")
            shapes.append(shape)
            widths.append(width)
            labels.extend(self._labels(name, width))
        return PackedFieldLayout(
            names=names,
            feature_shapes=tuple(shapes),
            widths=tuple(widths),
            labels=tuple(labels),
        )

    @staticmethod
    def _validate_cat_tensors(
        tensors: Sequence[torch.Tensor], *, kind: str
    ) -> None:
        if not tensors:
            return
        first = tensors[0]
        for tensor in tensors[1:]:
            if tensor.dtype != first.dtype or tensor.device != first.device:
                raise ValueError(
                    f"All {kind} fields must have the same dtype and device."
                )

    def pack_variable(self, fields: Mapping[str, torch.Tensor]) -> torch.Tensor:
        layout = self.variable_layout(fields)
        tensors = [
            fields[name].reshape(fields[name].shape[0], width)
            for name, width in zip(layout.names, layout.widths)
        ]
        self._validate_cat_tensors(tensors, kind="variable")
        if tensors:
            packed = torch.cat(tensors, dim=-1)
        else:
            packed = torch.empty((0, 0))
        self._variable_layout = layout
        if self.channel_axis == "first":
            packed = packed.movedim(-1, 0)
        return packed

    def pack_constants(self, fields: Mapping[str, torch.Tensor]) -> torch.Tensor:
        layout = self.constant_layout(fields)
        tensors = [fields[name].reshape(-1) for name in layout.names]
        self._validate_cat_tensors(tensors, kind="constant")
        packed = torch.cat(tensors) if tensors else torch.empty(0)
        self._constant_layout = layout
        return packed

    # Short aliases are convenient when using the packer outside DataProcessor.
    pack_fields = pack_variable
    pack_constant_fields = pack_constants

    def to_channel_last(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move a packed variable tensor's channels to the last axis."""

        if self.channel_axis == "last":
            return tensor
        if tensor.ndim < 2:
            raise ValueError("A packed variable tensor must have at least two axes.")
        return tensor.movedim(-2, -1)

    def from_channel_last(self, tensor: torch.Tensor) -> torch.Tensor:
        """Restore the configured variable channel axis."""

        if self.channel_axis == "last":
            return tensor
        if tensor.ndim < 2:
            raise ValueError("A packed variable tensor must have at least two axes.")
        return tensor.movedim(-1, -2)

    def unpack_variable(
        self,
        tensor: torch.Tensor,
        *,
        layout: PackedFieldLayout | None = None,
    ) -> dict[str, torch.Tensor]:
        layout = layout or self._variable_layout
        if layout is None:
            raise RuntimeError("Pack variable fields before unpacking a tensor.")
        channel_last = self.to_channel_last(tensor)
        if channel_last.shape[-1] != layout.n_channels:
            raise ValueError(
                f"Packed tensor has {channel_last.shape[-1]} channels; "
                f"the field layout expects {layout.n_channels}."
            )
        result: dict[str, torch.Tensor] = {}
        for name, shape in zip(layout.names, layout.feature_shapes):
            value = channel_last[..., layout.slices[name]]
            result[name] = value.reshape(value.shape[:-1] + shape)
        return result

    def unpack_constants(
        self,
        tensor: torch.Tensor,
        *,
        layout: PackedFieldLayout | None = None,
    ) -> dict[str, torch.Tensor]:
        layout = layout or self._constant_layout
        if layout is None:
            raise RuntimeError("Pack constant fields before unpacking a tensor.")
        if tensor.shape[-1] != layout.n_channels:
            raise ValueError(
                f"Packed tensor has {tensor.shape[-1]} channels; "
                f"the constant layout expects {layout.n_channels}."
            )
        result: dict[str, torch.Tensor] = {}
        for name, shape in zip(layout.names, layout.feature_shapes):
            value = tensor[..., layout.slices[name]]
            result[name] = value.reshape(value.shape[:-1] + shape)
        return result


class DataProcessor:
    """Transform one raw ``CanteraDataset`` sample into model tensors."""

    def __init__(
        self,
        *,
        field_packer: FieldPacker,
        normalizer: Normalizer | None = None,
        normalization_config: NormalizationConfig | None = None,
        target_transform: TargetTransformConfig | None = None,
        preserve_auxiliary: bool = True,
    ):
        self.field_packer = field_packer
        self.normalizer = normalizer or IdentityNormalizer()
        self.normalization_config = normalization_config or NormalizationConfig()
        self.target_transform = target_transform or TargetTransformConfig()
        self.preserve_auxiliary = preserve_auxiliary
        self._output_layout: PackedFieldLayout | None = None

    def _target_fields(
        self,
        input_fields: Mapping[str, torch.Tensor],
        output_fields: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if not self.target_transform.is_delta:
            return dict(output_fields)

        targets: dict[str, torch.Tensor] = {}
        for name, output in output_fields.items():
            if name not in input_fields:
                raise KeyError(
                    f"Delta target {name!r} is absent from the input fields."
                )
            input_field = input_fields[name]
            if input_field.shape[1:] != output.shape[1:]:
                raise ValueError(
                    f"Input and output shapes for {name!r} are incompatible: "
                    f"{tuple(input_field.shape)} and {tuple(output.shape)}."
                )
            reference = input_field[-1]
            if self.target_transform.is_direct_delta:
                targets[name] = output - reference
            else:
                first = output[:1] - reference
                later = output[1:] - output[:-1]
                targets[name] = torch.cat((first, later), dim=0)
        return targets

    def _normalize_fields(
        self,
        fields: Mapping[str, torch.Tensor],
        *,
        delta: bool,
    ) -> dict[str, torch.Tensor]:
        operation = (
            self.normalizer.delta_normalize
            if delta
            else self.normalizer.normalize
        )
        return {name: operation(value, name) for name, value in fields.items()}

    def __call__(self, raw_sample: Mapping[str, Any]) -> dict[str, Any]:
        required = {"input_fields", "output_fields", "constant_inputs"}
        missing = required - raw_sample.keys()
        if missing:
            raise KeyError(
                "Raw sample is missing keys: " + ", ".join(sorted(missing))
            )

        input_fields = dict(raw_sample["input_fields"])
        output_fields = dict(raw_sample["output_fields"])
        constant_fields = dict(raw_sample["constant_inputs"])
        target_fields = self._target_fields(input_fields, output_fields)

        config = self.normalization_config
        if config.enabled:
            if config.normalize_inputs:
                input_fields = self._normalize_fields(input_fields, delta=False)
            if config.normalize_targets:
                target_fields = self._normalize_fields(
                    target_fields, delta=self.target_transform.is_delta
                )
            if config.normalize_constants:
                constant_fields = self._normalize_fields(
                    constant_fields, delta=False
                )

        input_layout = self.field_packer.variable_layout(input_fields)
        output_layout = self.field_packer.variable_layout(target_fields)
        constant_layout = self.field_packer.constant_layout(constant_fields)
        x = self.field_packer.pack_variable(input_fields)
        y = self.field_packer.pack_variable(target_fields)
        constants = self.field_packer.pack_constants(constant_fields)
        self._output_layout = output_layout

        processed: dict[str, Any] = {
            "x": x,
            "y": y,
            "constants": constants,
            "labels": {
                "x": input_layout.labels,
                "y": output_layout.labels,
                "constants": constant_layout.labels,
            },
        }
        if self.preserve_auxiliary:
            for key, value in raw_sample.items():
                if key not in required:
                    processed[key] = value
        return processed

    process = __call__

    def inverse_reconstruct(
        self,
        prediction: torch.Tensor,
        model_input: torch.Tensor,
        *,
        as_fields: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Convert model targets back to physical future states.

        ``prediction`` and ``model_input`` may be individual samples or batched
        tensors from a DataLoader. They must use this processor's configured
        channel axis.
        """

        target = self.field_packer.to_channel_last(prediction)
        inputs = self.field_packer.to_channel_last(model_input)
        config = self.normalization_config

        if config.enabled and config.normalize_inputs:
            inputs = self.normalizer.denormalize_flattened(inputs, "variable")

        if config.enabled and config.normalize_targets:
            if self.target_transform.is_delta:
                target = self.normalizer.delta_denormalize_flattened(
                    target, "variable"
                )
            else:
                target = self.normalizer.denormalize_flattened(
                    target, "variable"
                )

        if self.target_transform.is_delta:
            if inputs.shape[-1] != target.shape[-1]:
                raise ValueError(
                    "Delta reconstruction requires matching input and output "
                    f"channels, got {inputs.shape[-1]} and {target.shape[-1]}."
                )
            reference = inputs[..., -1, :].unsqueeze(-2)
            if self.target_transform.is_direct_delta:
                target = reference + target
            else:
                target = reference + torch.cumsum(target, dim=-2)

        reconstructed = self.field_packer.from_channel_last(target)
        if not as_fields:
            return reconstructed
        if self._output_layout is None:
            raise RuntimeError("Process at least one sample before requesting fields.")
        return self.field_packer.unpack_variable(
            reconstructed, layout=self._output_layout
        )

    # Conventional aliases for inference code.
    inverse_transform = inverse_reconstruct
    reconstruct = inverse_reconstruct


class ProcessedDataset(Dataset):
    """Apply a ``DataProcessor`` lazily around another map-style dataset."""

    def __init__(self, raw_dataset: Dataset, processor: DataProcessor):
        self.raw_dataset = raw_dataset
        self.processor = processor

    def __len__(self) -> int:
        return len(self.raw_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.processor(self.raw_dataset[index])

    def close(self) -> None:
        close = getattr(self.raw_dataset, "close", None)
        if close is not None:
            close()


__all__ = [
    "DataProcessor",
    "FieldPacker",
    "NormalizationConfig",
    "PackedFieldLayout",
    "ProcessedDataset",
    "TargetTransformConfig",
]
