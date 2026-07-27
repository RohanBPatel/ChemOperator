"""Tensor-only normalization utilities for processed reactor datasets.

Normalizers expose both field-wise methods (used before packing) and flattened
methods (useful for model outputs whose fields have already been concatenated
along their last dimension).  Statistics are kept as tensors and moved to the
input tensor's device and dtype at call time.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol

import torch


FieldMode = Literal["variable", "constant"]


class Normalizer(Protocol):
    """Interface required by :class:`DataProcessor`."""

    def normalize(self, x: torch.Tensor, field: str) -> torch.Tensor: ...

    def denormalize(self, x: torch.Tensor, field: str) -> torch.Tensor: ...

    def delta_normalize(self, x: torch.Tensor, field: str) -> torch.Tensor: ...

    def delta_denormalize(self, x: torch.Tensor, field: str) -> torch.Tensor: ...

    def normalize_flattened(
        self, x: torch.Tensor, mode: FieldMode
    ) -> torch.Tensor: ...

    def denormalize_flattened(
        self, x: torch.Tensor, mode: FieldMode
    ) -> torch.Tensor: ...

    def delta_normalize_flattened(
        self, x: torch.Tensor, mode: Literal["variable"]
    ) -> torch.Tensor: ...

    def delta_denormalize_flattened(
        self, x: torch.Tensor, mode: Literal["variable"]
    ) -> torch.Tensor: ...


class IdentityNormalizer:
    """A no-op normalizer with the complete normalizer interface."""

    def normalize(self, x: torch.Tensor, field: str) -> torch.Tensor:
        return x

    def denormalize(self, x: torch.Tensor, field: str) -> torch.Tensor:
        return x

    def delta_normalize(self, x: torch.Tensor, field: str) -> torch.Tensor:
        return x

    def delta_denormalize(self, x: torch.Tensor, field: str) -> torch.Tensor:
        return x

    def normalize_flattened(
        self, x: torch.Tensor, mode: FieldMode
    ) -> torch.Tensor:
        return x

    def denormalize_flattened(
        self, x: torch.Tensor, mode: FieldMode
    ) -> torch.Tensor:
        return x

    def delta_normalize_flattened(
        self, x: torch.Tensor, mode: Literal["variable"]
    ) -> torch.Tensor:
        return x

    def delta_denormalize_flattened(
        self, x: torch.Tensor, mode: Literal["variable"]
    ) -> torch.Tensor:
        return x


def _as_stat(value: Any) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if not tensor.is_floating_point():
        tensor = tensor.to(torch.get_default_dtype())
    return tensor


def _for_input(stat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype if x.is_floating_point() else stat.dtype
    return stat.to(device=x.device, dtype=dtype)


def _ordered_flattened(
    statistics: Mapping[str, torch.Tensor],
    fields: Sequence[str],
) -> torch.Tensor:
    if not fields:
        return torch.empty(0)
    return torch.cat([statistics[field].reshape(-1) for field in fields])


class ZScoreNormalizer:
    """Normalize fields with means and standard deviations.

    ``stats`` follows The Well's convention::

        {
            "mean": {"T": ..., "X": ...},
            "std": {"T": ..., "X": ...},
            "mean_delta": {"T": ..., "X": ...},
            "std_delta": {"T": ..., "X": ...},
        }

    The delta mappings are only required when delta targets are used.
    """

    def __init__(
        self,
        stats: Mapping[str, Mapping[str, Any]],
        variable_field_order: Sequence[str],
        constant_field_order: Sequence[str] = (),
        min_denom: float = 1e-8,
    ):
        self.variable_field_order = tuple(variable_field_order)
        self.constant_field_order = tuple(constant_field_order)
        all_fields = self.variable_field_order + self.constant_field_order

        self.means = self._read(stats, "mean", all_fields)
        self.stds = {
            name: value.abs().clamp_min(min_denom)
            for name, value in self._read(stats, "std", all_fields).items()
        }
        self.delta_means = self._read(
            stats, "mean_delta", self.variable_field_order
        )
        self.delta_stds = {
            name: value.abs().clamp_min(min_denom)
            for name, value in self._read(
                stats, "std_delta", self.variable_field_order
            ).items()
        }
        self._make_flattened_statistics()

    @staticmethod
    def _read(
        stats: Mapping[str, Mapping[str, Any]],
        statistic: str,
        fields: Sequence[str],
    ) -> dict[str, torch.Tensor]:
        if statistic not in stats:
            raise KeyError(f"Normalization statistics do not contain {statistic!r}.")
        values = stats[statistic]
        missing = [field for field in fields if field not in values]
        if missing:
            raise KeyError(
                f"{statistic!r} is missing fields: {', '.join(missing)}"
            )
        return {field: _as_stat(values[field]) for field in fields}

    def _make_flattened_statistics(self) -> None:
        self.flattened_means = {
            "variable": _ordered_flattened(
                self.means, self.variable_field_order
            ),
            "constant": _ordered_flattened(
                self.means, self.constant_field_order
            ),
        }
        self.flattened_stds = {
            "variable": _ordered_flattened(self.stds, self.variable_field_order),
            "constant": _ordered_flattened(self.stds, self.constant_field_order),
        }
        self.flattened_delta_means = {
            "variable": _ordered_flattened(
                self.delta_means, self.variable_field_order
            )
        }
        self.flattened_delta_stds = {
            "variable": _ordered_flattened(
                self.delta_stds, self.variable_field_order
            )
        }

    @staticmethod
    def _check_field(field: str, statistics: Mapping[str, torch.Tensor]) -> None:
        if field not in statistics:
            raise KeyError(f"No normalization statistics exist for field {field!r}.")

    @staticmethod
    def _flat_stats(
        x: torch.Tensor,
        mode: FieldMode,
        offsets: Mapping[str, torch.Tensor],
        scales: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mode not in offsets:
            raise ValueError(f"Unsupported normalization mode {mode!r}.")
        offset = _for_input(offsets[mode], x)
        scale = _for_input(scales[mode], x)
        if x.shape[-1] != offset.numel():
            raise ValueError(
                "Packed channel count does not match normalization statistics: "
                f"got {x.shape[-1]}, expected {offset.numel()}."
            )
        return offset, scale

    def normalize(self, x: torch.Tensor, field: str) -> torch.Tensor:
        self._check_field(field, self.means)
        return (x - _for_input(self.means[field], x)) / _for_input(
            self.stds[field], x
        )

    def denormalize(self, x: torch.Tensor, field: str) -> torch.Tensor:
        self._check_field(field, self.means)
        return x * _for_input(self.stds[field], x) + _for_input(
            self.means[field], x
        )

    def delta_normalize(self, x: torch.Tensor, field: str) -> torch.Tensor:
        self._check_field(field, self.delta_means)
        return (x - _for_input(self.delta_means[field], x)) / _for_input(
            self.delta_stds[field], x
        )

    def delta_denormalize(self, x: torch.Tensor, field: str) -> torch.Tensor:
        self._check_field(field, self.delta_means)
        return x * _for_input(self.delta_stds[field], x) + _for_input(
            self.delta_means[field], x
        )

    def normalize_flattened(
        self, x: torch.Tensor, mode: FieldMode
    ) -> torch.Tensor:
        mean, std = self._flat_stats(
            x, mode, self.flattened_means, self.flattened_stds
        )
        return (x - mean) / std

    def denormalize_flattened(
        self, x: torch.Tensor, mode: FieldMode
    ) -> torch.Tensor:
        mean, std = self._flat_stats(
            x, mode, self.flattened_means, self.flattened_stds
        )
        return x * std + mean

    def delta_normalize_flattened(
        self, x: torch.Tensor, mode: Literal["variable"]
    ) -> torch.Tensor:
        mean, std = self._flat_stats(
            x, mode, self.flattened_delta_means, self.flattened_delta_stds
        )
        return (x - mean) / std

    def delta_denormalize_flattened(
        self, x: torch.Tensor, mode: Literal["variable"]
    ) -> torch.Tensor:
        mean, std = self._flat_stats(
            x, mode, self.flattened_delta_means, self.flattened_delta_stds
        )
        return x * std + mean


class RMSNormalizer:
    """Normalize fields by root-mean-square statistics."""

    def __init__(
        self,
        stats: Mapping[str, Mapping[str, Any]],
        variable_field_order: Sequence[str],
        constant_field_order: Sequence[str] = (),
        min_denom: float = 1e-8,
    ):
        self.variable_field_order = tuple(variable_field_order)
        self.constant_field_order = tuple(constant_field_order)
        all_fields = self.variable_field_order + self.constant_field_order
        self.rmss = {
            name: value.abs().clamp_min(min_denom)
            for name, value in ZScoreNormalizer._read(
                stats, "rms", all_fields
            ).items()
        }
        self.delta_rmss = {
            name: value.abs().clamp_min(min_denom)
            for name, value in ZScoreNormalizer._read(
                stats, "rms_delta", self.variable_field_order
            ).items()
        }
        self.flattened_rmss = {
            "variable": _ordered_flattened(
                self.rmss, self.variable_field_order
            ),
            "constant": _ordered_flattened(
                self.rmss, self.constant_field_order
            ),
        }
        self.flattened_delta_rmss = {
            "variable": _ordered_flattened(
                self.delta_rmss, self.variable_field_order
            )
        }

    @staticmethod
    def _scale_field(
        x: torch.Tensor,
        field: str,
        statistics: Mapping[str, torch.Tensor],
        inverse: bool,
    ) -> torch.Tensor:
        ZScoreNormalizer._check_field(field, statistics)
        scale = _for_input(statistics[field], x)
        return x * scale if inverse else x / scale

    @staticmethod
    def _scale_flattened(
        x: torch.Tensor,
        mode: FieldMode,
        statistics: Mapping[str, torch.Tensor],
        inverse: bool,
    ) -> torch.Tensor:
        if mode not in statistics:
            raise ValueError(f"Unsupported normalization mode {mode!r}.")
        scale = _for_input(statistics[mode], x)
        if x.shape[-1] != scale.numel():
            raise ValueError(
                "Packed channel count does not match normalization statistics: "
                f"got {x.shape[-1]}, expected {scale.numel()}."
            )
        return x * scale if inverse else x / scale

    def normalize(self, x: torch.Tensor, field: str) -> torch.Tensor:
        return self._scale_field(x, field, self.rmss, inverse=False)

    def denormalize(self, x: torch.Tensor, field: str) -> torch.Tensor:
        return self._scale_field(x, field, self.rmss, inverse=True)

    def delta_normalize(self, x: torch.Tensor, field: str) -> torch.Tensor:
        return self._scale_field(x, field, self.delta_rmss, inverse=False)

    def delta_denormalize(self, x: torch.Tensor, field: str) -> torch.Tensor:
        return self._scale_field(x, field, self.delta_rmss, inverse=True)

    def normalize_flattened(
        self, x: torch.Tensor, mode: FieldMode
    ) -> torch.Tensor:
        return self._scale_flattened(
            x, mode, self.flattened_rmss, inverse=False
        )

    def denormalize_flattened(
        self, x: torch.Tensor, mode: FieldMode
    ) -> torch.Tensor:
        return self._scale_flattened(x, mode, self.flattened_rmss, inverse=True)

    def delta_normalize_flattened(
        self, x: torch.Tensor, mode: Literal["variable"]
    ) -> torch.Tensor:
        return self._scale_flattened(
            x, mode, self.flattened_delta_rmss, inverse=False
        )

    def delta_denormalize_flattened(
        self, x: torch.Tensor, mode: Literal["variable"]
    ) -> torch.Tensor:
        return self._scale_flattened(
            x, mode, self.flattened_delta_rmss, inverse=True
        )


class MinMaxNormalizer:
    """Linearly map field ranges to a configurable output range."""

    def __init__(
        self,
        stats: Mapping[str, Mapping[str, Any]],
        variable_field_order: Sequence[str],
        constant_field_order: Sequence[str] = (),
        feature_range: tuple[float, float] = (0.0, 1.0),
        min_denom: float = 1e-8,
    ):
        low, high = feature_range
        if high <= low:
            raise ValueError("feature_range must have an increasing (low, high).")
        self.variable_field_order = tuple(variable_field_order)
        self.constant_field_order = tuple(constant_field_order)
        self.feature_range = (float(low), float(high))
        self.min_denom = min_denom
        all_fields = self.variable_field_order + self.constant_field_order
        self.minimums = ZScoreNormalizer._read(stats, "min", all_fields)
        self.maximums = ZScoreNormalizer._read(stats, "max", all_fields)
        self.delta_minimums = ZScoreNormalizer._read(
            stats, "min_delta", self.variable_field_order
        )
        self.delta_maximums = ZScoreNormalizer._read(
            stats, "max_delta", self.variable_field_order
        )
        self.flattened_minimums = {
            "variable": _ordered_flattened(
                self.minimums, self.variable_field_order
            ),
            "constant": _ordered_flattened(
                self.minimums, self.constant_field_order
            ),
        }
        self.flattened_maximums = {
            "variable": _ordered_flattened(
                self.maximums, self.variable_field_order
            ),
            "constant": _ordered_flattened(
                self.maximums, self.constant_field_order
            ),
        }
        self.flattened_delta_minimums = {
            "variable": _ordered_flattened(
                self.delta_minimums, self.variable_field_order
            )
        }
        self.flattened_delta_maximums = {
            "variable": _ordered_flattened(
                self.delta_maximums, self.variable_field_order
            )
        }

    def _transform(
        self,
        x: torch.Tensor,
        minimum: torch.Tensor,
        maximum: torch.Tensor,
        *,
        inverse: bool,
    ) -> torch.Tensor:
        minimum = _for_input(minimum, x)
        maximum = _for_input(maximum, x)
        scale = (maximum - minimum).abs().clamp_min(self.min_denom)
        low, high = self.feature_range
        if inverse:
            return (x - low) * scale / (high - low) + minimum
        return (x - minimum) * (high - low) / scale + low

    def _field_transform(
        self,
        x: torch.Tensor,
        field: str,
        minimums: Mapping[str, torch.Tensor],
        maximums: Mapping[str, torch.Tensor],
        *,
        inverse: bool,
    ) -> torch.Tensor:
        ZScoreNormalizer._check_field(field, minimums)
        return self._transform(
            x, minimums[field], maximums[field], inverse=inverse
        )

    def _flat_transform(
        self,
        x: torch.Tensor,
        mode: FieldMode,
        minimums: Mapping[str, torch.Tensor],
        maximums: Mapping[str, torch.Tensor],
        *,
        inverse: bool,
    ) -> torch.Tensor:
        if mode not in minimums:
            raise ValueError(f"Unsupported normalization mode {mode!r}.")
        if x.shape[-1] != minimums[mode].numel():
            raise ValueError(
                "Packed channel count does not match normalization statistics: "
                f"got {x.shape[-1]}, expected {minimums[mode].numel()}."
            )
        return self._transform(
            x, minimums[mode], maximums[mode], inverse=inverse
        )

    def normalize(self, x: torch.Tensor, field: str) -> torch.Tensor:
        return self._field_transform(
            x, field, self.minimums, self.maximums, inverse=False
        )

    def denormalize(self, x: torch.Tensor, field: str) -> torch.Tensor:
        return self._field_transform(
            x, field, self.minimums, self.maximums, inverse=True
        )

    def delta_normalize(self, x: torch.Tensor, field: str) -> torch.Tensor:
        return self._field_transform(
            x,
            field,
            self.delta_minimums,
            self.delta_maximums,
            inverse=False,
        )

    def delta_denormalize(self, x: torch.Tensor, field: str) -> torch.Tensor:
        return self._field_transform(
            x,
            field,
            self.delta_minimums,
            self.delta_maximums,
            inverse=True,
        )

    def normalize_flattened(
        self, x: torch.Tensor, mode: FieldMode
    ) -> torch.Tensor:
        return self._flat_transform(
            x,
            mode,
            self.flattened_minimums,
            self.flattened_maximums,
            inverse=False,
        )

    def denormalize_flattened(
        self, x: torch.Tensor, mode: FieldMode
    ) -> torch.Tensor:
        return self._flat_transform(
            x,
            mode,
            self.flattened_minimums,
            self.flattened_maximums,
            inverse=True,
        )

    def delta_normalize_flattened(
        self, x: torch.Tensor, mode: Literal["variable"]
    ) -> torch.Tensor:
        return self._flat_transform(
            x,
            mode,
            self.flattened_delta_minimums,
            self.flattened_delta_maximums,
            inverse=False,
        )

    def delta_denormalize_flattened(
        self, x: torch.Tensor, mode: Literal["variable"]
    ) -> torch.Tensor:
        return self._flat_transform(
            x,
            mode,
            self.flattened_delta_minimums,
            self.flattened_delta_maximums,
            inverse=True,
        )


# Compatibility with The Well's class names.
ZScoreNormalization = ZScoreNormalizer
RMSNormalization = RMSNormalizer
MinMaxNormalization = MinMaxNormalizer


__all__ = [
    "FieldMode",
    "IdentityNormalizer",
    "MinMaxNormalization",
    "MinMaxNormalizer",
    "Normalizer",
    "RMSNormalization",
    "RMSNormalizer",
    "ZScoreNormalization",
    "ZScoreNormalizer",
]
