"""Tensor-only normalization utilities for processed reactor datasets.

Normalizers expose both field-wise methods (used before packing) and flattened
methods (useful for model outputs whose fields have already been concatenated
along their last dimension).  Statistics are kept as tensors and moved to the
input tensor's device and dtype at call time.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol, cast

import torch


FieldMode = Literal["variable", "constant"]
NormalizerState = dict[str, Any]

_NORMALIZER_STATE_SCHEMA = "chem-operator-normalizer"
_NORMALIZER_STATE_VERSION = 1


class Normalizer(Protocol):
    """Interface required by :class:`DataProcessor`."""

    def state_dict(self) -> NormalizerState:
        """Return a versioned, JSON- and ``torch.save``-safe state."""
        ...

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

    def state_dict(self) -> NormalizerState:
        return _normalizer_state("identity")

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


def _normalizer_state(normalizer_type: str, **values: Any) -> NormalizerState:
    return {
        "schema": _NORMALIZER_STATE_SCHEMA,
        "version": _NORMALIZER_STATE_VERSION,
        "type": normalizer_type,
        **values,
    }


def _tensor_state(value: torch.Tensor) -> dict[str, Any]:
    """Encode one tensor without relying on pickle-specific tensor objects."""
    tensor = value.detach().cpu().contiguous()
    return {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": list(tensor.shape),
        "data": tensor.tolist(),
    }


def _tensor_from_state(value: Any, *, name: str) -> torch.Tensor:
    if not isinstance(value, Mapping):
        raise TypeError(f"Serialized statistic {name!r} must be a mapping.")

    dtype_name = value.get("dtype")
    shape = value.get("shape")
    if not isinstance(dtype_name, str):
        raise TypeError(f"Serialized statistic {name!r} has no string dtype.")
    if (
        not isinstance(shape, Sequence)
        or isinstance(shape, (str, bytes))
        or not all(isinstance(size, int) and size >= 0 for size in shape)
    ):
        raise TypeError(f"Serialized statistic {name!r} has an invalid shape.")

    dtype = getattr(torch, dtype_name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(
            f"Serialized statistic {name!r} uses unsupported dtype {dtype_name!r}."
        )
    if "data" not in value:
        raise KeyError(f"Serialized statistic {name!r} has no data.")

    try:
        tensor = torch.as_tensor(value["data"], dtype=dtype)
        tensor = tensor.reshape(tuple(shape))
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(
            f"Serialized statistic {name!r} does not match shape {list(shape)!r}."
        ) from exc
    return tensor


def _statistics_state(
    statistics: Mapping[str, Mapping[str, torch.Tensor]],
) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        statistic: {
            field: _tensor_state(value)
            for field, value in sorted(fields.items())
        }
        for statistic, fields in sorted(statistics.items())
    }


def _statistics_from_state(
    value: Any,
) -> dict[str, dict[str, torch.Tensor]]:
    if not isinstance(value, Mapping):
        raise TypeError("Serialized normalizer statistics must be a mapping.")

    decoded: dict[str, dict[str, torch.Tensor]] = {}
    for statistic, raw_fields in value.items():
        if not isinstance(statistic, str) or not isinstance(raw_fields, Mapping):
            raise TypeError(
                "Serialized normalizer statistics must map names to field mappings."
            )
        decoded[statistic] = {}
        for field, raw_value in raw_fields.items():
            if not isinstance(field, str):
                raise TypeError("Serialized statistic field names must be strings.")
            decoded[statistic][field] = _tensor_from_state(
                raw_value,
                name=f"{statistic}/{field}",
            )
    return decoded


def _field_order(value: Any, *, name: str) -> tuple[str, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not all(isinstance(field, str) for field in value)
    ):
        raise TypeError(f"{name} must be a sequence of strings.")
    return tuple(value)


def _state_common(
    normalizer: Any,
    statistics: Mapping[str, Mapping[str, torch.Tensor]],
) -> dict[str, Any]:
    return {
        "variable_field_order": list(normalizer.variable_field_order),
        "constant_field_order": list(normalizer.constant_field_order),
        "min_denom": float(normalizer.min_denom),
        "statistics": _statistics_state(statistics),
    }


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
        self.min_denom = float(min_denom)
        all_fields = self.variable_field_order + self.constant_field_order

        self.means = self._read(stats, "mean", all_fields)
        self.stds = {
            name: value.abs().clamp_min(self.min_denom)
            for name, value in self._read(stats, "std", all_fields).items()
        }
        self.delta_means = self._read(
            stats, "mean_delta", self.variable_field_order
        )
        self.delta_stds = {
            name: value.abs().clamp_min(self.min_denom)
            for name, value in self._read(
                stats, "std_delta", self.variable_field_order
            ).items()
        }
        self._make_flattened_statistics()

    def state_dict(self) -> NormalizerState:
        return _normalizer_state(
            "zscore",
            **_state_common(
                self,
                {
                    "mean": self.means,
                    "std": self.stds,
                    "mean_delta": self.delta_means,
                    "std_delta": self.delta_stds,
                },
            ),
        )

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
        self.min_denom = float(min_denom)
        all_fields = self.variable_field_order + self.constant_field_order
        self.rmss = {
            name: value.abs().clamp_min(self.min_denom)
            for name, value in ZScoreNormalizer._read(
                stats, "rms", all_fields
            ).items()
        }
        self.delta_rmss = {
            name: value.abs().clamp_min(self.min_denom)
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

    def state_dict(self) -> NormalizerState:
        return _normalizer_state(
            "rms",
            **_state_common(
                self,
                {
                    "rms": self.rmss,
                    "rms_delta": self.delta_rmss,
                },
            ),
        )

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
        self.min_denom = float(min_denom)
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

    def state_dict(self) -> NormalizerState:
        state = _state_common(
            self,
            {
                "min": self.minimums,
                "max": self.maximums,
                "min_delta": self.delta_minimums,
                "max_delta": self.delta_maximums,
            },
        )
        state["feature_range"] = list(self.feature_range)
        return _normalizer_state("minmax", **state)

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


def normalizer_from_state_dict(state: Mapping[str, Any]) -> Normalizer:
    """Restore a normalizer from :meth:`Normalizer.state_dict` output.

    The serialized representation intentionally contains only mappings, lists,
    strings, integers, floats, booleans, and ``None``. It can therefore be
    round-tripped through JSON as well as through ``torch.save``.
    """
    if not isinstance(state, Mapping):
        raise TypeError("Normalizer state must be a mapping.")
    if state.get("schema") != _NORMALIZER_STATE_SCHEMA:
        raise ValueError(
            "Unsupported normalizer state schema "
            f"{state.get('schema')!r}; expected {_NORMALIZER_STATE_SCHEMA!r}."
        )
    if state.get("version") != _NORMALIZER_STATE_VERSION:
        raise ValueError(
            "Unsupported normalizer state version "
            f"{state.get('version')!r}; expected {_NORMALIZER_STATE_VERSION}."
        )

    normalizer_type = state.get("type")
    if normalizer_type == "identity":
        return cast(Normalizer, IdentityNormalizer())
    if normalizer_type not in {"zscore", "rms", "minmax"}:
        raise ValueError(f"Unsupported normalizer type {normalizer_type!r}.")

    variable_fields = _field_order(
        state.get("variable_field_order"),
        name="variable_field_order",
    )
    constant_fields = _field_order(
        state.get("constant_field_order"),
        name="constant_field_order",
    )
    try:
        min_denom = float(state["min_denom"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TypeError("Serialized normalizer min_denom must be numeric.") from exc
    statistics = _statistics_from_state(state.get("statistics"))

    if normalizer_type == "zscore":
        return cast(
            Normalizer,
            ZScoreNormalizer(
                statistics,
                variable_fields,
                constant_fields,
                min_denom=min_denom,
            ),
        )
    if normalizer_type == "rms":
        return cast(
            Normalizer,
            RMSNormalizer(
                statistics,
                variable_fields,
                constant_fields,
                min_denom=min_denom,
            ),
        )

    raw_range = state.get("feature_range")
    if (
        not isinstance(raw_range, Sequence)
        or isinstance(raw_range, (str, bytes))
        or len(raw_range) != 2
    ):
        raise TypeError("Serialized MinMax feature_range must contain two values.")
    try:
        feature_range = (float(raw_range[0]), float(raw_range[1]))
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "Serialized MinMax feature_range values must be numeric."
        ) from exc
    return cast(
        Normalizer,
        MinMaxNormalizer(
            statistics,
            variable_fields,
            constant_fields,
            feature_range=feature_range,
            min_denom=min_denom,
        ),
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
    "NormalizerState",
    "RMSNormalization",
    "RMSNormalizer",
    "ZScoreNormalization",
    "ZScoreNormalizer",
    "normalizer_from_state_dict",
]
