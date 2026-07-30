"""Metrics and timing summaries shared by benchmark model backends.

This module deliberately has no imports from the public benchmark facade or
the runner.  Model backends can therefore use it without introducing an import
cycle, and the resulting records can be written directly as JSON.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


PRIMARY_METRIC = "valid_normalized_rmse_macro"


def _finite_float(value: Any) -> float | None:
    """Return a regular Python float, preserving missing/non-finite values."""

    if value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if np.isfinite(converted) else None


@dataclass(frozen=True)
class MetricSet:
    """Serializable physical- and normalized-space prediction metrics.

    ``valid_normalized_rmse_macro`` is the equally weighted mean of the
    per-channel physical RMSE divided by that channel's training-set scale.
    Despite its historical ``valid`` prefix, the same field is used for final
    test metrics so tables and Ray Tune results have one stable primary key.
    """

    valid_normalized_rmse_macro: float
    per_field: dict[str, dict[str, float]] = field(default_factory=dict)
    positivity_violation_fraction: float = 0.0
    mass_fraction_consistency: dict[str, float] | None = None
    primary_metric: str = PRIMARY_METRIC

    @property
    def normalized_rmse_macro(self) -> float:
        """Alias for callers that do not want split-specific terminology."""

        return self.valid_normalized_rmse_macro

    def to_dict(self) -> dict[str, Any]:
        """Convert the record to JSON-safe built-in types."""

        result: dict[str, Any] = {
            "primary_metric": self.primary_metric,
            "valid_normalized_rmse_macro": float(
                self.valid_normalized_rmse_macro
            ),
            "normalized_rmse_macro": float(
                self.valid_normalized_rmse_macro
            ),
            "per_field": {
                str(name): {
                    str(metric): float(value)
                    for metric, value in values.items()
                }
                for name, values in self.per_field.items()
            },
            "positivity_violation_fraction": float(
                self.positivity_violation_fraction
            ),
            "mass_fraction_consistency": None,
        }
        if self.mass_fraction_consistency is not None:
            result["mass_fraction_consistency"] = {
                str(name): float(value)
                for name, value in self.mass_fraction_consistency.items()
            }
        return result

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> MetricSet:
        """Restore a metric record written by :meth:`to_dict`."""

        macro = values.get(
            "valid_normalized_rmse_macro",
            values.get("normalized_rmse_macro"),
        )
        if macro is None:
            raise KeyError(
                "Metric data needs 'valid_normalized_rmse_macro' or "
                "'normalized_rmse_macro'."
            )
        per_field = {
            str(name): {
                str(metric): float(value)
                for metric, value in dict(field_values).items()
            }
            for name, field_values in dict(
                values.get("per_field", {})
            ).items()
        }
        consistency = values.get("mass_fraction_consistency")
        return cls(
            valid_normalized_rmse_macro=float(macro),
            per_field=per_field,
            positivity_violation_fraction=float(
                values.get("positivity_violation_fraction", 0.0)
            ),
            mass_fraction_consistency=(
                None
                if consistency is None
                else {
                    str(name): float(value)
                    for name, value in dict(consistency).items()
                }
            ),
            primary_metric=str(
                values.get("primary_metric", PRIMARY_METRIC)
            ),
        )


@dataclass(frozen=True)
class TimingRecord:
    """Serializable wall-clock and inference timing measurements."""

    preprocessing_seconds: float | None = None
    tune_wall_seconds: float | None = None
    final_training_seconds: float | None = None
    inference_latency_median_seconds: float | None = None
    inference_latency_p95_seconds: float | None = None
    inference_throughput_samples_per_second: float | None = None
    solver_wall_seconds: float | None = None
    measurement_count: int = 0
    samples_per_measurement: int = 1
    warmup_repeats: int = 0
    samples: tuple[float, ...] = ()
    extra: dict[str, float | int | None] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the record to JSON-safe built-in types."""

        return {
            "preprocessing_seconds": _finite_float(
                self.preprocessing_seconds
            ),
            "tune_wall_seconds": _finite_float(self.tune_wall_seconds),
            "final_training_seconds": _finite_float(
                self.final_training_seconds
            ),
            "inference_latency_median_seconds": _finite_float(
                self.inference_latency_median_seconds
            ),
            "inference_latency_p95_seconds": _finite_float(
                self.inference_latency_p95_seconds
            ),
            "inference_throughput_samples_per_second": _finite_float(
                self.inference_throughput_samples_per_second
            ),
            "solver_wall_seconds": _finite_float(self.solver_wall_seconds),
            "measurement_count": int(self.measurement_count),
            "samples_per_measurement": int(self.samples_per_measurement),
            "warmup_repeats": int(self.warmup_repeats),
            "samples": [float(value) for value in self.samples],
            "extra": {
                str(name): (
                    None if value is None else float(value)
                )
                for name, value in self.extra.items()
            },
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> TimingRecord:
        """Restore a timing record written by :meth:`to_dict`."""

        timing_fields = (
            "preprocessing_seconds",
            "tune_wall_seconds",
            "final_training_seconds",
            "inference_latency_median_seconds",
            "inference_latency_p95_seconds",
            "inference_throughput_samples_per_second",
            "solver_wall_seconds",
        )
        kwargs = {
            name: _finite_float(values.get(name))
            for name in timing_fields
        }
        samples = tuple(
            float(value)
            for value in values.get("samples", ())
            if _finite_float(value) is not None
        )
        extra = {
            str(name): _finite_float(value)
            for name, value in dict(values.get("extra", {})).items()
        }
        return cls(
            **kwargs,
            measurement_count=int(
                values.get("measurement_count", len(samples))
            ),
            samples_per_measurement=int(
                values.get("samples_per_measurement", 1)
            ),
            warmup_repeats=int(values.get("warmup_repeats", 0)),
            samples=samples,
            extra=extra,
        )


def _as_numpy(value: Any) -> np.ndarray:
    """Convert NumPy-, array-, or torch-like values without requiring torch."""

    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    numpy = getattr(value, "numpy", None)
    if callable(numpy):
        value = numpy()
    return np.asarray(value, dtype=np.float64)


def _channel_arrays(
    values: Any,
    channel_names: Sequence[str] | None,
    channel_axis: int,
) -> tuple[tuple[str, ...], tuple[np.ndarray, ...]]:
    """Return fields in channel-major form from mappings or dense arrays."""

    if isinstance(values, Mapping):
        names = (
            tuple(str(name) for name in channel_names)
            if channel_names is not None
            else tuple(str(name) for name in values)
        )
        missing = [name for name in names if name not in values]
        if missing:
            raise KeyError(f"Prediction data is missing channels: {missing}.")
        return names, tuple(_as_numpy(values[name]) for name in names)

    array = _as_numpy(values)
    if array.ndim == 0:
        array = array.reshape(1)
    axis = int(channel_axis)
    if axis < 0:
        axis += array.ndim
    if axis < 0 or axis >= array.ndim:
        raise ValueError(
            f"channel_axis={channel_axis} is invalid for shape {array.shape}."
        )
    channels = np.moveaxis(array, axis, 0)
    if channel_names is None:
        names = tuple(f"channel_{index}" for index in range(channels.shape[0]))
    else:
        names = tuple(str(name) for name in channel_names)
    if len(names) != channels.shape[0]:
        raise ValueError(
            f"Received {len(names)} channel names for "
            f"{channels.shape[0]} channels."
        )
    return names, tuple(channels[index] for index in range(channels.shape[0]))


def _resolve_scales(
    channel_names: Sequence[str],
    train_scales: Mapping[str, Any] | Sequence[float] | np.ndarray,
    epsilon: float,
) -> dict[str, float]:
    """Resolve and stabilize per-channel training scales."""

    if isinstance(train_scales, Mapping):
        missing = [name for name in channel_names if name not in train_scales]
        if missing:
            raise KeyError(f"Training scales are missing channels: {missing}.")
        raw_values = [train_scales[name] for name in channel_names]
    else:
        raw_values = list(np.asarray(train_scales).reshape(-1))
        if len(raw_values) != len(channel_names):
            raise ValueError(
                f"Received {len(raw_values)} scales for "
                f"{len(channel_names)} channels."
            )

    result: dict[str, float] = {}
    for name, raw_value in zip(channel_names, raw_values, strict=True):
        if isinstance(raw_value, Mapping):
            raw_value = raw_value.get(
                "scale",
                raw_value.get("std", raw_value.get("rms")),
            )
        value = _finite_float(raw_value)
        if value is None:
            raise ValueError(f"Training scale for {name!r} is not finite.")
        result[name] = max(abs(value), epsilon)
    return result


def compute_metrics(
    y_true: Any,
    y_pred: Any,
    channel_names: Sequence[str] | None,
    train_scales: Mapping[str, Any] | Sequence[float] | np.ndarray,
    *,
    channel_axis: int = 1,
    nonnegative_channels: Sequence[str] | None = None,
    mass_fraction_channels: Sequence[str] | None = None,
    positivity_tolerance: float = 0.0,
    mass_fraction_tolerance: float = 1e-3,
    epsilon: float = 1e-12,
) -> MetricSet:
    """Calculate benchmark accuracy and physical-consistency metrics.

    Arrays may use any number of spatial dimensions.  For dense arrays,
    ``channel_axis`` identifies the output channel dimension.  Mappings from
    channel name to arrays are also accepted and ignore ``channel_axis``.

    Parameters
    ----------
    train_scales:
        One physical training-set scale per channel, normally its fitted
        standard deviation.  Zero scales are clamped to ``epsilon``.
    nonnegative_channels:
        Channels on which to evaluate positivity.  ``None`` means all output
        channels; an empty sequence disables the aggregate check.
    mass_fraction_channels:
        Channels that form one mass-fraction vector.  When provided, their
        predicted sum and [0, 1] bounds are evaluated.
    """

    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    if positivity_tolerance < 0 or mass_fraction_tolerance < 0:
        raise ValueError("Consistency tolerances must be non-negative.")

    true_names, true_fields = _channel_arrays(
        y_true, channel_names, channel_axis
    )
    pred_names, pred_fields = _channel_arrays(
        y_pred, true_names, channel_axis
    )
    if true_names != pred_names:
        raise ValueError("True and predicted channel names differ.")
    for name, true, predicted in zip(
        true_names, true_fields, pred_fields, strict=True
    ):
        if true.shape != predicted.shape:
            raise ValueError(
                f"Shape mismatch for {name!r}: "
                f"{true.shape} != {predicted.shape}."
            )
        if not np.isfinite(true).all() or not np.isfinite(predicted).all():
            raise ValueError(
                f"Non-finite truth or prediction values in channel {name!r}."
            )

    scales = _resolve_scales(true_names, train_scales, epsilon)
    nonnegative = (
        set(true_names)
        if nonnegative_channels is None
        else {str(name) for name in nonnegative_channels}
    )
    unknown_nonnegative = nonnegative.difference(true_names)
    if unknown_nonnegative:
        raise KeyError(
            "Unknown nonnegative channels: "
            f"{sorted(unknown_nonnegative)}."
        )

    per_field: dict[str, dict[str, float]] = {}
    normalized_rmses: list[float] = []
    negative_values = 0
    positivity_values = 0
    predicted_by_name: dict[str, np.ndarray] = {}

    for name, true, predicted in zip(
        true_names, true_fields, pred_fields, strict=True
    ):
        error = predicted - true
        rmse = float(np.sqrt(np.mean(np.square(error))))
        true_norm = float(np.sqrt(np.sum(np.square(true))))
        error_norm = float(np.sqrt(np.sum(np.square(error))))
        if error_norm == 0.0:
            relative_l2 = 0.0
        else:
            relative_l2 = error_norm / max(true_norm, epsilon)
        normalized_rmse = rmse / scales[name]
        field_negative_fraction = float(
            np.mean(predicted < -positivity_tolerance)
        )
        per_field[name] = {
            "rmse": rmse,
            "relative_l2": relative_l2,
            "max_abs": float(np.max(np.abs(error), initial=0.0)),
            "normalized_rmse": normalized_rmse,
            "positivity_violation_fraction": field_negative_fraction,
            "train_scale": scales[name],
        }
        normalized_rmses.append(normalized_rmse)
        predicted_by_name[name] = predicted
        if name in nonnegative:
            negative_values += int(
                np.count_nonzero(predicted < -positivity_tolerance)
            )
            positivity_values += predicted.size

    positivity_fraction = (
        negative_values / positivity_values if positivity_values else 0.0
    )

    mass_consistency: dict[str, float] | None = None
    if mass_fraction_channels is not None:
        fraction_names = tuple(str(name) for name in mass_fraction_channels)
        if not fraction_names:
            raise ValueError("mass_fraction_channels may not be empty.")
        missing = set(fraction_names).difference(predicted_by_name)
        if missing:
            raise KeyError(
                f"Unknown mass-fraction channels: {sorted(missing)}."
            )
        fraction_fields = [
            predicted_by_name[name] for name in fraction_names
        ]
        reference_shape = fraction_fields[0].shape
        if any(field.shape != reference_shape for field in fraction_fields):
            raise ValueError(
                "Mass-fraction channels must have identical shapes."
            )
        fraction_array = np.stack(fraction_fields, axis=0)
        fraction_sum = np.sum(fraction_array, axis=0)
        sum_error = fraction_sum - 1.0
        out_of_bounds = np.logical_or(
            fraction_array < -mass_fraction_tolerance,
            fraction_array > 1.0 + mass_fraction_tolerance,
        )
        mass_consistency = {
            "mean_sum": float(np.mean(fraction_sum)),
            "mean_absolute_sum_error": float(np.mean(np.abs(sum_error))),
            "rmse_sum_error": float(
                np.sqrt(np.mean(np.square(sum_error)))
            ),
            "max_absolute_sum_error": float(
                np.max(np.abs(sum_error), initial=0.0)
            ),
            "sum_violation_fraction": float(
                np.mean(np.abs(sum_error) > mass_fraction_tolerance)
            ),
            "out_of_bounds_fraction": float(np.mean(out_of_bounds)),
        }

    return MetricSet(
        valid_normalized_rmse_macro=float(np.mean(normalized_rmses)),
        per_field=per_field,
        positivity_violation_fraction=float(positivity_fraction),
        mass_fraction_consistency=mass_consistency,
    )


def summarize_timings(
    latencies_seconds: Sequence[float] | np.ndarray,
    *,
    samples_per_measurement: int = 1,
    sample_count: int | None = None,
    warmup_repeats: int = 0,
    preprocessing_seconds: float | None = None,
    tune_wall_seconds: float | None = None,
    final_training_seconds: float | None = None,
    solver_wall_seconds: float | None = None,
    extra: Mapping[str, float | int | None] | None = None,
) -> TimingRecord:
    """Summarize repeated inference timings while ignoring invalid samples.

    Non-finite and negative measurements are discarded.  An empty collection
    yields a record with missing latency/throughput values instead of raising,
    which lets failed or interrupted benchmark cells still emit artifacts.
    """

    if sample_count is not None:
        if samples_per_measurement != 1:
            raise ValueError(
                "Use either sample_count or samples_per_measurement, not both."
            )
        samples_per_measurement = sample_count
    if samples_per_measurement < 1:
        raise ValueError("samples_per_measurement must be positive.")
    if warmup_repeats < 0:
        raise ValueError("warmup_repeats must be non-negative.")

    raw = np.asarray(latencies_seconds, dtype=np.float64).reshape(-1)
    valid = raw[np.logical_and(np.isfinite(raw), raw >= 0.0)]
    if valid.size:
        median = float(np.median(valid))
        p95 = float(np.percentile(valid, 95))
        total_time = float(np.sum(valid))
        throughput = (
            float(valid.size * samples_per_measurement / total_time)
            if total_time > 0.0
            else None
        )
    else:
        median = None
        p95 = None
        throughput = None

    return TimingRecord(
        preprocessing_seconds=_finite_float(preprocessing_seconds),
        tune_wall_seconds=_finite_float(tune_wall_seconds),
        final_training_seconds=_finite_float(final_training_seconds),
        inference_latency_median_seconds=median,
        inference_latency_p95_seconds=p95,
        inference_throughput_samples_per_second=throughput,
        solver_wall_seconds=_finite_float(solver_wall_seconds),
        measurement_count=int(valid.size),
        samples_per_measurement=int(samples_per_measurement),
        warmup_repeats=int(warmup_repeats),
        samples=tuple(float(value) for value in valid),
        extra={
            str(name): _finite_float(value)
            for name, value in (extra or {}).items()
        },
    )


# A concise alias is convenient in model backends and keeps compatibility with
# early benchmark prototypes.
compute_metric_set = compute_metrics
summarize_timing = summarize_timings


__all__ = [
    "PRIMARY_METRIC",
    "MetricSet",
    "TimingRecord",
    "compute_metric_set",
    "compute_metrics",
    "summarize_timing",
    "summarize_timings",
]
