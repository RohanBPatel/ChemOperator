"""Unit tests for benchmark accuracy and timing summaries."""

from __future__ import annotations

import json

import numpy as np
import pytest

from chem_operator._benchmark.metrics import (
    MetricSet,
    TimingRecord,
    compute_metrics,
    summarize_timings,
)


def test_macro_normalized_rmse_and_physical_field_metrics() -> None:
    truth = np.array([[[1.0, 3.0], [0.25, 0.75]]])
    prediction = np.array([[[2.0, 1.0], [0.20, 0.70]]])

    metrics = compute_metrics(
        truth,
        prediction,
        ("temperature", "fuel"),
        {"temperature": 2.0, "fuel": 0.5},
    )

    temperature_normalized = np.sqrt(2.5) / 2.0
    fuel_normalized = 0.05 / 0.5
    assert metrics.valid_normalized_rmse_macro == pytest.approx(
        (temperature_normalized + fuel_normalized) / 2.0
    )
    assert metrics.per_field["temperature"]["rmse"] == pytest.approx(
        np.sqrt(2.5)
    )
    assert metrics.per_field["temperature"]["relative_l2"] == pytest.approx(
        np.sqrt(5.0) / np.sqrt(10.0)
    )
    assert metrics.per_field["temperature"]["max_abs"] == 2.0
    assert metrics.positivity_violation_fraction == 0.0


def test_mapping_inputs_positivity_and_mass_fraction_consistency() -> None:
    truth = {
        "X_CH4": np.array([[0.2, 0.4]]),
        "X_H2": np.array([[0.8, 0.6]]),
    }
    # Use the reverse insertion order to exercise name-based alignment.
    prediction = {
        "X_H2": np.array([[1.1, 0.7]]),
        "X_CH4": np.array([[-0.1, 0.4]]),
    }

    metrics = compute_metrics(
        truth,
        prediction,
        ("X_CH4", "X_H2"),
        {"X_CH4": 0.2, "X_H2": 0.2},
        mass_fraction_channels=("X_CH4", "X_H2"),
    )

    assert metrics.positivity_violation_fraction == pytest.approx(0.25)
    assert metrics.mass_fraction_consistency is not None
    assert metrics.mass_fraction_consistency[
        "mean_absolute_sum_error"
    ] == pytest.approx(0.05)
    assert metrics.mass_fraction_consistency[
        "out_of_bounds_fraction"
    ] == pytest.approx(0.5)


def test_metric_set_json_round_trip() -> None:
    metrics = compute_metrics(
        np.array([[[0.0, 0.0]]]),
        np.array([[[0.0, 1.0]]]),
        ("species",),
        (0.0,),
    )
    encoded = json.loads(json.dumps(metrics.to_dict()))
    restored = MetricSet.from_dict(encoded)

    assert np.isfinite(restored.valid_normalized_rmse_macro)
    assert restored == metrics


def test_timing_summary_filters_invalid_measurements_and_round_trips() -> None:
    timing = summarize_timings(
        [0.1, np.nan, -1.0, 0.3],
        samples_per_measurement=4,
        warmup_repeats=10,
        tune_wall_seconds=12.0,
    )

    assert timing.measurement_count == 2
    assert timing.inference_latency_median_seconds == pytest.approx(0.2)
    assert timing.inference_latency_p95_seconds == pytest.approx(0.29)
    assert timing.inference_throughput_samples_per_second == pytest.approx(
        20.0
    )
    assert timing.samples == (0.1, 0.3)
    restored = TimingRecord.from_dict(
        json.loads(json.dumps(timing.to_dict()))
    )
    assert restored == timing


def test_empty_timing_measurements_produce_missing_summary() -> None:
    timing = summarize_timings([np.nan, -0.1])

    assert timing.measurement_count == 0
    assert timing.inference_latency_median_seconds is None
    assert timing.inference_latency_p95_seconds is None
    assert timing.inference_throughput_samples_per_second is None
