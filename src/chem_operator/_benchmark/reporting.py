"""Aggregate benchmark artifacts into comparison tables and figures.

The reporting layer accepts either result-like Python objects or a benchmark
run directory.  It intentionally uses structural conversion (mappings,
dataclasses, or ``to_dict``) instead of importing ``BenchmarkResult`` so the
public facade and runner remain free to import this module.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
import json
import os
from pathlib import Path
import re
from typing import Any

_matplotlib_config = os.environ.get("MPLCONFIGDIR")
if (
    not _matplotlib_config
    or not Path(_matplotlib_config).exists()
    or not os.access(_matplotlib_config, os.W_OK)
):
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd


PRIMARY_METRIC_COLUMN = "valid_normalized_rmse_macro"
_BASE_COLUMNS = (
    "run_id",
    "problem",
    "model",
    "status",
    PRIMARY_METRIC_COLUMN,
    "preprocessing_seconds",
    "tune_wall_seconds",
    "final_training_seconds",
    "inference_latency_median_seconds",
    "inference_latency_p95_seconds",
    "inference_throughput_samples_per_second",
    "solver_wall_seconds",
    "parameter_count",
    "checkpoint_size_bytes",
    "artifact_dir",
    "error",
    "metrics_json",
    "timings_json",
)
_SUCCESS_STATUSES = {
    "",
    "complete",
    "completed",
    "ok",
    "success",
    "succeeded",
}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "value"):
        return value.value
    return str(value)


def _as_mapping(value: Any) -> dict[str, Any]:
    """Convert a result component to a shallow regular mapping."""

    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        if isinstance(converted, Mapping):
            return dict(converted)
    if is_dataclass(value):
        return asdict(value)
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, Mapping):
        return {
            key: item
            for key, item in attributes.items()
            if not str(key).startswith("_")
        }
    return {}


def _nested_mapping(mapping: Mapping[str, Any], key: str) -> dict[str, Any]:
    return _as_mapping(mapping.get(key))


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _find_scalar(
    mapping: Mapping[str, Any],
    keys: Sequence[str],
) -> Any:
    """Find a scalar under preferred keys, descending into nested mappings."""

    for key in keys:
        if key in mapping and not isinstance(mapping[key], Mapping):
            return mapping[key]
    for preferred_container in ("test", "validation", "valid", "summary"):
        nested = _nested_mapping(mapping, preferred_container)
        found = _find_scalar(nested, keys) if nested else None
        if found is not None:
            return found
    for value in mapping.values():
        nested = _as_mapping(value)
        if nested:
            found = _find_scalar(nested, keys)
            if found is not None:
                return found
    return None


def _identity(value: Any) -> str | None:
    """Extract a registry name from a scalar or small specification mapping."""

    if value is None:
        return None
    if isinstance(value, Mapping) or is_dataclass(value):
        mapping = _as_mapping(value)
        for key in ("name", "key", "id", "problem", "model"):
            if key in mapping:
                return _identity(mapping[key])
        return None
    if hasattr(value, "value"):
        value = value.value
    text = str(value).strip()
    return text or None


def _status_text(value: Any, *, has_metric: bool) -> str:
    status = _identity(value)
    if status is None:
        return "success" if has_metric else "failed"
    return status.lower()


def _field_metric_columns(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Expose per-field scalars in the CSV without fixing the channel set."""

    per_field = _nested_mapping(metrics, "per_field")
    columns: dict[str, float] = {}
    for field_name, raw_values in per_field.items():
        values = _as_mapping(raw_values)
        safe_field = _slug(str(field_name)).replace("-", "_")
        for metric_name, raw_value in values.items():
            value = _finite(raw_value)
            if value is not None:
                safe_metric = _slug(str(metric_name)).replace("-", "_")
                columns[f"field_{safe_field}_{safe_metric}"] = value
    return columns


def result_to_row(result: Any) -> dict[str, Any]:
    """Normalize one ``BenchmarkResult``-like object for tabular reporting."""

    raw = _as_mapping(result)
    config = _nested_mapping(raw, "config")
    metrics = _nested_mapping(raw, "metrics")
    if not metrics:
        metrics = _nested_mapping(raw, "metric_set")
    timings = _nested_mapping(raw, "timings")
    if not timings:
        timings = _nested_mapping(raw, "timing")

    metric = _finite(
        _find_scalar(
            metrics,
            (
                PRIMARY_METRIC_COLUMN,
                "normalized_rmse_macro",
                "test_normalized_rmse_macro",
                "primary_metric_value",
            ),
        )
    )
    if metric is None:
        metric = _finite(
            _find_scalar(
                raw,
                (
                    PRIMARY_METRIC_COLUMN,
                    "normalized_rmse_macro",
                    "test_normalized_rmse_macro",
                ),
            )
        )

    problem = _identity(
        raw.get("problem", raw.get("problem_name", config.get("problem")))
    )
    model = _identity(
        raw.get("model", raw.get("model_name", config.get("model")))
    )
    artifacts = _nested_mapping(raw, "artifact_paths")
    if not artifacts:
        artifacts = _nested_mapping(raw, "artifacts")
    artifact_dir = raw.get(
        "artifact_dir",
        raw.get(
            "run_directory",
            raw.get(
                "output_dir",
                artifacts.get("root", artifacts.get("directory")),
            ),
        ),
    )
    if artifact_dir is None:
        candidate = artifacts.get("model") or artifacts.get("checkpoint")
        if candidate is not None:
            artifact_dir = str(Path(candidate).parent)

    status = _status_text(raw.get("status"), has_metric=metric is not None)
    row: dict[str, Any] = {
        "run_id": _identity(
            raw.get("run_id", config.get("run_id"))
        ),
        "problem": problem,
        "model": model,
        "status": status,
        PRIMARY_METRIC_COLUMN: metric,
        "preprocessing_seconds": _finite(
            _find_scalar(timings, ("preprocessing_seconds",))
        ),
        "tune_wall_seconds": _finite(
            _find_scalar(
                timings,
                ("tune_wall_seconds", "tuning_seconds", "tune_seconds"),
            )
        ),
        "final_training_seconds": _finite(
            _find_scalar(
                timings,
                ("final_training_seconds", "training_seconds"),
            )
        ),
        "inference_latency_median_seconds": _finite(
            _find_scalar(
                timings,
                (
                    "inference_latency_median_seconds",
                    "latency_median_seconds",
                    "median_seconds",
                ),
            )
        ),
        "inference_latency_p95_seconds": _finite(
            _find_scalar(
                timings,
                (
                    "inference_latency_p95_seconds",
                    "latency_p95_seconds",
                    "p95_seconds",
                ),
            )
        ),
        "inference_throughput_samples_per_second": _finite(
            _find_scalar(
                timings,
                (
                    "inference_throughput_samples_per_second",
                    "throughput_samples_per_second",
                ),
            )
        ),
        "solver_wall_seconds": _finite(
            _find_scalar(timings, ("solver_wall_seconds", "solver_seconds"))
        ),
        "parameter_count": _finite(
            raw.get(
                "parameter_count",
                _find_scalar(metrics, ("parameter_count", "n_parameters")),
            )
        ),
        "checkpoint_size_bytes": _finite(
            raw.get(
                "checkpoint_size_bytes",
                _find_scalar(metrics, ("checkpoint_size_bytes", "model_bytes")),
            )
        ),
        "artifact_dir": (
            None if artifact_dir is None else str(artifact_dir)
        ),
        "error": raw.get(
            "error",
            raw.get("failure_reason", raw.get("message")),
        ),
        "metrics_json": json.dumps(
            metrics, default=_json_default, sort_keys=True
        ),
        "timings_json": json.dumps(
            timings, default=_json_default, sort_keys=True
        ),
    }
    row.update(_field_metric_columns(metrics))
    return row


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _merge_missing(
    destination: dict[str, Any],
    source: Mapping[str, Any],
) -> None:
    for key, value in source.items():
        if key not in destination or destination[key] is None:
            destination[key] = value


def _directory_records(root: Path) -> list[dict[str, Any]]:
    """Reconstruct result objects from per-cell artifact directories."""

    artifact_names = (
        "result.json",
        "resolved_config.json",
        "manifest.json",
        "metrics.json",
        "timings.json",
    )
    directories: set[Path] = set()
    for name in artifact_names:
        directories.update(
            path.parent
            for path in root.rglob(name)
            if not {
                "summary",
                "ray",
            }.intersection(path.relative_to(root).parts)
        )

    records: list[dict[str, Any]] = []
    for directory in sorted(directories):
        record = _load_json(directory / "result.json")
        resolved = _load_json(directory / "resolved_config.json")
        manifest = _load_json(directory / "manifest.json")
        _merge_missing(record, resolved)
        _merge_missing(record, manifest)

        metrics = _load_json(directory / "metrics.json")
        timings = _load_json(directory / "timings.json")
        if metrics:
            record["metrics"] = metrics
        if timings:
            record["timings"] = timings
        record["artifact_dir"] = str(directory)

        relative_parts = directory.relative_to(root).parts
        if len(relative_parts) >= 2:
            record.setdefault("problem", relative_parts[-2])
            record.setdefault("model", relative_parts[-1])
        if len(relative_parts) >= 3:
            record.setdefault("run_id", relative_parts[-3])
        if record:
            records.append(record)

    if records:
        return records

    csv_path = root if root.is_file() else root / "results.csv"
    if not csv_path.is_file() and root.is_dir():
        csv_path = root / "summary" / "results.csv"
    if csv_path.is_file():
        return pd.read_csv(csv_path).replace({np.nan: None}).to_dict(
            orient="records"
        )
    return []


def collect_result_rows(
    results_or_run_directory: Any,
) -> list[dict[str, Any]]:
    """Collect normalized rows without generating plots or writing files."""

    if isinstance(results_or_run_directory, (str, os.PathLike)):
        records: Iterable[Any] = _directory_records(
            Path(results_or_run_directory)
        )
    elif isinstance(results_or_run_directory, pd.DataFrame):
        records = results_or_run_directory.to_dict(orient="records")
    elif (
        isinstance(results_or_run_directory, Mapping)
        or is_dataclass(results_or_run_directory)
        or callable(getattr(results_or_run_directory, "to_dict", None))
    ):
        records = (results_or_run_directory,)
    else:
        records = results_or_run_directory
    return [result_to_row(record) for record in records]


def _complete_rows(
    rows: list[dict[str, Any]],
    expected_problems: Sequence[str] | None,
    expected_models: Sequence[str] | None,
) -> list[dict[str, Any]]:
    if expected_problems is None or expected_models is None:
        return rows
    completed = list(rows)
    present = {
        (str(row.get("problem")), str(row.get("model")))
        for row in rows
    }
    for problem in expected_problems:
        for model in expected_models:
            key = (str(problem), str(model))
            if key not in present:
                completed.append(
                    result_to_row(
                        {
                            "problem": problem,
                            "model": model,
                            "status": "missing",
                            "error": "No result artifact was found.",
                        }
                    )
                )
    return completed


def _ordered_unique(
    values: Iterable[Any],
    preferred: Sequence[str] | None,
) -> list[str]:
    if preferred is not None:
        return [str(value) for value in preferred]
    result: list[str] = []
    for value in values:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        text = str(value)
        if text not in result:
            result.append(text)
    return result


def _successful(row: Mapping[str, Any]) -> bool:
    return str(row.get("status", "")).lower() in _SUCCESS_STATUSES


def _cell_value(
    rows: Sequence[Mapping[str, Any]],
    problem: str,
    model: str,
    column: str,
) -> tuple[float | None, str]:
    matching = [
        row
        for row in rows
        if str(row.get("problem")) == problem
        and str(row.get("model")) == model
    ]
    values = [
        value
        for row in matching
        if _successful(row)
        and (value := _finite(row.get(column))) is not None
    ]
    if values:
        return float(np.mean(values)), "success"
    if matching:
        statuses = {str(row.get("status", "")).lower() for row in matching}
        if "missing" in statuses:
            return None, "missing"
        return None, "failed"
    return None, "missing"


def _matrix(
    rows: Sequence[Mapping[str, Any]],
    problems: Sequence[str],
    models: Sequence[str],
    column: str,
) -> tuple[np.ndarray, list[list[str]]]:
    values = np.full((len(problems), len(models)), np.nan, dtype=float)
    statuses: list[list[str]] = []
    for row_index, problem in enumerate(problems):
        status_row: list[str] = []
        for column_index, model in enumerate(models):
            value, status = _cell_value(
                rows, problem, model, column
            )
            if value is not None:
                values[row_index, column_index] = value
            status_row.append(status)
        statuses.append(status_row)
    return values, statuses


def _format_value(value: float, value_format: str) -> str:
    try:
        return value_format.format(value)
    except (ValueError, IndexError):
        return f"{value:.3g}"


def _plot_heatmap(
    rows: Sequence[Mapping[str, Any]],
    problems: Sequence[str],
    models: Sequence[str],
    column: str,
    output_path: Path,
    *,
    title: str,
    colorbar_label: str,
    scale: float = 1.0,
    value_format: str = "{:.3g}",
) -> None:
    values, statuses = _matrix(rows, problems, models, column)
    values = values * scale
    figure_width = max(5.5, 1.25 * max(len(models), 1) + 2.5)
    figure_height = max(3.5, 0.65 * max(len(problems), 1) + 1.8)
    fig, axis = plt.subplots(
        figsize=(figure_width, figure_height), constrained_layout=True
    )
    if not problems or not models:
        axis.axis("off")
        axis.text(
            0.5,
            0.5,
            "No benchmark results",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        fig.suptitle(title)
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return

    cmap = matplotlib.colormaps["viridis"].with_extremes(bad="#e5e7eb")
    masked = np.ma.masked_invalid(values)
    image = axis.imshow(masked, aspect="auto", cmap=cmap)
    if np.isfinite(values).any():
        colorbar = fig.colorbar(image, ax=axis, shrink=0.85)
        colorbar.set_label(colorbar_label)
    axis.set_xticks(range(len(models)), labels=models, rotation=25, ha="right")
    axis.set_yticks(range(len(problems)), labels=problems)
    axis.set_xlabel("Model")
    axis.set_ylabel("Problem")
    axis.set_title(title)

    finite_values = values[np.isfinite(values)]
    threshold = (
        float(np.nanmin(finite_values) + np.nanmax(finite_values)) / 2.0
        if finite_values.size
        else 0.0
    )
    for row_index in range(len(problems)):
        for column_index in range(len(models)):
            value = values[row_index, column_index]
            if np.isfinite(value):
                label = _format_value(float(value), value_format)
                color = "white" if value <= threshold else "black"
            else:
                label = (
                    "FAIL"
                    if statuses[row_index][column_index] == "failed"
                    else "—"
                )
                color = "#7f1d1d" if label == "FAIL" else "#4b5563"
            axis.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_accuracy_vs_latency(
    rows: Sequence[Mapping[str, Any]],
    models: Sequence[str],
    output_path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    plotted = False
    colors = matplotlib.colormaps["tab10"](
        np.linspace(0.0, 1.0, max(len(models), 1))
    )
    for index, model in enumerate(models):
        model_rows = []
        for row in rows:
            accuracy = _finite(row.get(PRIMARY_METRIC_COLUMN))
            latency = _finite(
                row.get("inference_latency_median_seconds")
            )
            if (
                str(row.get("model")) == model
                and _successful(row)
                and accuracy is not None
                and latency is not None
            ):
                model_rows.append((row, accuracy, latency * 1_000.0))
        if not model_rows:
            continue
        plotted = True
        axis.scatter(
            [item[2] for item in model_rows],
            [item[1] for item in model_rows],
            s=52,
            label=model,
            color=colors[index],
            alpha=0.85,
        )
        for row, accuracy, latency_ms in model_rows:
            axis.annotate(
                str(row.get("problem", "")),
                (latency_ms, accuracy),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=7,
            )
    if plotted:
        axis.legend(title="Model", frameon=False)
    else:
        axis.text(
            0.5,
            0.5,
            "No successful cells with latency and accuracy",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    axis.set_xlabel("Median inference latency (ms)")
    axis.set_ylabel("Macro normalized RMSE (lower is better)")
    axis.set_title("Accuracy versus inference latency")
    axis.grid(alpha=0.2)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_problem_comparison(
    rows: Sequence[Mapping[str, Any]],
    problem: str,
    models: Sequence[str],
    output_path: Path,
) -> None:
    columns = (
        (PRIMARY_METRIC_COLUMN, 1.0, "Normalized RMSE"),
        ("final_training_seconds", 1.0, "Final training (s)"),
        ("inference_latency_median_seconds", 1_000.0, "Latency (ms)"),
    )
    fig, axes = plt.subplots(
        1, 3, figsize=(10.5, 3.7), constrained_layout=True
    )
    positions = np.arange(len(models))
    for axis, (column, scale, title) in zip(axes, columns, strict=True):
        heights: list[float] = []
        statuses: list[str] = []
        for model in models:
            value, status = _cell_value(rows, problem, model, column)
            heights.append(np.nan if value is None else value * scale)
            statuses.append(status)
        finite = np.asarray(heights, dtype=float)
        display = np.nan_to_num(finite, nan=0.0)
        colors = [
            "#4c78a8" if np.isfinite(value) else "#d1d5db"
            for value in finite
        ]
        bars = axis.bar(positions, display, color=colors)
        for bar, value, status in zip(
            bars, finite, statuses, strict=True
        ):
            if not np.isfinite(value):
                bar.set_hatch("//")
                axis.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    0.0,
                    "FAIL" if status == "failed" else "—",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=7,
                )
        axis.set_xticks(positions, labels=models, rotation=25, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle(f"{problem}: model comparison")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip()).strip("-").lower()
    return slug or "unnamed"


def _default_output_directory(source: Any) -> Path:
    if isinstance(source, (str, os.PathLike)):
        path = Path(source)
        if path.is_file():
            return path.parent / "summary"
        if path.name == "summary":
            return path
        return path / "summary"
    return Path("artifacts") / "benchmarks" / "summary"


def build_report(
    results_or_run_directory: Any,
    output_dir: str | os.PathLike[str] | None = None,
    *,
    expected_problems: Sequence[str] | None = None,
    expected_models: Sequence[str] | None = None,
) -> dict[str, Path | tuple[Path, ...]]:
    """Write the benchmark matrix CSV and all comparison figures.

    Parameters
    ----------
    results_or_run_directory:
        An iterable of result-like objects, one result-like object, a pandas
        frame, or a benchmark run directory containing per-cell JSON files.
    output_dir:
        Report destination.  A directory input defaults to its ``summary``
        child; in-memory results default to ``artifacts/benchmarks/summary``.
    expected_problems, expected_models:
        Optional registry order.  Supplying both also materializes absent
        combinations as ``status="missing"`` rows in ``results.csv``.
    """

    rows = collect_result_rows(results_or_run_directory)
    rows = _complete_rows(rows, expected_problems, expected_models)
    problems = _ordered_unique(
        (row.get("problem") for row in rows), expected_problems
    )
    models = _ordered_unique(
        (row.get("model") for row in rows), expected_models
    )

    destination = (
        Path(output_dir)
        if output_dir is not None
        else _default_output_directory(results_or_run_directory)
    )
    destination.mkdir(parents=True, exist_ok=True)
    per_problem_directory = destination / "per_problem"
    per_problem_directory.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(rows)
    for column in _BASE_COLUMNS:
        if column not in frame:
            frame[column] = pd.Series(dtype="object")
    extra_columns = sorted(
        column for column in frame.columns if column not in _BASE_COLUMNS
    )
    frame = frame[[*_BASE_COLUMNS, *extra_columns]]
    results_csv = destination / "results.csv"
    frame.to_csv(results_csv, index=False)

    figure_specs = (
        (
            "accuracy_heatmap",
            PRIMARY_METRIC_COLUMN,
            "Model accuracy by reactor problem",
            "Macro normalized RMSE",
            1.0,
            "{:.3g}",
        ),
        (
            "tune_time_heatmap",
            "tune_wall_seconds",
            "Ray Tune wall time by reactor problem",
            "Seconds",
            1.0,
            "{:.1f}",
        ),
        (
            "training_time_heatmap",
            "final_training_seconds",
            "Final training time by reactor problem",
            "Seconds",
            1.0,
            "{:.1f}",
        ),
        (
            "inference_latency_heatmap",
            "inference_latency_median_seconds",
            "Median inference latency by reactor problem",
            "Milliseconds",
            1_000.0,
            "{:.2f}",
        ),
    )
    artifacts: dict[str, Path | tuple[Path, ...]] = {
        "results_csv": results_csv
    }
    for (
        artifact_name,
        column,
        title,
        colorbar_label,
        scale,
        value_format,
    ) in figure_specs:
        output_path = destination / f"{artifact_name}.png"
        _plot_heatmap(
            rows,
            problems,
            models,
            column,
            output_path,
            title=title,
            colorbar_label=colorbar_label,
            scale=scale,
            value_format=value_format,
        )
        artifacts[artifact_name] = output_path

    accuracy_latency_path = destination / "accuracy_vs_latency.png"
    _plot_accuracy_vs_latency(rows, models, accuracy_latency_path)
    artifacts["accuracy_vs_latency"] = accuracy_latency_path

    per_problem_paths: list[Path] = []
    for problem in problems:
        output_path = (
            per_problem_directory
            / f"{_slug(problem)}_model_comparison.png"
        )
        _plot_problem_comparison(rows, problem, models, output_path)
        per_problem_paths.append(output_path)
    artifacts["per_problem"] = tuple(per_problem_paths)
    return artifacts


__all__ = [
    "PRIMARY_METRIC_COLUMN",
    "build_report",
    "collect_result_rows",
    "result_to_row",
]
