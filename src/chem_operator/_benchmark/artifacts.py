"""Versioned artifact and checkpoint helpers for benchmark runs."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any, Mapping

import numpy as np
import torch


SCHEMA_VERSION = "chem-operator-benchmark-v1"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_json(path: str | Path, values: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(values), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class CheckpointBundle:
    """Contents needed to restore a trained model and preprocessing."""

    model_state: Mapping[str, Any]
    preprocessing_state: Mapping[str, Any]
    problem_spec: Mapping[str, Any]
    model_spec: Mapping[str, Any]
    best_config: Mapping[str, Any]
    dataset_fingerprint: str
    schema: str = SCHEMA_VERSION

    def save(self, directory: str | Path) -> dict[str, Path]:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        model_path = root / "model.pt"
        preprocessing_path = root / "preprocessing.pt"
        torch.save(dict(self.model_state), model_path)
        torch.save(
            {
                **dict(self.preprocessing_state),
                "schema": self.schema,
                "problem_spec": dict(self.problem_spec),
                "model_spec": dict(self.model_spec),
                "best_config": dict(self.best_config),
                "dataset_fingerprint": self.dataset_fingerprint,
            },
            preprocessing_path,
        )
        return {
            "model": model_path,
            "preprocessing": preprocessing_path,
        }


@dataclass
class ArtifactManifest:
    """JSON-safe provenance for a benchmark cell."""

    run_id: str
    problem: str
    model: str
    seed: int
    dataset_fingerprint: str
    files: dict[str, str] = field(default_factory=dict)
    schema: str = SCHEMA_VERSION
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    environment: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        problem: str,
        model: str,
        seed: int,
        dataset_fingerprint: str,
    ) -> "ArtifactManifest":
        package_names = (
            "cantera",
            "deepxde",
            "neuraloperator",
            "numpy",
            "optuna",
            "ray",
            "torch",
        )
        versions: dict[str, str] = {}
        for name in package_names:
            try:
                versions[name] = importlib_metadata.version(name)
            except importlib_metadata.PackageNotFoundError:
                continue
        repository = Path(__file__).resolve().parents[3]
        git_commit: str | None = None
        head = repository / ".git" / "HEAD"
        if head.is_file():
            head_value = head.read_text(encoding="utf-8").strip()
            if head_value.startswith("ref: "):
                reference = repository / ".git" / head_value[5:]
                if reference.is_file():
                    git_commit = reference.read_text(encoding="utf-8").strip()
            else:
                git_commit = head_value
        environment = {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "packages": versions,
            "git_commit": git_commit,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
            "pid": os.getpid(),
        }
        return cls(
            run_id=run_id,
            problem=problem,
            model=model,
            seed=seed,
            dataset_fingerprint=dataset_fingerprint,
            environment=environment,
        )

    def save(self, path: str | Path) -> Path:
        return write_json(path, asdict(self))


def write_history(
    path: str | Path,
    history: Mapping[str, list[float]],
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    names = list(history)
    rows = max((len(history[name]) for name in names), default=0)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", *names])
        writer.writeheader()
        for index in range(rows):
            writer.writerow(
                {
                    "epoch": index + 1,
                    **{
                        name: (
                            history[name][index]
                            if index < len(history[name])
                            else ""
                        )
                        for name in names
                    },
                }
            )
    return destination


def save_pod(path: str | Path, pod: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        mean=np.asarray(pod.mean),
        basis=np.asarray(pod.basis),
        explained_variance_ratio=np.asarray(pod.explained_variance_ratio),
        cumulative_explained_variance=np.asarray(
            pod.cumulative_explained_variance
        ),
        output_shape=np.asarray(pod.output_shape, dtype=np.int64),
    )
    return destination


def load_torch(path: str | Path) -> dict[str, Any]:
    """Load a tensor/scalar-only checkpoint produced by this package."""

    return torch.load(Path(path), map_location="cpu", weights_only=True)


def make_run_id(prefix: str = "benchmark") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{stamp}"


__all__ = [
    "ArtifactManifest",
    "CheckpointBundle",
    "SCHEMA_VERSION",
    "load_torch",
    "make_run_id",
    "read_json",
    "save_pod",
    "write_history",
    "write_json",
]
