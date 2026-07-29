from __future__ import annotations
from pathlib import Path
from collections.abc import Iterable, Sequence, Callable, Mapping
from typing import Any, Literal, Protocol
from itertools import product
from dataclasses import dataclass
import json
from copy import deepcopy

from tqdm import tqdm, trange

import h5py

import cantera as ct
import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd

import torch
from torch.utils.data import Dataset

from chem_operator.utils import get_mechanism_file
from chem_operator.sampling import ParameterSpec

@dataclass
class CaseParameters:
    initial_conditions: dict | None = None
    boundary_conditions: dict | None = None
    controls: dict | None = None
    geometry: dict | None = None
    physical_parameters: dict | None = None
    solver_parameters: dict | None = None
    mechanism_parameters: dict | None = None

@dataclass
class SimulationRecord:
    # Examples: time, z, x/y grid
    coordinates: dict[str, np.ndarray]

    # Arrays sharing one or more coordinate dimensions
    fields: dict[str, np.ndarray]
    
    # Values constant over the record
    constants: dict[str, float | np.ndarray]

    # Names, units, phase information, provenance, ...
    metadata: dict

    def to_SolutionArray(self) -> ct.SolutionArray:
        mechanism = self.metadata.get("mechanism")
        if mechanism is None:
            raise ValueError("metadata must contain 'mechanism'.")

        gas = ct.Solution(get_mechanism_file(mechanism))

        times = self.coordinates.get("t")
        if times is None:
            raise ValueError("coordinates must contain time 't'.")

        T = self.fields.get("T")
        P = self.fields.get("P")
        X = self.fields.get("X")
        Y = self.fields.get("Y")

        if T is None or P is None:
            raise ValueError("fields must contain 'T' and 'P'.")

        if X is None and Y is None:
            raise ValueError("fields must contain either 'X' or 'Y'.")

        states = ct.SolutionArray(gas, extra=["t"])

        for i, t in enumerate(times):
            if X is not None:
                gas.TPX = T[i], P[i], X[i]
            else:
                gas.TPY = T[i], P[i], Y[i]

            states.append(gas.state, t=t)

        return states

class CaseSimulator(Protocol):
    name: str

    @property
    def parameter_space(self) -> Mapping[str, ParameterSpec] | None:
        ...

    def make_case(
        self,
        params: Mapping[str, Any],
    ) -> CaseParameters:
        ...

    def run_case(
        self,
        case: CaseParameters,
    ) -> SimulationRecord:
        ...

class SimulationDatasetGenerator:
    """
    may need to be refactored to save simulation trajecories one at a time to reduce RAM consumption
    currently all generated then all saved
    """
    def __init__(
        self,
        simulator: CaseSimulator,
        output_path: str | Path, # move to save method?
        seed: int = 0,
        # storage_format: Literal["hdf5", "zarr"] = "hdf5"
        # resume: bool = True
        # num_workers: int = 1
        # sampling_policy: SamplingPolicy
        # failure_policy: Literal["raise", "skip", "retry"] = "retry"
        # save_solver_diagnostics: bool = True
    ):
        self.simulator = simulator
        self.output_path = Path(output_path)
        self.seed = seed
        # self.overwrite = overwrite

        # if self.output_path.exists() and not self.overwrite:
        #     raise FileExistsError(f"{self.output_path} already exists.")

    @staticmethod
    def split_parameter_space(
        parameter_space: Mapping[str, ParameterSpec],
    ):
        grid_specs = {}
        sampled_specs = {}

        for name, spec in parameter_space.items():
            if spec.is_grid:
                grid_specs[name] = spec
            else:
                sampled_specs[name] = spec

        return sampled_specs, grid_specs

    @staticmethod
    def iter_grid_combinations(
        grid_specs: Mapping[str, ParameterSpec],
    ):
        if not grid_specs:
            yield 0, {}
            return

        names = list(grid_specs)
        value_lists = [grid_specs[name].grid_values() for name in names]

        for grid_idx, values in enumerate(product(*value_lists)):
            yield grid_idx, dict(zip(names, values))

    def generate_split(
        self,
        split: str,
        n_cases: int,
        seed: int,
    ) -> list[SimulationRecord]:
        rng = np.random.default_rng(seed)
        records = []

        sampled_specs, grid_specs = SimulationDatasetGenerator.split_parameter_space(
            self.simulator.parameter_space
        )

        record_idx = 0

        print(f"{split = }")

        for base_case_idx in trange(n_cases):
            sampled_params = {
                name: spec.sample(rng)
                for name, spec in sampled_specs.items()
            }

            for grid_idx, grid_params in SimulationDatasetGenerator.iter_grid_combinations(grid_specs):
                params = sampled_params | grid_params

                case = self.simulator.make_case(params)
                try:
                    record = self.simulator.run_case(case)
                except Exception as e:
                    print(f"Simulation case [{base_case_idx = },{record_idx = }] failed")
                    print(e)
                    continue

                record.metadata.update(
                    {
                        "record_idx": record_idx,
                        "base_case_idx": base_case_idx,
                        "grid_idx": grid_idx,
                        "split": split,
                        "simulator": self.simulator.name,
                        "seed": seed,
                        "params": deepcopy(params),
                    }
                )

                records.append(record)
                record_idx += 1

        return records

    def generate_splits(
        self,
        n_cases: int = 10,
        train_fraction: float = 0.8,
        valid_fraction: float = 0.1,
        test_fraction: float = 0.1,
    ) -> dict[str, list[SimulationRecord]]:
        if not np.isclose(train_fraction + valid_fraction + test_fraction, 1.0):
            raise ValueError("Split fractions must sum to 1.")

        n_train = int(n_cases * train_fraction)
        n_valid = int(n_cases * valid_fraction)
        n_test = n_cases - n_train - n_valid

        return {
            "train": self.generate_split("train", n_train, self.seed),
            "valid": self.generate_split("valid", n_valid, self.seed + 1),
            "test": self.generate_split("test", n_test, self.seed + 2),
        }
    
    def save_split(
        self,
        split: str,
        records: list[SimulationRecord],
        overwrite: bool = False,
    ) -> None:
        self.output_path.mkdir(parents=True, exist_ok=True)
        path = self.output_path / f"{self.simulator.name}_{split}.h5"

        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} already exists. Use overwrite = True.")

        mode = "w" if overwrite else "x"

        with h5py.File(path, mode) as h5:
            h5.attrs["schema"] = "simulation-record-split-v0"
            h5.attrs["simulator"] = self.simulator.name
            h5.attrs["split"] = split
            h5.attrs["n_cases"] = len(records)

            if records:
                h5.create_dataset(
                    "field_names",
                    data=np.array(list(records[0].fields.keys()), dtype="S"),
                )
                h5.create_dataset(
                    "constant_names",
                    data=np.array(list(records[0].constants.keys()), dtype="S"),
                )

            cases_group = h5.create_group("cases")

            for i, record in enumerate(records):
                case_group = cases_group.create_group(f"{i:06d}")
                self._save_record(case_group, record)

    def save_splits(
            self, 
            records_splits: dict[str, list[SimulationRecord]], 
            overwrite: bool = False
    ):
        for split, records in records_splits.items():
            self.save_split(split, records, overwrite)
    
    def _save_record(
        self,
        group: h5py.Group,
        record: SimulationRecord,
    ) -> None:
        coordinates_group = group.create_group("coordinates")
        fields_group = group.create_group("fields")
        constants_group = group.create_group("constants")

        for name, value in record.coordinates.items():
            coordinates_group.create_dataset(name, data=np.asarray(value))

        for name, value in record.fields.items():
            fields_group.create_dataset(name, data=np.asarray(value))

        for name, value in record.constants.items():
            self._save_value(constants_group, name, value)

        metadata_json = json.dumps(record.metadata, default=self._json_default)
        group.attrs["metadata"] = metadata_json

    def _save_value(
        self,
        group: h5py.Group,
        name: str,
        value,
    ) -> None:
        if isinstance(value, dict):
            subgroup = group.create_group(name)
            for key, subvalue in value.items():
                self._save_value(subgroup, key, subvalue)
        elif isinstance(value, str):
            group.attrs[name] = value
        elif np.isscalar(value):
            group.attrs[name] = value
        else:
            group.create_dataset(name, data=np.asarray(value))

    def _json_default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, Path):
            return str(obj)
        return str(obj)

def raw_steps_to_possible_sample_t0s(
    total_steps: int,
    n_steps_input: int,
    n_steps_output: int,
    dt_stride: int,
) -> int:
    required_steps = 1 + dt_stride * (
        n_steps_input + n_steps_output - 1
    )
    return max(0, total_steps - required_steps + 1)

class ChemOperatorDataset(Dataset):
    """Lazy reader for one SimulationDatasetGenerator HDF5 file.

    Samples are returned as structured dictionaries of raw, unnormalized
    tensors. The file is scanned once for names and valid windows, then each
    worker opens its own HDF5 handle on demand.
    """

    VALID_TASKS = {
        "next_step",
        "rollout",
        "operator_pointwise",
        "operator_cartesian",
        "steady_map",
        "field_map",
    }

    def __init__(
        self,
        path: str | Path,
        *,
        task: Literal[
            "next_step",
            "rollout",
            "operator_pointwise",
            "operator_cartesian",
            "steady_map",
            "field_map",
        ] = "next_step",
        input_fields: Sequence[str] | None = None,
        output_fields: Sequence[str] | None = None,
        constant_inputs: Sequence[str] | None = None,
        coordinate_name: str = "time",
        n_steps_input: int = 1,
        n_steps_output: int = 1,
        index_stride: int = 1,
        prediction_horizon: float | None = None,
        full_trajectory_mode: bool = False,
        dtype: torch.dtype | str | np.dtype | type | None = None,
    ):
        super().__init__()

        if task not in self.VALID_TASKS:
            raise ValueError(f"Unsupported task {task!r}.")
        if n_steps_input < 1 or n_steps_output < 1 or index_stride < 1:
            raise ValueError(
                "n_steps_input, n_steps_output, and index_stride must be positive."
            )
        if prediction_horizon is not None and prediction_horizon < 0:
            raise ValueError("prediction_horizon must be non-negative.")

        self.path = Path(path)
        self.task = task
        self.coordinate_name = coordinate_name
        self.n_steps_input = n_steps_input
        self.n_steps_output = n_steps_output
        self.index_stride = index_stride
        self.prediction_horizon = prediction_horizon
        self.full_trajectory_mode = full_trajectory_mode
        self.dtype = self._resolve_dtype(dtype)

        if not self.path.is_file():
            raise FileNotFoundError(
                f"ChemOperatorDataset expects one HDF5 file: {self.path}"
            )

        self._file_handle: h5py.File | None = None

        self.input_fields = tuple(input_fields) if input_fields is not None else None
        self.output_fields = tuple(output_fields) if output_fields is not None else None
        self.constant_inputs = (
            tuple(constant_inputs) if constant_inputs is not None else None
        )

        self.sample_index: list[tuple[str, int, int]] = []
        self.case_n_steps: dict[str, int] = {}
        self.source_coordinate_names: dict[str, str] = {}
        self.case_indices: dict[str, int] = {}

        self.field_names: tuple[str, ...] = ()
        self.constant_names: tuple[str, ...] = ()
        self.coordinate_names: tuple[str, ...] = ()

        self._build_index()

        if self.input_fields is None:
            self.input_fields = self.field_names
        if self.output_fields is None:
            self.output_fields = self.input_fields
        if self.constant_inputs is None:
            self.constant_inputs = self.constant_names

        self._validate_requested_names()

        if not self.sample_index:
            raise ValueError("No valid samples were found for the requested windowing.")

    @staticmethod
    def _decode(value: Any) -> Any:
        if isinstance(value, bytes):
            return value.decode()
        if isinstance(value, np.generic):
            return value.item()
        return value

    @classmethod
    def _decode_attr_dict(cls, attrs: h5py.AttributeManager) -> dict[str, Any]:
        return {key: cls._decode(value) for key, value in attrs.items()}

    @classmethod
    def _decode_names(cls, values: Any) -> tuple[str, ...]:
        return tuple(str(cls._decode(value)) for value in np.asarray(values))

    @staticmethod
    def _dataset_paths(group: h5py.Group, prefix: str = "") -> list[str]:
        paths: list[str] = []
        for name, value in group.items():
            path = f"{prefix}/{name}" if prefix else name
            if isinstance(value, h5py.Dataset):
                paths.append(path)
            elif isinstance(value, h5py.Group):
                paths.extend(ChemOperatorDataset._dataset_paths(value, path))
        return paths

    @staticmethod
    def _constant_paths(group: h5py.Group, prefix: str = "") -> list[str]:
        paths = [f"{prefix}/{name}" if prefix else name for name in group.attrs]
        for name, value in group.items():
            path = f"{prefix}/{name}" if prefix else name
            if isinstance(value, h5py.Dataset):
                paths.append(path)
            elif isinstance(value, h5py.Group):
                paths.extend(ChemOperatorDataset._constant_paths(value, path))
        return paths

    @staticmethod
    def _get_dataset(group: h5py.Group, name: str) -> h5py.Dataset:
        value: h5py.Group | h5py.Dataset = group
        for part in name.split("/"):
            value = value[part]
        if not isinstance(value, h5py.Dataset):
            raise KeyError(f"{name!r} is not a dataset.")
        return value

    @staticmethod
    def _read_constant(group: h5py.Group, name: str) -> Any:
        parts = name.split("/")
        current: h5py.Group | h5py.Dataset = group

        for part in parts[:-1]:
            current = current[part]
            if not isinstance(current, h5py.Group):
                raise KeyError(f"{name!r} does not name a constant.")

        leaf = parts[-1]
        if isinstance(current, h5py.Group) and leaf in current.attrs:
            return current.attrs[leaf]
        value = current[leaf]
        if isinstance(value, h5py.Dataset):
            return value[()]
        raise KeyError(f"{name!r} does not name a tensor-valued constant.")

    @staticmethod
    def _resolve_dtype(
        dtype: torch.dtype | str | np.dtype | type | None,
    ) -> torch.dtype | None:
        if dtype is None:
            return None
        if isinstance(dtype, torch.dtype):
            return dtype
        if isinstance(dtype, str):
            dtype_name = dtype.removeprefix("torch.")
            torch_dtype = getattr(torch, dtype_name, None)
            if isinstance(torch_dtype, torch.dtype):
                return torch_dtype

        try:
            dtype_name = np.dtype(dtype).name
        except TypeError as exc:
            raise TypeError(f"Unsupported dtype {dtype!r}.") from exc

        torch_dtype = getattr(torch, dtype_name, None)
        if not isinstance(torch_dtype, torch.dtype):
            raise TypeError(f"Unsupported dtype {dtype!r}.")
        return torch_dtype

    def _to_tensor(self, value: Any, *, name: str) -> torch.Tensor:
        value = ChemOperatorDataset._decode(value)
        array = np.asarray(value)
        if array.dtype.kind in {"O", "S", "U"}:
            raise TypeError(f"{name!r} is non-numeric and cannot be a tensor.")
        tensor = torch.as_tensor(array)
        if self.dtype is not None:
            tensor = tensor.to(dtype=self.dtype)
        return tensor

    def _resolve_coordinate_name(self, coordinates: h5py.Group) -> str:
        if self.coordinate_name in coordinates:
            return self.coordinate_name
        if self.coordinate_name == "time" and "t" in coordinates:
            return "t"
        if len(coordinates) == 1:
            return next(iter(coordinates.keys()))

        available = ", ".join(sorted(coordinates.keys()))
        raise KeyError(
            f"Coordinate {self.coordinate_name!r} was not found. "
            f"Available coordinates: {available}."
        )

    def _case_steps(self, case: h5py.Group, source_coordinate_name: str) -> int:
        coordinate = np.asarray(case["coordinates"][source_coordinate_name])
        if coordinate.ndim == 0:
            raise ValueError(
                f"Coordinate {source_coordinate_name!r} must have at least one step."
            )
        return int(coordinate.shape[0])

    def _field_shape_matches_steps(
        self,
        dataset: h5py.Dataset,
        n_steps: int,
    ) -> bool:
        return dataset.ndim > 0 and dataset.shape[0] == n_steps

    def _read_coordinate_values(
        self,
        case: h5py.Group,
        source_coordinate_name: str,
    ) -> np.ndarray:
        values = np.asarray(case["coordinates"][source_coordinate_name])
        if values.ndim != 1:
            values = values.reshape(values.shape[0], -1)[:, 0]
        return values

    def _index_at_horizon(
        self,
        coordinates: np.ndarray,
        input_last_index: int,
    ) -> int:
        if self.prediction_horizon is None:
            return input_last_index + self.index_stride

        target = coordinates[input_last_index] + self.prediction_horizon
        after_input = np.arange(input_last_index + 1, coordinates.shape[0])
        candidates = after_input[coordinates[after_input] >= target]
        if candidates.size == 0:
            return coordinates.shape[0]
        return int(candidates[0])

    def _first_output_index(
        self,
        case: h5py.Group,
        input_start: int,
        source_coordinate_name: str,
    ) -> int:
        input_last = input_start + (self.n_steps_input - 1) * self.index_stride
        if self.prediction_horizon is None:
            return input_last + self.index_stride
        coordinates = self._read_coordinate_values(case, source_coordinate_name)
        return self._index_at_horizon(coordinates, input_last)

    def _output_count(self, n_steps: int, output_start: int) -> int:
        if self.full_trajectory_mode or self.task in {
            "operator_cartesian",
            "field_map",
        }:
            return max(0, 1 + (n_steps - 1 - output_start) // self.index_stride)
        return self.n_steps_output

    def _output_indices(self, n_steps: int, output_start: int) -> np.ndarray:
        count = self._output_count(n_steps, output_start)
        return output_start + np.arange(count, dtype=np.int64) * self.index_stride

    def _sample_is_valid(
        self,
        n_steps: int,
        input_start: int,
        output_start: int,
    ) -> bool:
        input_last = input_start + (self.n_steps_input - 1) * self.index_stride
        output_indices = self._output_indices(n_steps, output_start)
        return (
            input_start >= 0
            and input_last < n_steps
            and output_indices.size > 0
            and int(output_indices[-1]) < n_steps
        )

    def _build_index(self) -> None:
        field_names: tuple[str, ...] | None = None
        constant_names: tuple[str, ...] | None = None
        coordinate_names: tuple[str, ...] | None = None

        with h5py.File(self.path, "r") as file:
            if "cases" not in file:
                raise KeyError(f"{self.path} does not contain a 'cases' group.")

            cases = file["cases"]
            case_names = sorted(cases.keys())
            if not case_names:
                raise ValueError(f"{self.path} contains no cases.")

            for case_idx, case_name in enumerate(case_names):
                case = cases[case_name]
                source_coordinate_name = self._resolve_coordinate_name(
                    case["coordinates"]
                )
                n_steps = self._case_steps(case, source_coordinate_name)

                self.case_n_steps[case_name] = n_steps
                self.source_coordinate_names[case_name] = source_coordinate_name
                self.case_indices[case_name] = case_idx

                if field_names is None:
                    if "field_names" in file:
                        field_names = self._decode_names(file["field_names"][:])
                    else:
                        field_names = tuple(sorted(self._dataset_paths(case["fields"])))
                    constant_names = tuple(
                        sorted(self._constant_paths(case["constants"]))
                    )
                    coordinate_names = tuple(sorted(case["coordinates"].keys()))

                self._add_case_samples(
                    case_name,
                    case,
                    n_steps,
                    source_coordinate_name,
                )

        self.field_names = field_names or ()
        self.constant_names = constant_names or ()
        self.coordinate_names = coordinate_names or ()

    def _add_case_samples(
        self,
        case_name: str,
        case: h5py.Group,
        n_steps: int,
        source_coordinate_name: str,
    ) -> None:
        if self.task == "steady_map":
            output_start = n_steps - 1 - (self.n_steps_output - 1) * self.index_stride
            if self._sample_is_valid(n_steps, 0, output_start):
                self.sample_index.append((case_name, 0, output_start))
            return

        if self.full_trajectory_mode or self.task in {
            "operator_cartesian",
            "field_map",
        }:
            output_start = self._first_output_index(case, 0, source_coordinate_name)
            if self._sample_is_valid(n_steps, 0, output_start):
                self.sample_index.append((case_name, 0, output_start))
            return

        if self.task == "operator_pointwise":
            output_start = self._first_output_index(case, 0, source_coordinate_name)
            while self._sample_is_valid(n_steps, 0, output_start):
                self.sample_index.append((case_name, 0, output_start))
                output_start += self.index_stride
            return

        for input_start in range(n_steps):
            output_start = self._first_output_index(
                case,
                input_start,
                source_coordinate_name,
            )
            if not self._sample_is_valid(n_steps, input_start, output_start):
                break
            self.sample_index.append((case_name, input_start, output_start))

    def _validate_requested_names(self) -> None:
        available_fields = set(self.field_names)
        requested_fields = set(self.input_fields or ()) | set(self.output_fields or ())
        missing_fields = sorted(requested_fields - available_fields)
        if missing_fields:
            raise KeyError(
                "Requested fields are not present in the dataset: "
                + ", ".join(missing_fields)
            )

        available_constants = set(self.constant_names)
        missing_constants = sorted(set(self.constant_inputs or ()) - available_constants)
        if missing_constants:
            raise KeyError(
                "Requested constants are not present in the dataset: "
                + ", ".join(missing_constants)
            )

    def _file(self) -> h5py.File:
        if self._file_handle is None:
            self._file_handle = h5py.File(self.path, "r")
        return self._file_handle

    def _slice_field(
        self,
        dataset: h5py.Dataset,
        indices: np.ndarray,
        n_steps: int,
        *,
        name: str,
    ) -> torch.Tensor:
        if self._field_shape_matches_steps(dataset, n_steps):
            return self._to_tensor(np.asarray(dataset[indices]), name=name)
        return self._to_tensor(np.asarray(dataset), name=name)

    def _load_fields(
        self,
        case: h5py.Group,
        names: Sequence[str],
        indices: np.ndarray,
        n_steps: int,
    ) -> dict[str, torch.Tensor]:
        fields = {}
        for name in names:
            dataset = self._get_dataset(case["fields"], name)
            fields[name] = self._slice_field(
                dataset,
                indices,
                n_steps,
                name=f"fields/{name}",
            )
        return fields

    def _load_coordinates(
        self,
        case: h5py.Group,
        indices: np.ndarray,
        n_steps: int,
        source_coordinate_name: str,
    ) -> dict[str, torch.Tensor]:
        coordinates: dict[str, torch.Tensor] = {}
        for name, dataset in case["coordinates"].items():
            output_name = (
                self.coordinate_name
                if name == source_coordinate_name
                else name
            )
            if name == source_coordinate_name:
                coordinates[output_name] = self._to_tensor(
                    np.asarray(dataset[indices]),
                    name=f"coordinates/{name}",
                )
            else:
                coordinates[output_name] = self._to_tensor(
                    np.asarray(dataset),
                    name=f"coordinates/{name}",
                )
        return coordinates

    def _load_constants(self, case: h5py.Group) -> dict[str, torch.Tensor]:
        constants = {}
        for name in self.constant_inputs or ():
            constants[name] = self._to_tensor(
                self._read_constant(case["constants"], name),
                name=f"constants/{name}",
            )
        return constants

    def _case_metadata(self, case: h5py.Group) -> dict[str, Any]:
        metadata = self._decode_attr_dict(case.attrs)
        raw_metadata = metadata.pop("metadata", None)
        if raw_metadata is not None:
            try:
                decoded = json.loads(raw_metadata)
            except TypeError:
                decoded = json.loads(str(raw_metadata))
            metadata.update(decoded)
        return metadata

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        case_name, input_start, output_start = self.sample_index[index]
        file = self._file()
        case = file["cases"][case_name]
        n_steps = self.case_n_steps[case_name]

        input_indices = (
            input_start
            + np.arange(self.n_steps_input, dtype=np.int64) * self.index_stride
        )
        output_indices = self._output_indices(n_steps, output_start)

        metadata = self._case_metadata(case)
        source_coordinate_name = self.source_coordinate_names[case_name]
        metadata.update(
            {
                "task": self.task,
                "file_path": str(self.path),
                "case_name": case_name,
                "case_index": self.case_indices[case_name],
                "input_start_index": input_start,
                "output_start_index": output_start,
                "input_indices": input_indices.tolist(),
                "output_indices": output_indices.tolist(),
                "index_stride": self.index_stride,
                "coordinate_name": self.coordinate_name,
                "source_coordinate_name": source_coordinate_name,
            }
        )

        return {
            "input_fields": self._load_fields(
                case,
                self.input_fields or (),
                input_indices,
                n_steps,
            ),
            "output_fields": self._load_fields(
                case,
                self.output_fields or (),
                output_indices,
                n_steps,
            ),
            "constant_inputs": self._load_constants(case),
            "input_coordinates": self._load_coordinates(
                case,
                input_indices,
                n_steps,
                source_coordinate_name,
            ),
            "output_coordinates": self._load_coordinates(
                case,
                output_indices,
                n_steps,
                source_coordinate_name,
            ),
            "metadata": metadata,
        }

    def close(self) -> None:
        if getattr(self, "_file_handle", None) is not None:
            self._file_handle.close()
            self._file_handle = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file_handle"] = None
        return state

    def __del__(self):
        self.close()


# Public preprocessing API. These imports intentionally live at the end of the
# module so the HDF5 reader above is defined before the processing layer loads.
from chem_operator.normalization import (  # noqa: E402
    IdentityNormalizer,
    MinMaxNormalization,
    MinMaxNormalizer,
    Normalizer,
    RMSNormalization,
    RMSNormalizer,
    ZScoreNormalization,
    ZScoreNormalizer,
)
from chem_operator.dataset_processing import (  # noqa: E402
    DataProcessor,
    FieldPacker,
    NormalizationConfig,
    PackedFieldLayout,
    ProcessedDataset,
    TargetTransformConfig,
)
