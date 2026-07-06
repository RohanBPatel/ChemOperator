from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence
import numpy as np


class ParameterSpec(Protocol):
    @property
    def is_grid(self) -> bool: ...

    def sample(self, rng: np.random.Generator) -> Any: ...

    def grid_values(self) -> Sequence[Any]: ...


@dataclass(frozen=True)
class Constant:
    value: Any

    @property
    def is_grid(self) -> bool:
        return False

    def sample(self, rng: np.random.Generator) -> Any:
        return self.value

    def grid_values(self) -> Sequence[Any]:
        raise TypeError("Constant is not a grid parameter.")


@dataclass(frozen=True)
class Grid:
    values: Sequence[Any]

    @property
    def is_grid(self) -> bool:
        return True

    def sample(self, rng: np.random.Generator) -> Any:
        raise TypeError("Grid must be expanded, not sampled.")

    def grid_values(self) -> Sequence[Any]:
        return list(self.values)


@dataclass(frozen=True)
class Choice:
    values: Sequence[Any]

    @property
    def is_grid(self) -> bool:
        return False

    def sample(self, rng: np.random.Generator) -> Any:
        values = list(self.values)
        return values[int(rng.integers(0, len(values)))]

    def grid_values(self) -> Sequence[Any]:
        raise TypeError("Choice is sampled, not grid-expanded.")


@dataclass(frozen=True)
class Uniform:
    low: float
    high: float

    @property
    def is_grid(self) -> bool:
        return False

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.low, self.high))

    def grid_values(self) -> Sequence[Any]:
        raise TypeError("Uniform is sampled, not grid-expanded.")


@dataclass(frozen=True)
class LogUniform:
    low: float
    high: float

    @property
    def is_grid(self) -> bool:
        return False

    def sample(self, rng: np.random.Generator) -> float:
        return float(np.exp(rng.uniform(np.log(self.low), np.log(self.high))))

    def grid_values(self) -> Sequence[Any]:
        raise TypeError("LogUniform is sampled, not grid-expanded.")


@dataclass(frozen=True)
class Normal:
    mean: float
    std: float
    clip: tuple[float, float] | None = None

    @property
    def is_grid(self) -> bool:
        return False

    def sample(self, rng: np.random.Generator) -> float:
        value = float(rng.normal(self.mean, self.std))

        if self.clip is not None:
            value = float(np.clip(value, *self.clip))

        return value

    def grid_values(self) -> Sequence[Any]:
        raise TypeError("Normal is sampled, not grid-expanded.")


@dataclass(frozen=True)
class CallableSample:
    fn: Callable[[np.random.Generator], Any]

    @property
    def is_grid(self) -> bool:
        return False

    def sample(self, rng: np.random.Generator) -> Any:
        return self.fn(rng)

    def grid_values(self) -> Sequence[Any]:
        raise TypeError("CallableSample is sampled, not grid-expanded.")