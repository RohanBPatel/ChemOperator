import warnings
import cantera as ct
from pathlib import Path

# expected warning when reading mechanisms
warnings.filterwarnings(
    "ignore",
    message=".*NasaPoly2::validate.*",
    category=UserWarning,
)

# expected warning when reading mechanisms
warnings.filterwarnings(
    "ignore",
    message=".*discontinuity in h/RT detected.*",
    category=UserWarning,
)

# add example_data
ct.add_data_directory(Path(__file__).resolve().parent / "external" / "example_data")