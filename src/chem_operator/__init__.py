import warnings
import os
from pathlib import Path

# Cantera imports optional drawing helpers that initialize Matplotlib. Keep
# headless/library imports away from an unwritable user configuration folder.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cantera as ct

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
ct.add_data_directory(Path(__file__).resolve().parent / "example_data")
