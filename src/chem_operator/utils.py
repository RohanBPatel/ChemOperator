from __future__ import annotations
import os
from pathlib import Path
# from collections.abc import Iterable, Sequence, Callable
# from typing import Any, Protocol, Literal
# from io import StringIO
# import time
# from dataclasses import dataclass
from importlib.resources import files

# from tqdm import tqdm, trange

# import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# from sklearn.decomposition import PCA

# import torch
# from torch.utils.data import DataLoader, Dataset

os.environ["DDE_BACKEND"] = "pytorch"
import deepxde as dde

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

import warnings

warnings.filterwarnings(
    "ignore",
    message=".*NasaPoly2::validate.*",
    category=UserWarning,
)

warnings.filterwarnings(
    "ignore",
    message=".*discontinuity in h/RT detected.*",
    category=UserWarning,
)

def get_mechanism_file(file_name: str) -> Path:
    """
    Return the path to an example mechanism or data file.
    """
    return files("chem_operator.example_data") / file_name

def to_numpy(x):
    if hasattr(x, "detach"):  # torch tensor
        return x.detach().cpu().numpy()
    return np.asarray(x)

def add_filtered_handles(ax: plt.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())