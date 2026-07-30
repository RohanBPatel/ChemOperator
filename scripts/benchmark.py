"""Run reactor operator benchmarks from an explicit Python configuration.

Edit the values in the ``Python parameters`` section below, then run:

    python scripts/benchmark.py

Set ``RUN_MATRIX`` to ``True`` to benchmark every requested problem/model
combination and generate the aggregate comparison charts.  The default is a
single cell so an accidental invocation does not start a full tuning matrix.
"""

from __future__ import annotations

from pathlib import Path
from pprint import pprint

from chem_operator._benchmark import (
    BenchmarkConfig,
    list_models,
    list_problems,
    run_benchmark,
    run_matrix,
)


# ---------------------------------------------------------------------------
# Python parameters
# ---------------------------------------------------------------------------

RUN_MATRIX = True

# Used when RUN_MATRIX is False.
PROBLEM = "pfr"
MODEL = "fno"


# Used when RUN_MATRIX is True.  Replace either tuple with a subset if desired.
# MATRIX_PROBLEMS = tuple(list_problems())
# MATRIX_PROBLEMS = ('cstr', 'pfr', 'packed_bed', 'pipe_flow', 'pipe_flow_transient', 'q2d')
MATRIX_PROBLEMS = ('pfr', 'packed_bed', 'pipe_flow')
# MATRIX_MODELS = tuple(list_models())
# MATRIX_MODELS = ('fno', 'deeponet', 'pod-deeponet')
MATRIX_MODELS = ('deeponet', 'pod-deeponet')

SEED = 42
OUTPUT_ROOT = Path("artifacts/benchmarks")
DATA_ROOT = Path("datasets")

USE_RAY = False
TUNE_TIME_BUDGET_S = 5 * 60
TUNE_NUM_SAMPLES = 20
MAX_CONCURRENT_TRIALS = 1
CPUS_PER_TRIAL = 1.0
GPUS_PER_TRIAL = 1

TUNE_EPOCHS = 30
FINAL_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 15
CHECKPOINT_INTERVAL = 10
NUM_WORKERS = 2

# Optional output selection, for example ("T", "X_O2").  Use None for the
# problem registry defaults.  CSTR also accepts ("all",) and ("X:*",).
CHANNEL_OVERRIDES = None

MAX_TRAIN_CASES = None
MAX_VALID_CASES = None
MAX_TEST_CASES = None
DEVICE = "auto"
RESUME = False
INFERENCE_WARMUPS = 10
INFERENCE_REPEATS = 50
FAIL_FAST = False


CONFIG = BenchmarkConfig(
    problem=PROBLEM,
    model=MODEL,
    seed=SEED,
    output_root=OUTPUT_ROOT,
    data_root=DATA_ROOT,
    use_ray=USE_RAY,
    tune_time_budget_s=TUNE_TIME_BUDGET_S,
    tune_num_samples=TUNE_NUM_SAMPLES,
    max_concurrent_trials=MAX_CONCURRENT_TRIALS,
    cpus_per_trial=CPUS_PER_TRIAL,
    gpus_per_trial=GPUS_PER_TRIAL,
    tune_epochs=TUNE_EPOCHS,
    final_epochs=FINAL_EPOCHS,
    early_stopping_patience=EARLY_STOPPING_PATIENCE,
    checkpoint_interval=CHECKPOINT_INTERVAL,
    num_workers=NUM_WORKERS,
    channel_overrides=CHANNEL_OVERRIDES,
    max_train_cases=MAX_TRAIN_CASES,
    max_valid_cases=MAX_VALID_CASES,
    max_test_cases=MAX_TEST_CASES,
    device=DEVICE,
    resume=RESUME,
    inference_warmups=INFERENCE_WARMUPS,
    inference_repeats=INFERENCE_REPEATS,
    fail_fast=FAIL_FAST,
)


def show_parameters() -> None:
    """Print the exact Python configuration before starting work."""

    pprint(
        {
            "run_matrix": RUN_MATRIX,
            "matrix_problems": MATRIX_PROBLEMS if RUN_MATRIX else None,
            "matrix_models": MATRIX_MODELS if RUN_MATRIX else None,
            "benchmark": CONFIG.to_dict(),
        },
        sort_dicts=False,
    )


def main() -> None:
    show_parameters()
    if RUN_MATRIX:
        results = run_matrix(
            CONFIG,
            problems=MATRIX_PROBLEMS,
            models=MATRIX_MODELS,
        )
        pprint([result.to_dict() for result in results], sort_dicts=False)
        return

    result = run_benchmark(CONFIG)
    pprint(result.to_dict(), sort_dicts=False)


if __name__ == "__main__":
    main()
