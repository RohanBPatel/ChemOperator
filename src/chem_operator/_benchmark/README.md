## Reactor operator benchmarks

`scripts/benchmark.py` compares FNO, DeepONet, and POD-DeepONet on the
non-isothermal CSTR, PFR chain, 1D packed bed, steady pipe, transient pipe, and
Q2D CMR datasets. Every model uses the same train/validation/test split,
physical output channels, and train-only normalization for a problem.

The benchmark is configured entirely in Python. Edit the parameter block near
the top of `scripts/benchmark.py`; it prints the resolved settings before
starting. `RUN_MATRIX = False` runs the selected `PROBLEM` and `MODEL`.
`RUN_MATRIX = True` runs the requested problem/model tuples and produces the
comparison charts:

```python
RUN_MATRIX = True
MATRIX_PROBLEMS = ("cstr", "pfr", "packed_bed", "pipe_flow",
                   "pipe_flow_transient", "q2d")
MATRIX_MODELS = ("fno", "deeponet", "pod-deeponet")
TUNE_TIME_BUDGET_S = 60 * 60
```

Then run the Python driver:

```bash
python scripts/benchmark.py
```

The default Ray Tune policy is Optuna/TPE with ASHA and an equal one-hour
budget per cell. Resources, case limits, epochs, concurrency, channels, and
the budget are fields on `BenchmarkConfig`. Set `use_ray=False` for a quick
run with the registry's fixed defaults. CSTR uses its curated species set by
default; `channel_overrides=("all",)` enables `T`, `P`, and every mechanism
species, while `("X:*",)` selects every species without the scalar fields.

For programmatic use outside the driver, import the reusable benchmark engine:

```python
from chem_operator._benchmark import BenchmarkConfig, run_benchmark, run_matrix

one = run_benchmark(BenchmarkConfig(problem="cstr", model="fno"))
all_results = run_matrix(
    BenchmarkConfig(problem="cstr", model="fno"),
    problems=("cstr", "pfr"),
    models=("fno", "deeponet", "pod-deeponet"),
)
```

Artifacts are written under
`artifacts/benchmarks/{run_id}/{problem}/{model}/`. Each successful cell
contains resolved and best JSON configs, `model.pt`, `preprocessing.pt`,
metrics, timings, history, provenance, Ray trial state, and `pod.npz` when
applicable. The run's `summary/` directory contains the CSV matrix, accuracy
and timing heatmaps, accuracy/latency plot, and per-problem charts. Failed
cells remain in the table and plots.

Reload a checkpoint with:

```python
from chem_operator._benchmark import load_benchmark

saved = load_benchmark("artifacts/benchmarks/RUN/q2d/fno")
prediction = saved.predict(normalized_model_input)
```

The six original scripts retain their specialized physics, mesh,
super-resolution, and cost plots. Each also exposes
`run_unified_benchmark(model=...)` for the common matrix workflow.
