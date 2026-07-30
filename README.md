# ChemOperator

### Dev commands

```bash
uv run pytest -v
pyreverse -o png -p ChemOperator src/chem_operator/
```

## To Do

put issues in Github. summary and proposed work
- clean code
- split up big files. make clean
    - device.py separate. decides global in __init__ to use for everything
- docs
- testing
- physics informed
- benchmarking

- benchmark PhysicsNeMo FNO with and without physics informed loss
- [Q2D](./scripts/TODO_q2d_fno.md)
    - Compare accuracy of superresolution with low-fidelity to high fidelity interpolation.
    - Accuracy-constrained break-even
        - Need to generate specific test data h5 file that solves the same (tutorial) case (all `Constant`) at multiple resolutions.
- Make PhysicsNeMo example PDE for PFR with conjugate heat transfer
    - Use small number of chemicals, so the NeMo PDE isn't too large
- physics informed loss functions for CSTR, PFR, and packed_bed_1d