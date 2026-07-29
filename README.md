# ChemOperator

### Dev Commands
```
uv run pytest -v
pyreverse -o png -p ChemOperator src/chem_operator/
```

## To Do
- benchmark PhysicsNeMo FNO with and without physics informed loss
- [Q2D](./scripts/TODO_q2d_fno.md)
    - Compare accuracy of superresolution with low-fidelity to high fidelity interpolation.
    - Accuracy-constrained break-even
        - Need to generate specific test data h5 file that solves the same (tutorial) case (all `Constant`) at multiple resolutions.
- Make PhysicsNeMo example PDE for PFR with conjugate heat transfer
    - Use small number of chemicals, so the NeMo PDE isn't too large
- Mechanism file names should be passed to the metadata in SimulationRecord. 
    Problem here: src/chem_operator/reactors/q2d/dataset_generator.py