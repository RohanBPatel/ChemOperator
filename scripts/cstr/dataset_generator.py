import os
from pathlib import Path
import time
from copy import deepcopy

import cantera as ct
import numpy as np

from chem_operator.utils import get_mechanism_file
from chem_operator.datasets import CaseParameters, CaseSimulator, SimulationRecord, SimulationDatasetGenerator

prog_path = Path(os.path.abspath(""))

class CSTRCaseSimulator(CaseSimulator):
    name = "cstr"

    def __init__(
        self,
        phi_range=(0.5, 2.0),
        reactive_fraction_range=(0.02, 0.08),
    ):
        self.phi_range = phi_range
        self.reactive_fraction_range = reactive_fraction_range

    def sample_case(
        self,
        rng: np.random.Generator,
    ) -> CaseParameters:
        phi = rng.uniform(*self.phi_range)
        reactive_fraction = rng.uniform(*self.reactive_fraction_range)

        # X_fuel / X_O2 = phi / 11
        ratio = phi / 11.0

        x_o2 = reactive_fraction / (1.0 + ratio)
        x_fuel = ratio * x_o2
        x_he = 1.0 - reactive_fraction

        return CaseParameters(
            initial_conditions={
                "gas/T": 925, # Kelvin
                "gas/P": 1.046138 * ct.one_atm, # in atm. This equals 1.06 bars
                "gas/X": {"NC7H16": x_fuel, "O2": x_o2,"HE": x_he}
            },
            geometry={
                "reactor_volume": 30.5 * (1e-2) ** 3, # m^3
            },
            controls={
                "residence_time": 2, # s
                "pressure_controller_K": 1e-6,
            },
            solver_parameters={
                "t_final": 50.0,
                "dt": 5e-3, # None for adaptive stepping
                "adaptive": False,
            },
            mechanism_parameters={
                "phi": phi,
                "reactive_fraction": reactive_fraction,
            }
        )

    def run_case(
        self,
        case: CaseParameters,
    ) -> SimulationRecord:
        mechanism_file_name = "n-heptane-NUIG-2016.yaml"
        gas = ct.Solution(get_mechanism_file(mechanism_file_name))
        gas.TPX = case.initial_conditions["gas/T"], case.initial_conditions["gas/P"], case.initial_conditions["gas/X"]

        fuel_air_mixture_tank = ct.Reservoir(gas)
        exhaust = ct.Reservoir(gas)

        stirred_reactor = ct.IdealGasMoleReactor(gas, energy="off", volume=case.geometry["reactor_volume"])
        mass_flow_controller = ct.MassFlowController(
            upstream=fuel_air_mixture_tank,
            downstream=stirred_reactor,
            mdot=lambda t: stirred_reactor.mass / case.controls["residence_time"],
        )

        pressure_regulator = ct.PressureController(
            upstream=stirred_reactor,
            downstream=exhaust,
            primary=mass_flow_controller,
            K=case.controls["pressure_controller_K"],
        )

        reactor_network = ct.ReactorNet([stirred_reactor])

        states = ct.SolutionArray(gas, extra=["t"])

        tic = time.perf_counter()

        if case.solver_parameters["adaptive"]:
            t = 0.0
            while t < case.solver_parameters["t_final"]:
                t = reactor_network.step()
                states.append(stirred_reactor.phase.state, t=t)
        else:
            n_steps = int(np.round(case.solver_parameters["t_final"] / case.solver_parameters["dt"]))
            times = np.linspace(0.0, n_steps * case.solver_parameters["dt"], n_steps + 1)
            for t in times:
                reactor_network.advance(t)
                states.append(stirred_reactor.phase.state, t=t)
            pass

        toc = time.perf_counter()

        return SimulationRecord(
            coordinates={"t": states.t},
            fields={"T": states.T, "P": states.P, "X": states.X},
            constants=case.geometry | case.controls | case.mechanism_parameters,
            metadata={
                "wall_time": toc - tic,
                "mechanism": mechanism_file_name,
                "solver_parameters": deepcopy(case.solver_parameters)
            }
        )

if False:   
    cstr_simulator = CSTRCaseSimulator()
    cstr_dataset_generator = SimulationDatasetGenerator(
        cstr_simulator,
        prog_path.parent.parent / "datasets" / "cstr",
    )
    records_splits = cstr_dataset_generator.generate_splits(n_cases=20)
    cstr_dataset_generator.save_splits(records_splits, overwrite=False)