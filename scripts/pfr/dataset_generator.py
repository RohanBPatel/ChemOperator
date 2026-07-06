import os
from pathlib import Path
import time
from copy import deepcopy

import cantera as ct
import numpy as np

from chem_operator.utils import get_mechanism_file
from chem_operator.datasets import (
    CaseParameters,
    CaseSimulator,
    SimulationRecord,
    SimulationDatasetGenerator,
)

prog_path = Path(os.path.abspath(""))

class _BasePFRCaseSimulator(CaseSimulator):
    mechanism_file_name = "h2o2.yaml"

    def __init__(
        self,
        temperature_range=(1200.0, 1800.0),
        velocity_range=(0.004, 0.008),
        phi_range=(0.5, 2.0),
        ar_o2_ratio_range=(0.0, 0.2),
        pressure=ct.one_atm,
        length=1.5e-7,
        area=1.0e-4,
        n_steps=2000,
    ):
        self.temperature_range = temperature_range
        self.velocity_range = velocity_range
        self.phi_range = phi_range
        self.ar_o2_ratio_range = ar_o2_ratio_range
        self.pressure = pressure
        self.length = length
        self.area = area
        self.n_steps = n_steps

    def sample_case(
        self,
        rng: np.random.Generator,
    ) -> CaseParameters:
        temperature = rng.uniform(*self.temperature_range)
        inlet_velocity = rng.uniform(*self.velocity_range)
        phi = rng.uniform(*self.phi_range)
        ar_o2_ratio = rng.uniform(*self.ar_o2_ratio_range)

        composition = self._composition_from_phi(
            temperature=temperature,
            pressure=self.pressure,
            phi=phi,
            ar_o2_ratio=ar_o2_ratio,
        )

        return CaseParameters(
            initial_conditions={
                "gas/T": temperature,
                "gas/P": self.pressure,
                "gas/X": composition,
            },
            geometry={
                "length": self.length,
                "area": self.area,
            },
            controls={
                "inlet_velocity": inlet_velocity,
            },
            solver_parameters={
                "n_steps": self.n_steps,
            },
            mechanism_parameters={
                "phi": phi,
                "ar_o2_ratio": ar_o2_ratio,
            },
        )

    def _composition_from_phi(
        self,
        temperature: float,
        pressure: float,
        phi: float,
        ar_o2_ratio: float,
    ) -> dict[str, float]:
        gas = self._new_solution()
        gas.TP = temperature, pressure
        gas.set_equivalence_ratio(
            phi,
            fuel="H2",
            oxidizer={"O2": 1.0, "AR": ar_o2_ratio},
        )

        return {
            species: float(mole_fraction)
            for species, mole_fraction in zip(gas.species_names, gas.X)
            if mole_fraction > 0.0
        }

    def _new_solution(self) -> ct.Solution:
        mechanism_path = Path(get_mechanism_file(self.mechanism_file_name))
        if mechanism_path.exists():
            return ct.Solution(mechanism_path)
        return ct.Solution(self.mechanism_file_name)

    def _initial_gas(self, case: CaseParameters) -> ct.Solution:
        gas = self._new_solution()
        gas.TPX = (
            case.initial_conditions["gas/T"],
            case.initial_conditions["gas/P"],
            case.initial_conditions["gas/X"],
        )
        return gas

    def _record(
        self,
        case: CaseParameters,
        states: ct.SolutionArray,
        wall_time: float,
    ) -> SimulationRecord:
        return SimulationRecord(
            coordinates={"t": states.t, "z": states.z},
            fields={
                "T": states.T,
                "P": states.P,
                "X": states.X,
                "velocity": states.velocity,
            },
            constants=case.geometry | case.controls | case.mechanism_parameters,
            metadata={
                "wall_time": wall_time,
                "mechanism": self.mechanism_file_name,
                "solver_parameters": deepcopy(case.solver_parameters),
            },
        )


class PFRLagrangianParticleCaseSimulator(_BasePFRCaseSimulator):
    name = "pfr_lagrangian_particle"

    def run_case(
        self,
        case: CaseParameters,
    ) -> SimulationRecord:
        gas = self._initial_gas(case)

        area = case.geometry["area"]
        length = case.geometry["length"]
        inlet_velocity = case.controls["inlet_velocity"]
        n_steps = case.solver_parameters["n_steps"]

        mass_flow_rate = inlet_velocity * gas.density * area

        reactor = ct.IdealGasConstPressureReactor(gas, clone=True)
        reactor_network = ct.ReactorNet([reactor])

        t_total = length / inlet_velocity
        dt = t_total / n_steps

        states = ct.SolutionArray(reactor.phase, extra=["t", "z", "velocity"])
        states.append(
            reactor.phase.state,
            t=0.0,
            z=0.0,
            velocity=inlet_velocity,
        )

        tic = time.perf_counter()

        z = 0.0
        for step in range(1, n_steps + 1):
            t = step * dt
            reactor_network.advance(t)

            velocity = mass_flow_rate / (area * reactor.phase.density)
            z += velocity * dt

            states.append(reactor.phase.state, t=t, z=z, velocity=velocity)

        toc = time.perf_counter()

        return self._record(case, states, toc - tic)


class PFRChainOfReactorsCaseSimulator(_BasePFRCaseSimulator):
    name = "pfr_chain_of_reactors"

    def __init__(
        self,
        temperature_range=(1200.0, 1800.0),
        velocity_range=(0.004, 0.008),
        phi_range=(0.5, 2.0),
        ar_o2_ratio_range=(0.0, 0.2),
        pressure=ct.one_atm,
        length=1.5e-7,
        area=1.0e-4,
        n_steps=2000,
        pressure_controller_K=1e-12,
        max_time_step=1e4,
    ):
        super().__init__(
            temperature_range=temperature_range,
            velocity_range=velocity_range,
            phi_range=phi_range,
            ar_o2_ratio_range=ar_o2_ratio_range,
            pressure=pressure,
            length=length,
            area=area,
            n_steps=n_steps,
        )
        self.pressure_controller_K = pressure_controller_K
        self.max_time_step = max_time_step

    def sample_case(
        self,
        rng: np.random.Generator,
    ) -> CaseParameters:
        case = super().sample_case(rng)
        case.controls = case.controls | {
            "pressure_controller_K": self.pressure_controller_K,
        }
        case.solver_parameters = case.solver_parameters | {
            "max_time_step": self.max_time_step,
        }
        return case

    def run_case(
        self,
        case: CaseParameters,
    ) -> SimulationRecord:
        gas = self._initial_gas(case)

        area = case.geometry["area"]
        length = case.geometry["length"]
        inlet_velocity = case.controls["inlet_velocity"]
        pressure_controller_K = case.controls["pressure_controller_K"]
        n_steps = case.solver_parameters["n_steps"]
        max_time_step = case.solver_parameters["max_time_step"]

        mass_flow_rate = inlet_velocity * gas.density * area
        dz = length / n_steps

        reactor = ct.IdealGasReactor(gas, clone=True)
        reactor.volume = area * dz

        upstream = ct.Reservoir(gas, name="upstream", clone=True)
        downstream = ct.Reservoir(gas, name="downstream", clone=True)

        mass_flow_controller = ct.MassFlowController(
            upstream,
            reactor,
            mdot=mass_flow_rate,
        )
        _ = ct.PressureController(
            reactor,
            downstream,
            primary=mass_flow_controller,
            K=pressure_controller_K,
        )

        reactor_network = ct.ReactorNet([reactor])
        reactor_network.max_time_step = max_time_step

        states = ct.SolutionArray(reactor.phase, extra=["t", "z", "velocity"])
        states.append(
            reactor.phase.state,
            t=0.0,
            z=0.0,
            velocity=inlet_velocity,
        )

        tic = time.perf_counter()

        t = 0.0
        for step in range(1, n_steps + 1):
            upstream.phase.TDY = reactor.phase.TDY
            upstream.syncState()

            reactor_network.reinitialize()
            reactor_network.solve_steady()

            velocity = mass_flow_rate / (area * reactor.phase.density)
            t += reactor.mass / mass_flow_rate
            z = step * dz

            states.append(reactor.phase.state, t=t, z=z, velocity=velocity)

        toc = time.perf_counter()

        return self._record(case, states, toc - tic)


if True:
    pfr_lagrangian_simulator = PFRLagrangianParticleCaseSimulator()
    pfr_lagrangian_dataset_generator = SimulationDatasetGenerator(
        pfr_lagrangian_simulator,
        prog_path.parent.parent / "datasets" / "pfr_lagrangian_particle",
    )
    records_splits = pfr_lagrangian_dataset_generator.generate_splits(n_cases=20)
    pfr_lagrangian_dataset_generator.save_splits(records_splits, overwrite=True)

    # pfr_chain_simulator = PFRChainOfReactorsCaseSimulator()
    # pfr_chain_dataset_generator = SimulationDatasetGenerator(
    #     pfr_chain_simulator,
    #     prog_path.parent.parent / "datasets" / "pfr_chain_of_reactors",
    # )
    # records_splits = pfr_chain_dataset_generator.generate_splits(n_cases=20)
    # pfr_chain_dataset_generator.save_splits(records_splits, overwrite=False)
