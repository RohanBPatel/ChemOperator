from __future__ import annotations

import cantera as ct
import numpy as np

from chem_operator.datasets import CaseParameters
from chem_operator.reactors.cstr.dataset_generator import (
    CSTRCaseSimulator,
    NonIsothermalCSTRCaseSimulator,
)
from chem_operator.reactors.pfr.dataset_generator import (
    PFRChainOfReactorsSim,
    PFRNonIsothermalChainOfReactorsSim,
)


def test_non_isothermal_dataset_prefixes():
    assert NonIsothermalCSTRCaseSimulator.name == "cstr_non_isothermal"
    assert (
        PFRNonIsothermalChainOfReactorsSim.name
        == "pfr_non_isothermal_chain_of_reactors"
    )


def test_cstr_wall_cools_an_inert_reactor():
    case = CaseParameters(
        initial_conditions={
            "gas/T": 1000.0,
            "gas/P": ct.one_atm,
            "gas/X": {"HE": 1.0},
        },
        geometry={"reactor_volume": 1.0e-3, "wall_area": 1.0e-2},
        controls={
            "residence_time": 1.0,
            "pressure_controller_K": 1.0e-6,
            "solve_energy": True,
            "ambient_temperature": 300.0,
            "heat_transfer_coefficient": 10.0,
        },
        solver_parameters={"t_final": 0.1, "dt": 0.05, "adaptive": False},
        mechanism_parameters={},
    )

    record = CSTRCaseSimulator().run_case(case)

    np.testing.assert_allclose(record.fields["T"][0], 1000.0)
    assert record.fields["T"][-1] < record.fields["T"][0]


def test_pfr_wall_cools_an_inert_reactor():
    case = CaseParameters(
        initial_conditions={
            "gas/T": 1000.0,
            "gas/P": ct.one_atm,
            "gas/X": {"AR": 1.0},
        },
        geometry={
            "length": 0.03,
            "area": 1.0e-3,
            "wall_area_per_volume": 100.0,
        },
        controls={
            "inlet_velocity": 0.1,
            "pressure_controller_K": 1.0e-12,
            "solve_energy": True,
            "ambient_temperature": 300.0,
            "heat_transfer_coefficient": 100.0,
        },
        solver_parameters={"n_steps": 3, "max_time_step": 1.0e4},
        mechanism_parameters={},
    )

    record = PFRChainOfReactorsSim().run_case(case)

    assert np.all(np.diff(record.fields["T"]) < 0.0)
    assert record.fields["T"][-1] < case.initial_conditions["gas/T"]


def test_pfr_without_new_controls_keeps_energy_enabled():
    case = CaseParameters(
        initial_conditions={
            "gas/T": 1000.0,
            "gas/P": ct.one_atm,
            "gas/X": {"AR": 1.0},
        },
        geometry={"length": 0.02, "area": 1.0e-3},
        controls={
            "inlet_velocity": 0.1,
            "pressure_controller_K": 1.0e-12,
        },
        solver_parameters={"n_steps": 2, "max_time_step": 1.0e4},
        mechanism_parameters={},
    )

    record = PFRChainOfReactorsSim().run_case(case)

    np.testing.assert_allclose(record.fields["T"], 1000.0, rtol=1.0e-12)
