# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test mzm_generation analysis."""

from __future__ import annotations

import os

import mthree
import numpy as np
from qiskit_research.mzm_generation import KitaevHamiltonianAnalysis
from qiskit_research.mzm_generation.experiment import (
    KitaevHamiltonianExperiment,
    KitaevHamiltonianExperimentParameters,
)


def test_kitaev_hamiltonian_analysis(tmp_path):
    """Test KitaevHamiltonianAnalysis."""
    n_modes = 5
    tunneling = -1.0
    superconducting = 1.0
    chemical_potential = 1.0
    occupied_orbitals = ()
    backend_name = "aer_simulator"
    qubits = range(n_modes)
    params = KitaevHamiltonianExperimentParameters(
        timestamp="test",
        backend_name=backend_name,
        qubits=qubits,
        n_modes=n_modes,
        tunneling_values=[tunneling],
        superconducting_values=[superconducting],
        chemical_potential_values=[chemical_potential],
        occupied_orbitals_list=[occupied_orbitals],
        dynamical_decoupling_sequences=None,
        basedir=tmp_path.as_posix(),
        seed=1234,
    )
    experiment = KitaevHamiltonianExperiment(params)

    mit = mthree.M3Mitigation(experiment.backend)
    mit.cals_from_system(qubits, shots=1000, async_cal=True)
    filename = os.path.join(
        tmp_path, "data", "readout_calibration", backend_name, "test.json"
    )
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    mit.cals_to_file(filename)

    experiment_data = experiment.run(shots=1000)
    experiment_data.block_for_results()
    analysis = KitaevHamiltonianAnalysis()
    experiment_data = analysis.run(experiment_data, replace_results=True)
    experiment_data.block_for_results()
    fidelity_witness_avg = experiment_data.analysis_results(
        "fidelity_witness_avg"
    ).value
    energy_error = experiment_data.analysis_results("energy_error").value
    values, _ = fidelity_witness_avg[None]["pur"]
    np.testing.assert_allclose(values, 1.0, atol=1e-2)
    values, _ = energy_error[None]["pur"]
    np.testing.assert_allclose(values, 0.0, atol=1e-2)
