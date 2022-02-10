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

"""Test mzm_generation utils."""

import unittest

import numpy as np
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info import random_hermitian, random_statevector
from qiskit_nature.operators.second_quantization import QuadraticHamiltonian
from qiskit_research.mzm_generation.experiment import (
    CircuitParameters,
    KitaevHamiltonianExperiment,
)
from qiskit_research.mzm_generation.utils import (
    compute_correlation_matrix,
    correlation_matrix,
    counts_to_quasis,
    expectation,
    expectation_from_correlation_matrix,
    jordan_wigner,
    kitaev_hamiltonian,
)


def _random_antisymmetric(dim: int):
    mat = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    return mat - mat.T


class TestMZMGenerationUtils(unittest.TestCase):
    """Test PhasedXXMinusYYGate."""

    def test_expectation_from_correlation_matrix_exact(self):
        dim = 5

        hermitian_part = np.array(random_hermitian(5))
        antisymmetric_part = _random_antisymmetric(5)
        constant = np.random.randn()
        quad_ham = QuadraticHamiltonian(hermitian_part, antisymmetric_part, constant)

        hamiltonian = quad_ham._fermionic_op()
        hamiltonian_jw = jordan_wigner(hamiltonian).to_matrix()
        state = np.array(random_statevector(2 ** dim))
        corr = correlation_matrix(state)

        exp1, var1 = expectation_from_correlation_matrix(hamiltonian, corr)
        exp2, var2 = expectation_from_correlation_matrix(quad_ham, corr)
        exp_expected = expectation(hamiltonian_jw, state)
        np.testing.assert_allclose(exp1, exp_expected, atol=1e-8)
        np.testing.assert_allclose(exp2, exp_expected, atol=1e-8)
        np.testing.assert_allclose(var1, var2, atol=1e-8)

    def test_expectation_from_correlation_matrix_sample(self):
        n_modes = 5
        tunneling = -1.0
        superconducting = 1 + 2j
        chemical_potential = 1.0
        occupied_orbitals = ()
        experiment = KitaevHamiltonianExperiment(
            experiment_id="test",
            qubits=list(range(n_modes)),
            tunneling_values=[tunneling],
            superconducting_values=[superconducting],
            chemical_potential_values=[chemical_potential],
            occupied_orbitals_list=[occupied_orbitals],
        )
        backend = AerSimulator(method="statevector")
        experiment_data = experiment.run(backend=backend, shots=1000)
        experiment_data.block_for_results()
        data = {}
        for result in experiment_data.data():
            (
                _tunneling,
                _superconducting,
                _chemical_potential,
                _occupied_orbitals,
                permutation,
                measurement_label,
            ) = result["metadata"]["params"]
            params = CircuitParameters(
                tunneling=_tunneling,
                superconducting=_superconducting
                if isinstance(_superconducting, float)
                else complex(*_superconducting),
                chemical_potential=_chemical_potential,
                occupied_orbitals=tuple(_occupied_orbitals),
                permutation=tuple(permutation),
                measurement_label=measurement_label,
            )
            data[params] = result
        quasis = {}
        for permutation, label in experiment.measurement_labels():
            params = CircuitParameters(
                tunneling,
                superconducting,
                chemical_potential,
                occupied_orbitals,
                permutation,
                label,
            )
            counts = data[params]["counts"]
            quasis[permutation, label] = counts_to_quasis(counts)
        corr, cov = compute_correlation_matrix(quasis, experiment)
        quad_ham = kitaev_hamiltonian(
            n_modes,
            tunneling=tunneling,
            superconducting=superconducting,
            chemical_potential=chemical_potential,
        )
        hamiltonian = quad_ham._fermionic_op()
        exp1, var1 = expectation_from_correlation_matrix(quad_ham, corr, cov)
        exp2, var2 = expectation_from_correlation_matrix(hamiltonian, corr, cov)
        np.testing.assert_allclose(exp1, exp2, atol=1e-8)
        np.testing.assert_allclose(var1, var2, atol=1e-8)
