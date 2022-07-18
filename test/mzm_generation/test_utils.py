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

from __future__ import annotations

import unittest
from typing import Optional

import numpy as np
from qiskit.quantum_info import random_hermitian, random_statevector
from qiskit_nature.operators.second_quantization import QuadraticHamiltonian
from qiskit_research.mzm_generation.experiment import (
    CircuitParameters,
    KitaevHamiltonianExperiment,
    KitaevHamiltonianExperimentParameters,
)
from qiskit_research.mzm_generation.utils import (
    _CovarianceDict,
    compute_correlation_matrix,
    correlation_matrix_from_state_vector,
    counts_to_quasis,
    covariance_matrix,
    expectation,
    expectation_from_correlation_matrix,
    fidelity_witness,
    jordan_wigner,
    kitaev_hamiltonian,
    measurement_labels,
)


def _random_antisymmetric(dim: int):
    mat = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    return mat - mat.T


def _fidelity_witness_alt(
    corr: np.ndarray, corr_target: np.ndarray, cov: Optional[_CovarianceDict] = None
) -> tuple[float, float]:
    m, _ = corr.shape
    n = m // 2
    Tt = corr_target[:n, :n]
    St = corr_target[:n, n:]
    constant = np.trace(2 * Tt @ Tt - Tt - St @ St.conj() - St.conj() @ St)
    hermitian_part = np.eye(n) - 2 * Tt.conj()
    antisymmetric_part = -2 * St.conj()
    op = QuadraticHamiltonian(hermitian_part, antisymmetric_part)
    exp, std = expectation_from_correlation_matrix(op, corr, cov)
    return 1 - exp - constant, std


class TestMZMGenerationUtils(unittest.TestCase):
    """Test MZM utils."""

    def test_covariance_matrix(self):
        """Test computing covariance matrix."""
        dim = 5

        hermitian_part = np.array(random_hermitian(5))
        antisymmetric_part = _random_antisymmetric(5)
        constant = np.random.randn()
        quad_ham = QuadraticHamiltonian(hermitian_part, antisymmetric_part, constant)

        hamiltonian = quad_ham.to_fermionic_op()
        hamiltonian_jw = jordan_wigner(hamiltonian).to_matrix()
        _, vecs = np.linalg.eigh(hamiltonian_jw)
        state = vecs[:, 0]
        corr = correlation_matrix_from_state_vector(state)
        cov = covariance_matrix(corr)

        np.testing.assert_allclose(corr @ corr, corr, atol=1e-8)
        np.testing.assert_allclose(cov @ cov, -np.eye(2 * dim), atol=1e-8)

    def test_expectation_from_correlation_matrix_exact(self):
        """Test computing expectation value from exact correlation matrix."""
        dim = 5

        hermitian_part = np.array(random_hermitian(5))
        antisymmetric_part = _random_antisymmetric(5)
        constant = np.random.randn()
        quad_ham = QuadraticHamiltonian(hermitian_part, antisymmetric_part, constant)

        hamiltonian = quad_ham.to_fermionic_op()
        hamiltonian_jw = jordan_wigner(hamiltonian).to_matrix()
        state = np.array(random_statevector(2**dim))
        corr = correlation_matrix_from_state_vector(state)

        exp1, var1 = expectation_from_correlation_matrix(hamiltonian, corr)
        exp2, var2 = expectation_from_correlation_matrix(quad_ham, corr)
        exp_expected = expectation(hamiltonian_jw, state)
        np.testing.assert_allclose(exp1, exp_expected, atol=1e-8)
        np.testing.assert_allclose(exp2, exp_expected, atol=1e-8)
        np.testing.assert_allclose(var1, var2, atol=1e-8)

    def test_expectation_from_correlation_matrix_sample(self):
        """Test computing expectation value from experimental correlation matrix."""
        n_modes = 5
        tunneling = -1.0
        superconducting = 1 + 2j
        chemical_potential = 1.0
        occupied_orbitals = ()
        backend_name = "aer_simulator"
        params = KitaevHamiltonianExperimentParameters(
            timestamp="test",
            backend_name=backend_name,
            qubits=range(n_modes),
            n_modes=n_modes,
            tunneling_values=[tunneling],
            superconducting_values=[superconducting],
            chemical_potential_values=[chemical_potential],
            occupied_orbitals_list=[occupied_orbitals],
            dynamical_decoupling_sequences=None,
        )
        experiment = KitaevHamiltonianExperiment(params)

        experiment_data = experiment.run(shots=1000)
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
                dynamical_decoupling_sequence,
                pauli_twirl_index,
            ) = result["metadata"]["params"]
            params = CircuitParameters(
                tunneling=_tunneling,
                superconducting=_superconducting,
                chemical_potential=_chemical_potential,
                occupied_orbitals=tuple(_occupied_orbitals),
                permutation=tuple(permutation),
                measurement_label=measurement_label,
                dynamical_decoupling_sequence=dynamical_decoupling_sequence,
                pauli_twirl_index=pauli_twirl_index,
            )
            data[params] = result
        quasis = {}
        for permutation, label in measurement_labels(n_modes):
            params = CircuitParameters(
                tunneling,
                superconducting,
                chemical_potential,
                occupied_orbitals,
                permutation,
                label,
                dynamical_decoupling_sequence,
                pauli_twirl_index=None,
            )
            counts = data[params]["counts"]
            quasis[permutation, label] = counts_to_quasis(counts)
        corr, cov = compute_correlation_matrix(quasis)
        quad_ham = kitaev_hamiltonian(
            n_modes,
            tunneling=tunneling,
            superconducting=superconducting,
            chemical_potential=chemical_potential,
        )
        hamiltonian = quad_ham.to_fermionic_op()
        exp1, std1 = expectation_from_correlation_matrix(quad_ham, corr, cov)
        exp2, std2 = expectation_from_correlation_matrix(hamiltonian, corr, cov)
        np.testing.assert_allclose(exp1, exp2, atol=1e-8)
        np.testing.assert_allclose(std1, std2, atol=1e-8)

        # test fidelity witness
        transformation_matrix, _, _ = quad_ham.diagonalizing_bogoliubov_transform()
        W1 = transformation_matrix[:, :n_modes]
        W2 = transformation_matrix[:, n_modes:]
        full_transformation_matrix = np.block([[W1, W2], [W2.conj(), W1.conj()]])
        occupation = np.zeros(n_modes)
        occupation[list(occupied_orbitals)] = 1.0
        corr_diag = np.diag(np.concatenate([occupation, 1 - occupation]))
        corr_exact = (
            full_transformation_matrix.T.conj() @ corr_diag @ full_transformation_matrix
        )

        val1, std1 = fidelity_witness(corr, corr_exact, cov)
        val2, std2 = _fidelity_witness_alt(corr, corr_exact, cov)
        np.testing.assert_allclose(val1, val2, atol=1e-8)
        np.testing.assert_allclose(std1, std2, atol=1e-8)

    def test_real_valued_optimization(self):
        """Test that measurements are optimized when circuits are real."""
        n_modes = 3
        tunneling = -1.0
        superconducting = 1.0
        chemical_potential = 1.0
        occupied_orbitals = ()
        backend_name = "aer_simulator"
        params = KitaevHamiltonianExperimentParameters(
            timestamp="test",
            backend_name=backend_name,
            qubits=range(n_modes),
            n_modes=n_modes,
            tunneling_values=[tunneling],
            superconducting_values=[superconducting],
            chemical_potential_values=[chemical_potential],
            occupied_orbitals_list=[occupied_orbitals],
            dynamical_decoupling_sequences=None,
        )
        experiment = KitaevHamiltonianExperiment(params)
        self.assertEqual(len(experiment.circuits()), 9)
