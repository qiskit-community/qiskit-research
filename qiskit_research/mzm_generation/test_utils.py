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
from qiskit.quantum_info import random_hermitian, random_statevector
from qiskit_nature.operators.second_quantization import QuadraticHamiltonian
from qiskit_research.mzm_generation.utils import (
    correlation_matrix,
    expectation,
    expectation_from_correlation_matrix,
    jordan_wigner,
)


def _random_antisymmetric(dim: int):
    mat = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    return mat - mat.T


class TestMZMGenerationUtils(unittest.TestCase):
    """Test PhasedXXMinusYYGate."""

    def test_expectation_from_correlation_matrix(self):
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
