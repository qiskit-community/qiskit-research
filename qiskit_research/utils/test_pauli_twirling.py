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

"""Test dynamical decoupling."""

import unittest

from qiskit import transpile
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.test.mock import FakeMumbai
from qiskit_research.utils.pauli_twirling import add_pauli_twirls, TWIRL_GATES

import numpy as np


class TestPauliTwirling(unittest.TestCase):
    """Test Pauli twirling."""

    def test_add_pauli_twirls_to_rzx(self):
        backend = FakeMumbai()
        theta = Parameter('$\\theta$')
        phi = Parameter('$\\phi')
        circuit = QuantumCircuit(3)
        circuit.h([1, 2])
        circuit.rzx(theta, 0, 1)
        circuit.h(1)
        circuit.rzx(phi, 1, 2)
        circuit.h(2)
        twirled_circs = add_pauli_twirls(circuit, backend, "rzx", 5, 1)

        rng = np.random.default_rng()
        param_bind = {}
        for param in circuit.parameters:
            param_bind[param] = rng.random()

        circuit.assign_parameters(param_bind, inplace=True)
        for t_circ in twirled_circs[0]:
            t_circ.assign_parameters(param_bind, inplace=True)
            self.assertNotEqual(t_circ, circuit)
            self.assertTrue(Operator(t_circ).equiv(Operator(circuit)))

    def test_add_pauli_twirls_to_rzz(self):
        backend = FakeMumbai()
        theta = Parameter('$\\theta$')
        phi = Parameter('$\\phi')
        circuit = QuantumCircuit(3)
        circuit.h([1, 2])
        circuit.rzz(theta, 0, 1)
        circuit.h(1)
        circuit.rzz(phi, 1, 2)
        circuit.h(2)
        twirled_circs = add_pauli_twirls(circuit, backend, "rzz", 5, 1)

        rng = np.random.default_rng()
        param_bind = {}
        for param in circuit.parameters:
            param_bind[param] = rng.random()

        circuit.assign_parameters(param_bind, inplace=True)
        for t_circ in twirled_circs[0]:
            t_circ.assign_parameters(param_bind, inplace=True)
            self.assertNotEqual(t_circ, circuit)
            self.assertTrue(Operator(t_circ).equiv(Operator(circuit)))
