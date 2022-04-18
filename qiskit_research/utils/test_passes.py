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

"""Test Pauli twirling."""

import unittest

import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import XXPlusYYGate
from qiskit.transpiler import PassManager
from qiskit.quantum_info import Operator
from qiskit.test.mock import FakeMumbai
from qiskit_research.utils.passes import XXPlusYYtoRZX


class TestPasses(unittest.TestCase):
    """Test passes."""

    def test_xxplusyy_to_rzx(self):
        theta = np.random.uniform(-10, 10)
        beta = np.random.uniform(-10, 10)
        gate = XXPlusYYGate(theta, beta)
        register = QuantumRegister(2)
        circuit = QuantumCircuit(register)
        circuit.append(gate, register)
        pass_ = XXPlusYYtoRZX()
        pass_manager = PassManager([pass_])
        decomposed = pass_manager.run(circuit)
        self.assertTrue(Operator(circuit).equiv(Operator(decomposed)))
