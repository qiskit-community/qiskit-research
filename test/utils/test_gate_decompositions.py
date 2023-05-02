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
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import RZZGate, XXMinusYYGate, XXPlusYYGate
from qiskit.quantum_info import Operator
from qiskit.transpiler import PassManager

from qiskit_research.utils.gate_decompositions import (
    ControlledRZZToCX,
    RZXWeylDecomposition,
    XXMinusYYtoRZX,
    XXPlusYYtoRZX,
)


class TestPasses(unittest.TestCase):
    """Test passes."""

    def test_controlled_rzz_to_cx(self):
        """Test controlled RZZGate to CXGate decomposition."""
        rng = np.random.default_rng()
        theta = rng.uniform(-10, 10)
        gate = RZZGate(theta).control(1)
        register = QuantumRegister(3)
        circuit = QuantumCircuit(register)
        circuit.append(gate, register)
        pass_ = ControlledRZZToCX()
        pass_manager = PassManager([pass_])
        decomposed = pass_manager.run(circuit)
        self.assertTrue(Operator(circuit).equiv(Operator(decomposed)))

    def test_xxplusyy_to_rzx(self):
        """Test XXPlusYYGate to RZXGate decomposition."""
        rng = np.random.default_rng()
        theta = rng.uniform(-10, 10)
        beta = rng.uniform(-10, 10)
        gate = XXPlusYYGate(theta, beta)
        register = QuantumRegister(2)
        circuit = QuantumCircuit(register)
        circuit.append(gate, register)
        pass_ = XXPlusYYtoRZX()
        pass_manager = PassManager([pass_])
        decomposed = pass_manager.run(circuit)
        self.assertTrue(Operator(circuit).equiv(Operator(decomposed)))

    def test_xxminusyy_to_rzx(self):
        """Test XXMinusYYGate to RZXGate decomposition."""
        rng = np.random.default_rng()
        theta = rng.uniform(-10, 10)
        beta = rng.uniform(-10, 10)
        gate = XXMinusYYGate(theta, beta)
        register = QuantumRegister(2)
        circuit = QuantumCircuit(register)
        circuit.append(gate, register)
        pass_ = XXMinusYYtoRZX()
        pass_manager = PassManager([pass_])
        decomposed = pass_manager.run(circuit)
        self.assertTrue(Operator(circuit).equiv(Operator(decomposed)))

    def test_rzx_weyl_decomposition(self):
        """Test RZXWeylDecomposition."""

        qc = QuantumCircuit(3)
        qc.rxx(np.pi / 3, 0, 1)
        qc.ryy(np.pi / 5, 1, 2)
        qc.rzz(np.pi / 7, 2, 0)
        pm = PassManager(RZXWeylDecomposition())
        qc_w = pm.run(qc)

        self.assertNotIn("rxx", qc_w.count_ops())
        self.assertNotIn("ryy", qc_w.count_ops())
        self.assertNotIn("rzz", qc_w.count_ops())
        self.assertIn("rzx", qc_w.count_ops())
        self.assertTrue(np.allclose(Operator(qc), Operator(qc_w), atol=1e-8))
