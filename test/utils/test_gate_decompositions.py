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
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel
from qiskit.circuit.library import XXMinusYYGate, XXPlusYYGate
from qiskit.opflow import I, X, Y, Z, PauliTrotterEvolution, Suzuki
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import UnrollCustomDefinitions
from qiskit.quantum_info import Operator
from qiskit_research.utils.gate_decompositions import (
    RZXWeylDecomposition,
    XXMinusYYtoRZX,
    XXPlusYYtoRZX,
)


class TestPasses(unittest.TestCase):
    """Test passes."""

    def test_xxplusyy_to_rzx(self):
        """Test XXPlusYYGate to RZXGate decomposition."""
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

    def test_xxminusyy_to_rzx(self):
        """Test XXMinusYYGate to RZXGate decomposition."""
        theta = np.random.uniform(-10, 10)
        beta = np.random.uniform(-10, 10)
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

        JJ = np.random.uniform(-10, 10)
        hh = np.random.uniform(-10, 10)
        tt = np.random.uniform(0, 10)

        ham = -JJ * sum(
            [
                I ^ X ^ X,
                I ^ Y ^ Y,
                I ^ Z ^ Z,
                X ^ X ^ I,
                Y ^ Y ^ I,
                Z ^ Z ^ I,
            ]
        ) + hh * sum([I ^ I ^ X, I ^ X ^ I, X ^ I ^ I])
        U_ham = (ham * tt).exp_i()

        trot_circ = (
            PauliTrotterEvolution(trotter_mode=Suzuki(order=2, reps=1))
            .convert(U_ham)
            .to_circuit()
        )
        basis_gates = ["rx", "rz", "rxx", "ryy", "rzz"]
        pm = PassManager(
            [UnrollCustomDefinitions(sel, basis_gates), RZXWeylDecomposition()]
        )
        trot_circ_w = pm.run(trot_circ)

        self.assertTrue(Operator(trot_circ).equiv(Operator(trot_circ_w)))
