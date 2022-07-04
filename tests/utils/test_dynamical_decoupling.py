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

from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.providers.fake_provider import FakeMumbai
from qiskit_research.utils.convenience import add_dynamical_decoupling


def test_add_dynamical_decoupling():
    """Test adding dynamical decoupling."""
    # TODO make this an actual test
    backend = FakeMumbai()
    circuit = QuantumCircuit(3)
    circuit.cx(0, 1)
    circuit.rz(1.0, 1)
    circuit.cx(0, 1)
    circuit.rx(1.0, [0, 1, 2])
    transpiled = transpile(circuit, backend)
    transpiled_dd = add_dynamical_decoupling(transpiled, backend, "XY4pm")
    assert isinstance(transpiled_dd, QuantumCircuit)
