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
from qiskit.providers.fake_provider import FakeWashington
from qiskit_research.utils.convenience import (
    add_dynamical_decoupling,
    add_pulse_calibrations,
)


def test_add_dynamical_decoupling():
    """Test adding dynamical decoupling."""
    # TODO make this an actual test
    circuit = QuantumCircuit(3)
    circuit.cx(0, 1)
    circuit.rz(1.0, 1)
    circuit.cx(0, 1)
    circuit.rx(1.0, [0, 1, 2])

    backend = FakeWashington()
    transpiled = transpile(circuit, backend)
    transpiled_dd = add_dynamical_decoupling(
        transpiled, backend, "XY4pm", add_pulse_cals=True
    )
    assert isinstance(transpiled_dd, QuantumCircuit)


def test_add_pulse_calibrations():
    """Test adding dynamical decoupling."""
    circuit = QuantumCircuit(2)
    backend = FakeWashington()
    add_pulse_calibrations(circuit, backend)
    for key in circuit.calibrations["xp"]:
        drag_xp = circuit.calibrations["xp"][key].instructions[0][1].operands[0]
        drag_xm = circuit.calibrations["xm"][key].instructions[0][1].operands[0]
        drag_yp = circuit.calibrations["yp"][key].instructions[1][1].operands[0]
        drag_ym = circuit.calibrations["ym"][key].instructions[1][1].operands[0]
        assert drag_xm.amp == -drag_xp.amp
        assert drag_yp.amp == drag_xp.amp
        assert drag_ym.amp == -drag_xp.amp
