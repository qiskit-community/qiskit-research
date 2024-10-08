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

from qiskit.circuit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

from qiskit_research.utils.convenience import (
    add_dynamical_decoupling,
    add_pulse_calibrations,
)
from qiskit_research.utils.dynamical_decoupling import PulseMethod


class TestDynamicalDecoupling(unittest.TestCase):
    """Test PeriodicDynamicalDecoupling pass."""

    def test_add_dynamical_decoupling(self):
        """Test adding dynamical decoupling."""
        # TODO make this an actual test
        circuit = QuantumCircuit(3)
        circuit.h(1)
        circuit.cx(1, 2)
        circuit.cx(0, 1)
        circuit.rz(1.0, 1)
        circuit.cx(0, 1)
        circuit.rx(1.0, [0, 1, 2])

        backend = FakeSherbrooke()
        target = backend.target
        pm = generate_preset_pass_manager(target=target, optimization_level=2)
        transpiled = pm.run(circuit)
        transpiled_dd = add_dynamical_decoupling(
            transpiled, target, "XY4pm", add_pulse_cals=True
        )
        self.assertIsInstance(transpiled_dd, QuantumCircuit)
        self.assertIn("xp", transpiled_dd.count_ops())
        self.assertIn("yp", transpiled_dd.count_ops())
        self.assertIn("xm", transpiled_dd.count_ops())
        self.assertIn("ym", transpiled_dd.count_ops())

    def test_add_urdd_dynamical_decoupling(self):
        """Test adding dynamical decoupling."""
        # TODO make this an actual test
        circuit = QuantumCircuit(3)
        circuit.h(1)
        circuit.cx(1, 2)
        circuit.cx(0, 1)
        circuit.rz(1.0, 1)
        circuit.cx(0, 1)
        circuit.rx(1.0, [0, 1, 2])

        backend = FakeSherbrooke()
        target = backend.target
        pm = generate_preset_pass_manager(2, backend)
        transpiled = pm.run(circuit)
        transpiled_urdd4 = add_dynamical_decoupling(
            transpiled,
            target,
            "URDD",
            add_pulse_cals=True,
            urdd_pulse_num=4,
            pulse_method=PulseMethod.AMPLITUDEINVERT,
        )
        self.assertIsInstance(transpiled_urdd4, QuantumCircuit)
        self.assertIn("pi_phi", transpiled_urdd4.count_ops())
        self.assertEqual(transpiled_urdd4.count_ops()["pi_phi"] % 4, 0)

        transpiled_urdd8 = add_dynamical_decoupling(
            transpiled,
            target,
            "URDD",
            add_pulse_cals=True,
            urdd_pulse_num=8,
            pulse_method=PulseMethod.IQSUM,
        )
        self.assertIsInstance(transpiled_urdd8, QuantumCircuit)
        self.assertIn("pi_phi", transpiled_urdd8.count_ops())
        self.assertEqual(transpiled_urdd8.count_ops()["pi_phi"] % 8, 0)

    def test_add_pulse_calibrations(self):
        """Test adding dynamical decoupling."""
        circuit = QuantumCircuit(2)
        backend = FakeSherbrooke()
        target = backend.target
        add_pulse_calibrations(circuit, target)
        for key in circuit.calibrations["xp"]:
            drag_xp = circuit.calibrations["xp"][key].instructions[0][1].operands[0]
            drag_xm = circuit.calibrations["xm"][key].instructions[0][1].operands[0]
            drag_yp = circuit.calibrations["yp"][key].instructions[1][1].operands[0]
            drag_ym = circuit.calibrations["ym"][key].instructions[1][1].operands[0]
            self.assertEqual(drag_xm.amp, -drag_xp.amp)
            self.assertEqual(drag_yp.amp, drag_xp.amp)
            self.assertEqual(drag_ym.amp, -drag_yp.amp)
