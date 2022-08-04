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
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import XGate, YGate
from qiskit.providers.fake_provider import FakeWashington

from qiskit.transpiler.instruction_durations import InstructionDurations
from qiskit.transpiler.passes.scheduling import ALAPScheduleAnalysis
from qiskit.transpiler import PassManager

from qiskit_research.utils.custom_passes.periodic_dynamical_decoupling import (
    PeriodicDynamicalDecoupling,
)


class TestPeriodicDynamicalDecoupling(unittest.TestCase):
    def test_add_periodic_dynamical_decoupling(self):
        circuit = QuantumCircuit(4)
        circuit.h(0)
        for i in range(3):
            circuit.cx(i, i + 1)
        circuit.measure_all()

        durations = InstructionDurations(
            [
                ("h", 0, 50),
                ("cx", [0, 1], 700),
                ("cx", [1, 2], 200),
                ("cx", [2, 3], 300),
                ("x", None, 50),
                ("measure", None, 1000),
                ("reset", None, 1500),
            ]
        )
        pulse_alignment = 25

        pm = PassManager(
            [
                ALAPScheduleAnalysis(durations=durations),
                PeriodicDynamicalDecoupling(
                    durations=durations,
                    base_dd_sequence=[XGate(), XGate()],
                    max_repeats=3,
                    avg_min_delay=50,
                    pulse_alignment=pulse_alignment,
                    skip_reset_qubits=False,
                ),
            ]
        )
        circ_dd = pm.run(circuit)

        test_circ = QuantumCircuit(4)
        test_circ.h(0)

        test_circ.delay(50, 1)

        test_circ.cx(0, 1)

        test_circ.delay(25, 2)
        for i in range(2):
            test_circ.x(2)
            test_circ.delay(75, 2)
        test_circ.x(2)
        test_circ.delay(100, 2)
        for i in range(2):
            test_circ.x(2)
            test_circ.delay(75, 2)
        test_circ.x(2)
        test_circ.delay(25, 2)

        test_circ.cx(1, 2)

        test_circ.delay(50, 3)
        for i in range(2):
            test_circ.x(3)
            test_circ.delay(100, 3)
        test_circ.x(3)
        test_circ.delay(150, 3)
        for i in range(2):
            test_circ.x(3)
            test_circ.delay(100, 3)
        test_circ.x(3)
        test_circ.delay(50, 3)

        test_circ.cx(2, 3)

        test_circ.delay(25, 0)
        test_circ.x(0)
        test_circ.delay(75, 0)
        test_circ.x(0)
        test_circ.delay(100, 0)
        test_circ.x(0)
        test_circ.delay(75, 0)
        test_circ.x(0)
        test_circ.delay(25, 0)

        test_circ.delay(50, 1)
        test_circ.x(1)
        test_circ.delay(100, 1)
        test_circ.x(1)
        test_circ.delay(50, 1)

        test_circ.measure_all()

        self.assertTrue(circ_dd == test_circ)
