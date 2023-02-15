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

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import XGate

from qiskit.transpiler.instruction_durations import InstructionDurations
from qiskit.transpiler.passes.scheduling import ALAPScheduleAnalysis
from qiskit.transpiler import PassManager

from qiskit_research.utils.periodic_dynamical_decoupling import (
    PeriodicDynamicalDecoupling,
)


class TestPeriodicDynamicalDecoupling(unittest.TestCase):
    """Test PeriodicDynamicalDecoupling pass."""

    @unittest.skip("string comparison is flaky for some reason")
    def test_add_periodic_dynamical_decoupling(self):
        """Test adding XX sequence with max 3 repeats and min_avg_delay"""
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

        # pylint: disable=trailing-whitespace
        self.assertEqual(
            str(circ_dd.draw()).strip(),
            """
              ┌───┐           ┌───────────────┐ ┌───┐┌───────────────┐ ┌───┐»
   q_0: ──────┤ H ├────────■──┤ Delay(25[dt]) ├─┤ X ├┤ Delay(75[dt]) ├─┤ X ├»
        ┌─────┴───┴─────┐┌─┴─┐└───────────────┘ └───┘└───────────────┘ └───┘»
   q_1: ┤ Delay(50[dt]) ├┤ X ├──────────────────────────────────────────────»
        ├───────────────┤├───┤┌───────────────┐ ┌───┐┌───────────────┐ ┌───┐»
   q_2: ┤ Delay(25[dt]) ├┤ X ├┤ Delay(75[dt]) ├─┤ X ├┤ Delay(75[dt]) ├─┤ X ├»
        ├───────────────┤├───┤├───────────────┴┐├───┤├───────────────┴┐├───┤»
   q_3: ┤ Delay(50[dt]) ├┤ X ├┤ Delay(100[dt]) ├┤ X ├┤ Delay(100[dt]) ├┤ X ├»
        └───────────────┘└───┘└────────────────┘└───┘└────────────────┘└───┘»
meas: 4/════════════════════════════════════════════════════════════════════»
                                                                            »
«        ┌────────────────┐┌───┐┌───────────────┐ ┌───┐┌───────────────┐      »
«   q_0: ┤ Delay(100[dt]) ├┤ X ├┤ Delay(75[dt]) ├─┤ X ├┤ Delay(25[dt]) ├──────»
«        └────────────────┘└───┘└───────────────┘ └───┘└───────────────┘      »
«   q_1: ─────────────────────────────────────────────────────────────────────»
«        ┌────────────────┐┌───┐┌───────────────┐ ┌───┐┌───────────────┐ ┌───┐»
«   q_2: ┤ Delay(100[dt]) ├┤ X ├┤ Delay(75[dt]) ├─┤ X ├┤ Delay(75[dt]) ├─┤ X ├»
«        ├────────────────┤├───┤├───────────────┴┐├───┤├───────────────┴┐├───┤»
«   q_3: ┤ Delay(150[dt]) ├┤ X ├┤ Delay(100[dt]) ├┤ X ├┤ Delay(100[dt]) ├┤ X ├»
«        └────────────────┘└───┘└────────────────┘└───┘└────────────────┘└───┘»
«meas: 4/═════════════════════════════════════════════════════════════════════»
«                                                                             »
«                                                                           »
«   q_0: ───────────────────────────────────────────────────────────────────»
«                              ┌───────────────┐┌───┐┌────────────────┐┌───┐»
«   q_1: ───────────────────■──┤ Delay(50[dt]) ├┤ X ├┤ Delay(100[dt]) ├┤ X ├»
«        ┌───────────────┐┌─┴─┐└───────────────┘└───┘└────────────────┘└───┘»
«   q_2: ┤ Delay(25[dt]) ├┤ X ├────────■────────────────────────────────────»
«        ├───────────────┤└───┘      ┌─┴─┐                                  »
«   q_3: ┤ Delay(50[dt]) ├───────────┤ X ├──────────────────────────────────»
«        └───────────────┘           └───┘                                  »
«meas: 4/═══════════════════════════════════════════════════════════════════»
«                                                                           »
«                          ░ ┌─┐         
«   q_0: ──────────────────░─┤M├─────────
«        ┌───────────────┐ ░ └╥┘┌─┐      
«   q_1: ┤ Delay(50[dt]) ├─░──╫─┤M├──────
«        └───────────────┘ ░  ║ └╥┘┌─┐   
«   q_2: ──────────────────░──╫──╫─┤M├───
«                          ░  ║  ║ └╥┘┌─┐
«   q_3: ──────────────────░──╫──╫──╫─┤M├
«                          ░  ║  ║  ║ └╥┘
«meas: 4/═════════════════════╩══╩══╩══╩═
«                             0  1  2  3 
""".strip(),
        )
