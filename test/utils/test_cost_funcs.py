# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test cost function."""

import unittest

from qiskit.circuit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
from qiskit_research.utils.cost_funcs import avg_error_score


class TestScaledCostFuncs(unittest.TestCase):
    """Test cost function from vf2_utils"""

    def test_cost_func(self):
        """Test average error cost function"""
        backend = FakeSherbrooke()
        target = backend.target

        qc = QuantumCircuit(5)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)

        qc2 = qc.copy()
        qc2.cx(2, 3)
        qc2.cx(3, 4)

        pm = generate_preset_pass_manager(
            target=target, optimization_level=2, seed_transpiler=12345
        )
        qc_t = pm.run(qc)
        best_layout_score = avg_error_score(qc_t, target)

        self.assertGreater(best_layout_score, 0)
        self.assertLess(best_layout_score, 1)

        qc2_t = pm.run(qc2)
        best_layout_score2 = avg_error_score(qc2_t, target)

        self.assertLess(best_layout_score, best_layout_score2)
