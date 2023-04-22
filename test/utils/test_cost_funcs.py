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

"""Test pulse scaling."""

import unittest

from mapomatic import deflate_circuit, evaluate_layouts, matching_layouts
import numpy as np

from qiskit import transpile
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.providers.fake_provider import FakeWashington

from qiskit_research.utils.gates import SECRGate
from qiskit_research.utils.convenience import (
    add_dynamical_decoupling,
    attach_cr_pulses,
)
from qiskit_research.utils.cost_funcs import cost_func_scaled_cr


class TestScaledCostFuncs(unittest.TestCase):
    """Test cost functions for scaled pulses."""

    def test_cost_func_rzx(self):
        """Test cost function for RZX"""
        backend = FakeWashington()
        rng = np.random.default_rng(12345)

        phi = Parameter("$\\phi$")
        theta = Parameter("$\\theta$")

        qc = QuantumCircuit(3)
        qc.rzx(theta, 0, 1)
        qc.rzx(phi, 1, 2)

        qc2 = qc.copy()
        qc2.rzx(theta, 0, 1)
        qc2.rzx(phi, 1, 2)

        param_bind = {
            phi: rng.uniform(0.2, 0.8),
            theta: rng.uniform(0.2, 0.8),
        }

        qc_routed = transpile(qc, backend, optimization_level=3)
        qc_bound = attach_cr_pulses(qc_routed, backend, param_bind=param_bind)
        qc_sched = transpile(
            qc_bound, backend, optimization_level=0, scheduling_method="alap"
        )
        layouts = matching_layouts(deflate_circuit(qc_sched), backend)
        best_layout = evaluate_layouts(
            deflate_circuit(qc_sched),
            layouts,
            backend,
            cost_function=cost_func_scaled_cr,
        )[0]

        self.assertGreater(best_layout[1], 0)
        self.assertLess(best_layout[1], 1)

        qc2_routed = transpile(qc2, backend, initial_layout=best_layout[0])
        qc2_bound = attach_cr_pulses(qc2_routed, backend, param_bind=param_bind)
        qc2_sched = transpile(
            qc2_bound, backend, optimization_level=0, scheduling_method="alap"
        )
        best_layout2 = evaluate_layouts(
            deflate_circuit(qc2_sched),
            best_layout[0],
            backend,
            cost_function=cost_func_scaled_cr,
        )[0]

        self.assertLess(best_layout[1], best_layout2[1])

    def test_cost_func_secr(self):
        """Test cost function for RZX"""
        backend = FakeWashington()
        rng = np.random.default_rng(12345)

        phi = Parameter("$\\phi$")
        theta = Parameter("$\\theta$")

        qc = QuantumCircuit(3)
        qc.append(SECRGate(theta), [0, 1])
        qc.append(SECRGate(phi), [1, 2])

        qc2 = qc.copy()
        qc2.append(SECRGate(theta), [0, 1])
        qc2.append(SECRGate(phi), [1, 2])

        param_bind = {
            phi: rng.uniform(0.2, 0.8),
            theta: rng.uniform(0.2, 0.8),
        }

        qc_routed = transpile(qc, backend, optimization_level=3)
        qc_bound = attach_cr_pulses(qc_routed, backend, param_bind=param_bind)
        qc_sched = transpile(
            qc_bound, backend, optimization_level=0, scheduling_method="alap"
        )
        layouts = matching_layouts(deflate_circuit(qc_sched), backend)
        best_layout = evaluate_layouts(
            deflate_circuit(qc_sched),
            layouts,
            backend,
            cost_function=cost_func_scaled_cr,
        )[0]

        self.assertGreater(best_layout[1], 0)
        self.assertLess(best_layout[1], 1)

        qc2_routed = transpile(qc2, backend, initial_layout=best_layout[0])
        qc2_bound = attach_cr_pulses(qc2_routed, backend, param_bind=param_bind)
        qc2_sched = transpile(
            qc2_bound, backend, optimization_level=0, scheduling_method="alap"
        )
        best_layout2 = evaluate_layouts(
            deflate_circuit(qc2_sched),
            best_layout[0],
            backend,
            cost_function=cost_func_scaled_cr,
        )[0]

        self.assertLess(best_layout[1], best_layout2[1])

    def test_cost_func_dd(self):
        """Test cost function for RZX"""
        backend = FakeWashington()

        qc = QuantumCircuit(5)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)

        qc2 = qc.copy()
        qc2.cx(2, 3)
        qc2.cx(3, 4)

        layout = [0, 1, 2, 3, 4]
        qc_t = transpile(qc, backend, initial_layout=layout)
        qc_dd = add_dynamical_decoupling(qc_t, backend, "XY4pm")
        best_layout = evaluate_layouts(
            qc_dd,
            layout,
            backend,
            cost_function=cost_func_scaled_cr,
        )[0]

        self.assertGreater(best_layout[1], 0)
        self.assertLess(best_layout[1], 1)

        qc2_t = transpile(qc2, backend, initial_layout=layout)
        qc2_dd = add_dynamical_decoupling(qc2_t, backend, "XY4pm")
        best_layout2 = evaluate_layouts(
            qc2_dd,
            layout,
            backend,
            cost_function=cost_func_scaled_cr,
        )[0]

        self.assertLess(best_layout[1], best_layout2[1])


# TODO: Add unit test for ECR gate (i.e., ibm_sherbrooke) when resolved:
# https://github.com/Qiskit/qiskit-terra/issues/9553
