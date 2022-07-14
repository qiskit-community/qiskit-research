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

import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.providers.fake_provider import FakeMumbai
from qiskit.quantum_info import Operator
from qiskit_research.utils.convenience import scale_cr_pulses


class TestPulseScaling(unittest.TestCase):
    """Test pulse scaling."""

    def test_rzx_to_secr_forward(self):
        """Test pulse scaling RZX with forward SECR."""
        backend = FakeMumbai()
        rng = np.random.default_rng()

        JJ = Parameter("$J$")
        hh = Parameter("$h$")
        dt = Parameter("$dt$")

        param_bind = {
            JJ: rng.uniform(0, 1),
            hh: rng.uniform(0, 2),
            dt: rng.uniform(0, 0.5),
        }

        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.rz(-2 * JJ * dt, 1)
        qc.cx(0, 1)
        qc.rx(2 * hh * dt, [0, 1, 2])
        qc.cx(1, 2)
        qc.rz(-2 * JJ * dt, 2)
        qc.cx(1, 2)

        scaled_qc = scale_cr_pulses(qc, backend, unroll_rzx_to_ecr=True)
        scaled_qc.assign_parameters(param_bind, inplace=True)
        qc.assign_parameters(param_bind, inplace=True)

        self.assertTrue(Operator(qc).equiv(Operator(scaled_qc)))

    def test_rzx_to_secr_reverse(self):
        """Test pulse scaling RZX with reverse SECR."""
        backend = FakeMumbai()
        rng = np.random.default_rng()

        JJ = Parameter("$J$")
        hh = Parameter("$h$")
        dt = Parameter("$dt$")

        param_bind = {
            JJ: rng.uniform(0, 1),
            hh: rng.uniform(0, 2),
            dt: rng.uniform(0, 0.5),
        }

        qc = QuantumCircuit(3)
        qc.cx(2, 1)
        qc.rz(-2 * JJ * dt, 1)
        qc.cx(2, 1)
        qc.rx(2 * hh * dt, [0, 1, 2])
        qc.cx(1, 0)
        qc.rz(-2 * JJ * dt, 0)
        qc.cx(1, 0)

        scaled_qc = scale_cr_pulses(
            qc, backend, unroll_rzx_to_ecr=True, param_bind=param_bind
        )
        qc.assign_parameters(param_bind, inplace=True)

        self.assertTrue(Operator(qc).equiv(Operator(scaled_qc)))

    def test_rzx_to_secr(self):
        """Test pulse scaling with RZX gates."""
        backend = FakeMumbai()
        rng = np.random.default_rng()
        theta = rng.uniform(-np.pi, np.pi)

        qc = QuantumCircuit(2)
        qc.rzx(theta, 0, 1)

        scaled_qc = scale_cr_pulses(qc, backend, unroll_rzx_to_ecr=True, param_bind={})
        self.assertTrue(Operator(qc).equiv(Operator(scaled_qc)))

        qc = QuantumCircuit(2)
        qc.rzx(theta, 1, 0)

        scaled_qc = scale_cr_pulses(qc, backend, unroll_rzx_to_ecr=True, param_bind={})
        self.assertTrue(Operator(qc).equiv(Operator(scaled_qc)))
