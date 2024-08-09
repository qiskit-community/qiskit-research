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
from ddt import data, ddt
from qiskit import schedule
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.pulse import ControlChannel, Play
from qiskit.quantum_info import Operator
from qiskit.transpiler import PassManager, Target
from qiskit.transpiler.passes import Optimize1qGatesDecomposition
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

from qiskit_research.utils.convenience import scale_cr_pulses
from qiskit_research.utils.gate_decompositions import RZXtoEchoedCR
from qiskit_research.utils.pulse_scaling import (
    BASIS_GATES,
    ReduceAngles,
    SECRCalibrationBuilder,
)


@ddt
class TestPulseScaling(unittest.TestCase):
    """Test pulse scaling."""

    @data(FakeSherbrooke().target)
    def test_rzx_to_secr_forward(self, target: Target):
        """Test pulse scaling RZX with forward SECR."""
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

        scaled_qc = scale_cr_pulses(qc, target, unroll_rzx_to_ecr=True)
        scaled_qc.assign_parameters(param_bind, inplace=True)
        qc.assign_parameters(param_bind, inplace=True)

        self.assertTrue(Operator(qc).equiv(Operator(scaled_qc)))

    @data(FakeSherbrooke().target)
    def test_rzx_to_secr_reverse(self, target: Target):
        """Test pulse scaling RZX with reverse SECR."""
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
            qc, target, unroll_rzx_to_ecr=True, param_bind=param_bind
        )
        qc.assign_parameters(param_bind, inplace=True)

        self.assertTrue(Operator(qc).equiv(Operator(scaled_qc)))

    @data(FakeSherbrooke().target)
    def test_rzx_to_secr(self, target: Target):
        """Test pulse scaling with RZX gates."""
        rng = np.random.default_rng()
        theta = rng.uniform(-np.pi, np.pi)

        qc = QuantumCircuit(2)
        qc.rzx(theta, 0, 1)

        scaled_qc = scale_cr_pulses(qc, target, unroll_rzx_to_ecr=True, param_bind={})
        self.assertTrue(Operator(qc).equiv(Operator(scaled_qc)))

        qc = QuantumCircuit(2)
        qc.rzx(theta, 1, 0)

        scaled_qc = scale_cr_pulses(qc, target, unroll_rzx_to_ecr=True, param_bind={})
        self.assertTrue(Operator(qc).equiv(Operator(scaled_qc)))

    @data(FakeSherbrooke().target)
    def test_forced_rzz_template_match(self, target: Target):
        """Test forced template optimization for CX-RZ(1)-CX matches"""
        theta = Parameter("$\\theta$")
        rng = np.random.default_rng(12345)

        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.rz(theta, 1)
        qc.cx(0, 1)
        qc.rz(np.pi / 4, 1)

        scale_qc_no_match = scale_cr_pulses(
            qc,
            target,
            unroll_rzx_to_ecr=False,
            force_zz_matches=False,
            param_bind=None,
        )
        dag_no_match = circuit_to_dag(scale_qc_no_match)
        self.assertFalse(dag_no_match.collect_runs(["rzx"]))

        scale_qc_match = scale_cr_pulses(
            qc, target, unroll_rzx_to_ecr=False, force_zz_matches=True, param_bind=None
        )
        dag_match = circuit_to_dag(scale_qc_match)
        self.assertTrue(dag_match.collect_runs(["rzx"]))

        theta_set = rng.uniform(-np.pi, np.pi)
        self.assertTrue(
            Operator(qc.assign_parameters({theta: theta_set})).equiv(
                Operator(scale_qc_match.assign_parameters({theta: theta_set}))
            )
        )

    def test_angle_reduction(self):
        """Test Angle Reduction"""
        pm = PassManager(ReduceAngles(["rzx"]))

        qc1 = QuantumCircuit(2)
        qc1.rzx(9 * np.pi / 2, 0, 1)
        qc1_s = pm.run(qc1)

        qc2 = QuantumCircuit(2)
        qc2.rzx(42, 0, 1)
        qc2_s = pm.run(qc2)

        qc3 = QuantumCircuit(2)
        qc3.rzx(-np.pi, 0, 1)
        qc3_s = pm.run(qc3)

        self.assertAlmostEqual(qc1_s.data[0].operation.params[0], np.pi / 2)
        self.assertAlmostEqual(qc2_s.data[0].operation.params[0], -1.9822971502571)
        self.assertAlmostEqual(qc3_s.data[0].operation.params[0], -np.pi)

    @data(FakeSherbrooke().target)
    def test_secr_calibration_builder(self, target: Target):
        """
        Test SECR Calibration Builder

        Note the circuit must first pass through the RZXtoEchoedCR pass to correct
        for the direction of the native CR operation.
        """
        inst_sched_map = target.instruction_schedule_map()
        coupling_map = target.build_coupling_map()
        ctrl_chans = [ControlChannel(idx) for idx in range(len(list(coupling_map)))]

        theta = -np.pi / 7
        qc = QuantumCircuit(2)
        qc.rzx(2 * theta, 1, 0)

        # Verify that there are no calibrations for this circuit yet.
        self.assertEqual(qc.calibrations, {})

        pm = PassManager(
            [
                RZXtoEchoedCR(target),
                SECRCalibrationBuilder(inst_sched_map),
                Optimize1qGatesDecomposition(BASIS_GATES),
            ]
        )
        qc_cal = pm.run(qc)
        sched = schedule(
            qc_cal, inst_map=inst_sched_map, meas_map=[list(range(target.num_qubits))]
        )

        crp_start_time = sched.filter(
            channels=ctrl_chans, instruction_types=[Play]
        ).instructions[0][0]
        crm_start_time = sched.filter(
            channels=ctrl_chans, instruction_types=[Play]
        ).instructions[1][0]

        crp_duration = (
            sched.filter(
                channels=ctrl_chans,
                instruction_types=[Play],
            )
            .instructions[0][1]
            .duration
        )
        crm_duration = (
            sched.filter(
                channels=ctrl_chans,
                instruction_types=[Play],
            )
            .instructions[1][1]
            .duration
        )

        # same duration for all 1Q native gates
        echo_duration = inst_sched_map.get("x", qubits=[0]).duration

        self.assertEqual(crp_start_time + crp_duration + echo_duration, crm_start_time)
        self.assertEqual(crp_duration, crm_duration)
