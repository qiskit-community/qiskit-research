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

import math
from collections.abc import Iterator
from typing import List, Union

import numpy as np
from qiskit import QuantumRegister
from qiskit.circuit import Gate
from qiskit.circuit import Instruction as CircuitInst
from qiskit.circuit import Qubit
from qiskit.circuit.library import (
    HGate,
    RXGate,
    RZGate,
    RZXGate,
    XGate,
    XXMinusYYGate,
    XXPlusYYGate,
)
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.exceptions import QiskitError
from qiskit.pulse import (
    ControlChannel,
    DriveChannel,
    GaussianSquare,
    Play,
    Schedule,
    ScheduleBlock,
    ShiftPhase,
)
from qiskit.pulse.instruction_schedule_map import (
    CalibrationPublisher,
    InstructionScheduleMap,
)
from qiskit.qasm import pi
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes.calibration.builders import CalibrationBuilder

from .gates import SECRGate


class RZXtoEchoedCR(TransformationPass):
    """
    Class for the RZXGate to echoed cross resonance gate pass. The RZXGate
    is equivalent to the SECR gate plus a second XGate on the control qubit
    to return it to the initial state.

    See: https://arxiv.org/abs/1603.04821
    """

    def __init__(
        self,
        instruction_schedule_map: InstructionScheduleMap = None,
    ):
        super().__init__()
        self._inst_map = instruction_schedule_map

    def run(
        self,
        dag: DAGCircuit,
    ) -> DAGCircuit:

        for rzx_run in dag.collect_runs(["rzx"]):
            qc = rzx_run[0].qargs[0].index
            qt = rzx_run[0].qargs[1].index
            cx_f_sched = self._inst_map.get("cx", qubits=[qc, qt])
            cx_r_sched = self._inst_map.get("cx", qubits=[qt, qc])
            cr_forward_dir = cx_f_sched.duration < cx_r_sched.duration

            for node in rzx_run:
                mini_dag = DAGCircuit()
                register = QuantumRegister(2)
                mini_dag.add_qreg(register)

                rzx_angle = node.op.params[0]

                if cr_forward_dir:
                    mini_dag.apply_operation_back(
                        SECRGate(rzx_angle), [register[0], register[1]]
                    )
                    mini_dag.apply_operation_back(XGate(), [register[0]])
                else:
                    mini_dag.apply_operation_back(HGate(), [register[0]])
                    mini_dag.apply_operation_back(HGate(), [register[1]])
                    mini_dag.apply_operation_back(
                        SECRGate(rzx_angle), [register[1], register[0]]
                    )
                    mini_dag.apply_operation_back(XGate(), [register[1]])
                    mini_dag.apply_operation_back(HGate(), [register[0]])
                    mini_dag.apply_operation_back(HGate(), [register[1]])

                dag.substitute_node_with_dag(node, mini_dag)

        return dag


class XXPlusYYtoRZX(TransformationPass):
    """Transformation pass to decompose XXPlusYYGate to RZXGate."""

    def _decomposition(
        self,
        register: QuantumRegister,
        gate: XXPlusYYGate,
    ) -> Iterator[tuple[Gate, tuple[Qubit, ...]]]:
        a, b = register
        theta, beta = gate.params

        yield RZGate(beta), (b,)

        yield HGate(), (a,)
        yield HGate(), (b,)

        yield RZGate(-0.5 * pi), (b,)
        yield RXGate(-0.5 * pi), (b,)
        yield RZGate(-0.5 * pi), (b,)
        yield RZXGate(-0.5 * theta), (a, b)
        yield RXGate(0.5 * theta), (b,)
        yield RZGate(-0.5 * pi), (b,)
        yield RXGate(-0.5 * pi), (b,)
        yield RZGate(-0.5 * pi), (b,)
        yield RZGate(-0.5 * theta), (b,)

        yield RZGate(0.5 * pi), (a,)
        yield HGate(), (a,)
        yield RZGate(0.5 * pi), (b,)
        yield HGate(), (b,)

        yield RZGate(-0.5 * pi), (b,)
        yield RXGate(-0.5 * pi), (b,)
        yield RZGate(-0.5 * pi), (b,)
        yield RZXGate(-0.5 * theta), (a, b)
        yield RXGate(0.5 * theta), (b,)
        yield RZGate(-0.5 * pi), (b,)
        yield RXGate(-0.5 * pi), (b,)
        yield RZGate(-0.5 * pi), (b,)
        yield RZGate(-0.5 * theta), (b,)

        yield HGate(), (a,)
        yield RZGate(-0.5 * pi), (a,)
        yield HGate(), (a,)
        yield HGate(), (b,)
        yield RZGate(-0.5 * pi), (b,)
        yield HGate(), (b,)

        yield RZGate(-beta), (b,)

    def run(
        self,
        dag: DAGCircuit,
    ) -> DAGCircuit:
        for run in dag.collect_runs(["xx_plus_yy"]):
            for node in run:
                mini_dag = DAGCircuit()
                register = QuantumRegister(2)
                mini_dag.add_qreg(register)

                for instr, qargs in self._decomposition(register, node.op):
                    mini_dag.apply_operation_back(instr, qargs)

                dag.substitute_node_with_dag(node, mini_dag)

        return dag


class XXMinusYYtoRZX(TransformationPass):
    """Transformation pass to decompose XXMinusYYGate to RZXGate."""

    def _decomposition(
        self,
        register: QuantumRegister,
        gate: XXMinusYYGate,
    ) -> Iterator[tuple[Gate, tuple[Qubit, ...]]]:
        a, b = register
        theta, beta = gate.params

        yield RZGate(-beta), (b,)

        yield HGate(), (a,)
        yield HGate(), (b,)

        yield RZGate(-0.5 * pi), (b,)
        yield RXGate(-0.5 * pi), (b,)
        yield RZGate(-0.5 * pi), (b,)
        yield RZXGate(0.5 * theta), (a, b)
        yield RXGate(-0.5 * theta), (b,)
        yield RZGate(-0.5 * pi), (b,)
        yield RXGate(-0.5 * pi), (b,)
        yield RZGate(-0.5 * pi), (b,)
        yield RZGate(0.5 * theta), (b,)

        yield RZGate(0.5 * pi), (a,)
        yield HGate(), (a,)
        yield RZGate(0.5 * pi), (b,)
        yield HGate(), (b,)

        yield RZGate(-0.5 * pi), (b,)
        yield RXGate(-0.5 * pi), (b,)
        yield RZGate(-0.5 * pi), (b,)
        yield RZXGate(-0.5 * theta), (a, b)
        yield RXGate(0.5 * theta), (b,)
        yield RZGate(-0.5 * pi), (b,)
        yield RXGate(-0.5 * pi), (b,)
        yield RZGate(-0.5 * pi), (b,)
        yield RZGate(-0.5 * theta), (b,)

        yield HGate(), (a,)
        yield RZGate(-0.5 * pi), (a,)
        yield HGate(), (a,)
        yield HGate(), (b,)
        yield RZGate(-0.5 * pi), (b,)
        yield HGate(), (b,)

        yield RZGate(beta), (b,)

    def run(
        self,
        dag: DAGCircuit,
    ) -> DAGCircuit:
        for run in dag.collect_runs(["xx_minus_yy"]):
            for node in run:
                mini_dag = DAGCircuit()
                register = QuantumRegister(2)
                mini_dag.add_qreg(register)

                for instr, qargs in self._decomposition(register, node.op):
                    mini_dag.apply_operation_back(instr, qargs)

                dag.substitute_node_with_dag(node, mini_dag)

        return dag
