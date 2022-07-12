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

"""Gate decompositions."""

from __future__ import annotations

from collections.abc import Iterator

from qiskit import QuantumRegister
from qiskit.circuit import Gate, Qubit
from qiskit.circuit.library import (
    HGate,
    RXGate,
    RZGate,
    RZXGate,
    XGate,
    XXMinusYYGate,
    XXPlusYYGate,
)
from qiskit.dagcircuit import DAGCircuit
from qiskit.providers.backend import Backend
from qiskit.pulse import ControlChannel, Play
from qiskit.qasm import pi
from qiskit.transpiler.basepasses import TransformationPass

from .gates import SECRGate


def cr_forward_direction(control, target, inst_sched_map, ctrl_chans) -> bool:
    """
    Determines if the direction of cross resonance is forward (True), applied on control qubit qc or
    reverse (False), applied to target qubit qt.
    """
    cx_sched = inst_sched_map.get("cx", qubits=[control, target])
    cx_ctrl_chan = (
        cx_sched.filter(
            channels=[
                ControlChannel(idx)
                for idx in range(len(inst_sched_map.qubits_with_instruction("cx")))
            ],
            instruction_types=Play,
        )
        .instructions[0][1]
        .channel
    )
    forward_ctrl_chan = ctrl_chans[(control, target)][0]
    reverse_ctrl_chan = ctrl_chans[(target, control)][0]

    if cx_ctrl_chan == forward_ctrl_chan:
        return True
    if cx_ctrl_chan == reverse_ctrl_chan:
        return False

    raise ValueError(f"Qubits {control} and {target} are not a cross resonance pair.")


class RZXtoEchoedCR(TransformationPass):
    """
    Class for the RZXGate to echoed cross resonance gate pass. The RZXGate
    is equivalent to the SECR gate plus a second XGate on the control qubit
    to return it to the initial state.

    See: https://arxiv.org/abs/1603.04821
    """

    def __init__(
        self,
        backend: Backend,
    ):
        super().__init__()
        self._inst_map = backend.defaults().instruction_schedule_map
        self._ctrl_chans = backend.configuration().control_channels

    def run(
        self,
        dag: DAGCircuit,
    ) -> DAGCircuit:

        for rzx_run in dag.collect_runs(["rzx"]):
            control = rzx_run[0].qargs[0].index
            target = rzx_run[0].qargs[1].index
            cr_forward_dir = cr_forward_direction(
                control, target, self._inst_map, self._ctrl_chans
            )

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

        yield RZGate(-beta), (b,)

        yield HGate(), (a,)
        yield HGate(), (b,)

        yield RZGate(-0.5 * pi), (b,)
        yield RXGate(-0.5 * pi), (b,)
        yield RZGate(-0.5 * pi), (b,)
        yield RZXGate(0.5 * theta), (a, b)
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
        yield RZXGate(0.5 * theta), (a, b)
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
