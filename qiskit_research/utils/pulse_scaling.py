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

"""Pulse scaling."""

from typing import Iterable, List, Optional, Union

from qiskit import pulse
from qiskit.circuit import Instruction as CircuitInst
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import CXGate, RZGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit, DAGNode, DAGOpNode
from qiskit.exceptions import QiskitError
from qiskit.providers.backend import Backend
from qiskit.pulse import (
    Schedule,
    ScheduleBlock,
)
from qiskit.qasm import pi
from qiskit.transpiler.basepasses import BasePass, TransformationPass
from qiskit.transpiler.passes import (
    CXCancellation,
    Optimize1qGatesDecomposition,
    RZXCalibrationBuilder,
    RZXCalibrationBuilderNoEcho,
    TemplateOptimization,
)
from qiskit.transpiler.passes.calibration.rzx_templates import rzx_templates
from qiskit_research.utils.gate_decompositions import (
    RZXtoEchoedCR,
)
from qiskit_research.utils.gates import SECRGate

BASIS_GATES = ["sx", "rz", "rzx", "cx"]


def cr_scaling_passes(
    backend: Backend,
    templates: List[QuantumCircuit],
    unroll_rzx_to_ecr: bool = True,
    force_zz_matches: Optional[bool] = True,
    param_bind: Optional[dict] = None,
) -> Iterable[BasePass]:
    """Yields transpilation passes for CR pulse scaling."""

    yield TemplateOptimization(**templates)
    yield CombineRuns(["rzx"])
    if force_zz_matches:
        yield ForceZZTemplateSubstitution()  # workaround for Terra Issue
    if unroll_rzx_to_ecr:
        yield RZXtoEchoedCR(backend)
    yield Optimize1qGatesDecomposition(BASIS_GATES)
    yield CXCancellation()
    yield CombineRuns(["rz"])
    if param_bind is not None:
        yield from pulse_attaching_passes(backend, param_bind)


def pulse_attaching_passes(
    backend: Backend,
    param_bind: dict,
) -> Iterable[BasePass]:
    """Yields transpilation passes for attaching pulse schedules."""
    inst_sched_map = backend.defaults().instruction_schedule_map
    channel_map = backend.configuration().qubit_channel_mapping

    yield BindParameters(param_bind)
    yield Optimize1qGatesDecomposition(BASIS_GATES)
    yield CXCancellation()
    yield SECRCalibrationBuilder(inst_sched_map, channel_map)
    yield RZXCalibrationBuilder(inst_sched_map, channel_map)


class CombineRuns(TransformationPass):
    # TODO: Check to see if this can be fixed in Optimize1qGatesDecomposition
    """Combine consecutive gates of same type.

    This works with Parameters whereas other transpiling passes do not.
    """

    def __init__(self, gate_names: List[str]):
        """
        Args:

            gate_names: list of strings corresponding to the types
                of singe-parameter gates to combine.
        """
        super().__init__()
        self._gate_names = gate_names

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        for gate_name in self._gate_names:
            for run in dag.collect_runs([gate_name]):
                partition = []
                chunk = []
                for i in range(len(run) - 1):
                    chunk.append(run[i])

                    qargs0 = run[i].qargs
                    qargs1 = run[i + 1].qargs

                    if qargs0 != qargs1:
                        partition.append(chunk)
                        chunk = []

                chunk.append(run[-1])
                partition.append(chunk)

                # simplify each chunk in the partition
                for chunk in partition:
                    theta = 0
                    for node in chunk:
                        theta += node.op.params[0]

                    # set the first chunk to sum of params
                    chunk[0].op.params[0] = theta

                    # remove remaining chunks if any
                    if len(chunk) > 1:
                        for node in chunk[1:]:
                            dag.remove_op_node(node)
        return dag


class ReduceAngles(TransformationPass):
    """Reduce angle of scaled pulses to between -pi and pi.

    This works only after Parameters are bound. Gate strings
    should only be single-parameter scaled pulses, i.e.
    'rzx' and 'secr'.
    """

    def __init__(self, gate_names: List[str]):
        """
        Args:

            gate_names: list of strings corresponding to the types
                of singe-parameter gates to reduce.
        """
        super().__init__()
        self._gate_names = gate_names

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        for gate_name in self._gate_names:
            for run in dag.collect_runs([gate_name]):
                for node in run:
                    theta = node.op.params[0]
                    node.op.params[0] = (float(theta) + pi) % (2 * pi) - pi

        return dag


class BindParameters(TransformationPass):
    """Bind Parameters to circuit."""

    def __init__(
        self,
        param_bind: dict,
    ):
        super().__init__()
        self._param_bind = param_bind

    def run(
        self,
        dag: DAGCircuit,
    ) -> DAGCircuit:
        # TODO: Must this convert the DAG back to a QuantumCircuit?
        circuit = dag_to_circuit(dag)
        circuit.assign_parameters(self._param_bind, inplace=True)
        return circuit_to_dag(circuit)


class ForceZZTemplateSubstitution(TransformationPass):
    """
    Force sequences of the form CX-RZ(1)-CX to match to ZZ(theta) template. This
    is a workaround for known Qiskit Terra Issue TODO
    """

    def __init__(
        self,
        template: Optional[QuantumCircuit] = None,
    ):
        super().__init__()
        if template is None:
            self._template = rzx_templates(["zz3"])["template_list"][0].copy()

    def get_zz_temp_sub(self) -> QuantumCircuit:
        """
        Returns the inverse of the ZZ part of the template.
        """
        rzx_dag = circuit_to_dag(self._template)
        temp_cx1_node = rzx_dag.front_layer()[0]
        for gp in rzx_dag.bfs_successors(temp_cx1_node):
            if gp[0] == temp_cx1_node:
                if isinstance(gp[1][0].op, CXGate) and isinstance(gp[1][1].op, RZGate):
                    temp_rz_node = gp[1][1]
                    temp_cx2_node = gp[1][0]

                    rzx_dag.remove_op_node(temp_cx1_node)
                    rzx_dag.remove_op_node(temp_rz_node)
                    rzx_dag.remove_op_node(temp_cx2_node)

        return dag_to_circuit(rzx_dag).inverse()

    def sub_zz_in_dag(
        self, dag: DAGCircuit, cx1_node: DAGNode, rz_node: DAGNode, cx2_node: DAGNode
    ) -> DAGCircuit:
        """
        Replaces ZZ part of the dag with it inverse from an rzx template.
        """
        zz_temp_sub = self.get_zz_temp_sub().assign_parameters(
            {self.get_zz_temp_sub().parameters[0]: rz_node.op.params[0]}
        )
        dag.remove_op_node(rz_node)
        dag.remove_op_node(cx2_node)

        qr = QuantumRegister(2, "q")
        mini_dag = DAGCircuit()
        mini_dag.add_qreg(qr)
        for _, (instr, qargs, _) in enumerate(zz_temp_sub.data):
            mini_dag.apply_operation_back(instr, qargs=qargs)

        dag.substitute_node_with_dag(
            node=cx1_node, input_dag=mini_dag, wires=[qr[0], qr[1]]
        )
        return dag

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """
        Finds patterns of CX-RZ(1)-CX and replaces them with inverse from template.
        """
        cx_runs = dag.collect_runs("cx")
        for run in cx_runs:
            cx1_node = run[0]
            gp = next(dag.bfs_successors(cx1_node))
            if isinstance(gp[0].op, CXGate):  # dunno why this is needed
                if isinstance(gp[1][0], DAGOpNode) and isinstance(gp[1][1], DAGOpNode):
                    if isinstance(gp[1][0].op, CXGate) and isinstance(
                        gp[1][1].op, RZGate
                    ):
                        rz_node = gp[1][1]
                        cx2_node = gp[1][0]
                        gp1 = next(dag.bfs_successors(rz_node))
                        if cx2_node in gp1[1]:
                            if (
                                (cx1_node.qargs[0].index == cx2_node.qargs[0].index)
                                and (cx1_node.qargs[1].index == cx2_node.qargs[1].index)
                                and (cx2_node.qargs[1].index == rz_node.qargs[0].index)
                            ):
                                dag = self.sub_zz_in_dag(
                                    dag, cx1_node, rz_node, cx2_node
                                )

        return dag


class SECRCalibrationBuilder(RZXCalibrationBuilderNoEcho):
    """
    Creates calibrations for SECRGate(theta) by stretching and compressing
    Gaussian square pulses in the CX gate. This is subclassed from RZXCalibrationBuilderNoEcho,
    and builds the schedule from the scaled single (non-echoed) CR pulses.
    """

    def supported(self, node_op: CircuitInst, qubits: List) -> bool:
        """Determine if a given node supports the calibration.

        Args:
            node_op: Target instruction object.
            qubits: Integer qubit indices to check.

        Returns:
            Return ``True`` is calibration can be provided.
        """
        return isinstance(node_op, SECRGate) and self._inst_map.has("cx", qubits)

    def get_calibration(
        self, node_op: CircuitInst, qubits: List
    ) -> Union[Schedule, ScheduleBlock]:
        """
        Builds scaled echoed cross resonance (SECR) by doing echoing two single
        (unechoed) CR pulses of opposite amplitude.
        """
        theta = node_op.params[0]
        try:
            theta = float(theta)
        except TypeError as ex:
            raise QiskitError("Target rotation angle is not assigned.") from ex

        op_plus = CircuitInst(
            name="rzx", num_qubits=2, num_clbits=0, params=[theta / 2.0]
        )
        op_minus = CircuitInst(
            name="rzx", num_qubits=2, num_clbits=0, params=[-theta / 2.0]
        )

        cr_plus = super().get_calibration(op_plus, qubits)
        echo_x_sched = self._inst_map.get("x", qubits=qubits[0])
        cr_minus = super().get_calibration(op_minus, qubits)

        with pulse.build(name=f"secr{theta}") as secr_sched:
            with pulse.align_sequential():
                pulse.call(cr_plus)
                pulse.call(echo_x_sched)
                pulse.call(cr_minus)

        return secr_sched
