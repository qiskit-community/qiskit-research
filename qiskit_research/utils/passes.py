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


class XXPlusYYtoRZX(TransformationPass):
    """Transformation pass to decompose XXPlusYYGate to RZXGate."""

    def __init__(
        self,
        instruction_schedule_map: InstructionScheduleMap = None,
    ):
        super().__init__()
        self._inst_map = instruction_schedule_map

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

    def __init__(
        self,
        instruction_schedule_map: InstructionScheduleMap = None,
    ):
        super().__init__()
        self._inst_map = instruction_schedule_map

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


class CombineRuns(TransformationPass):
    """
    Class to combine consecutive gates of same type. This works with
    Parameters whereas other transpiling passes do  not.

    TODO: Check to see if this can be fixed in Optimize1qGatesDecomposition
    """

    def __init__(
        self,
        gate_strs: List[str],
    ):
        super().__init__()
        self._gate_strs = gate_strs

    def run(
        self,
        dag: DAGCircuit,
    ) -> DAGCircuit:

        for gate_str in self._gate_strs:
            for grun in dag.collect_runs([gate_str]):
                partition = []
                chunk = []
                for ii in range(len(grun) - 1):
                    chunk.append(grun[ii])

                    qargs0 = grun[ii].qargs
                    qargs1 = grun[ii + 1].qargs

                    if qargs0 != qargs1:
                        partition.append(chunk)
                        chunk = []

                chunk.append(grun[-1])
                partition.append(chunk)

                # simplify each chunk in the partition
                for chunk in partition:
                    theta = 0
                    for ii in range(len(chunk)):
                        theta += chunk[ii].op.params[0]

                    # set the first chunk to sum of params
                    chunk[0].op.params[0] = theta

                    # remove remaining chunks if any
                    if len(chunk) > 1:
                        for nn in chunk[1:]:
                            dag.remove_op_node(nn)
            return dag


class BindParameters(TransformationPass):
    """
    Bind Parameters to circuit.

    TODO: Must this convert the DAG back to a QuantumCircuit?
    """

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
        # TODO: there should be a better way to do this
        circuit = dag_to_circuit(dag)
        circuit.assign_parameters(self._param_bind, inplace=True)
        return circuit_to_dag(circuit)


class SECRCalibrationBuilder(CalibrationBuilder):
    """
    Creates calibrations for SECRGate(theta) by stretching and compressing
    Gaussian square pulses in the CX gate. This is done by retrieving (for a given pair of
    qubits) the CX schedule in the instruction schedule map of the backend defaults.
    The CX schedule must be an echoed cross-resonance gate optionally with rotary tones.
    The cross-resonance drive tones and rotary pulses must be Gaussian square pulses.
    The width of the Gaussian square pulse is adjusted so as to match the desired rotation angle.
    If the rotation angle is small such that the width disappears then the amplitude of the
    zero width Gaussian square pulse (i.e. a Gaussian) is reduced to reach the target rotation
    angle. Additional details can be found in https://arxiv.org/abs/2012.11660.

    Note this is modified from RZXCalibrationBuilder in qiskit.transpiler.passes.calibrations
    """

    def __init__(
        self,
        instruction_schedule_map: InstructionScheduleMap = None,
        qubit_channel_mapping: List[List[str]] = None,
    ):
        """
        Initializes a SECRGate calibration builder.

        Args:
            instruction_schedule_map: The :obj:`InstructionScheduleMap` object representing the
                default pulse calibrations for the target backend
            qubit_channel_mapping: The list mapping qubit indices to the list of
                channel names that apply on that qubit.

        Raises:
            QiskitError: if open pulse is not supported by the backend.
        """
        super().__init__()
        if instruction_schedule_map is None or qubit_channel_mapping is None:
            raise QiskitError(
                "Calibrations can only be added to Pulse-enabled backends"
            )

        self._inst_map = instruction_schedule_map
        self._channel_map = qubit_channel_mapping

    def supported(self, node_op: CircuitInst, qubits: List) -> bool:
        """Determine if a given node supports the calibration.

        Args:
            node_op: Target instruction object.
            qubits: Integer qubit indices to check.

        Returns:
            Return ``True`` is calibration can be provided.
        """
        return isinstance(node_op, SECRGate)

    @staticmethod
    def rescale_cr_inst(instruction: Play, theta: float, sample_mult: int = 16) -> Play:
        """
        Args:
            instruction: The instruction from which to create a new shortened or lengthened pulse.
            theta: desired angle, pi/2 is assumed to be the angle that the pulse in the given
                play instruction implements.
            sample_mult: All pulses must be a multiple of sample_mult.

        Returns:
            qiskit.pulse.Play: The play instruction with the stretched compressed
                GaussianSquare pulse.

        Raises:
            QiskitError: if the pulses are not GaussianSquare.
        """
        pulse_ = instruction.pulse
        if isinstance(pulse_, GaussianSquare):
            amp = pulse_.amp
            width = pulse_.width
            sigma = pulse_.sigma
            n_sigmas = (pulse_.duration - width) / sigma

            # The error function is used because the Gaussian may have chopped tails.
            gaussian_area = abs(amp) * sigma * np.sqrt(2 * np.pi) * math.erf(n_sigmas)
            area = gaussian_area + abs(amp) * width

            target_area = abs(float(theta)) / (np.pi / 2.0) * area
            sign = theta / abs(float(theta))

            if target_area > gaussian_area:
                width = (target_area - gaussian_area) / abs(amp)
                duration = (
                    math.ceil((width + n_sigmas * sigma) / sample_mult) * sample_mult
                )
                return Play(
                    GaussianSquare(
                        amp=sign * amp, width=width, sigma=sigma, duration=duration
                    ),
                    channel=instruction.channel,
                )
            else:
                amp_scale = sign * target_area / gaussian_area
                duration = math.ceil(n_sigmas * sigma / sample_mult) * sample_mult
                return Play(
                    GaussianSquare(
                        amp=amp * amp_scale, width=0, sigma=sigma, duration=duration
                    ),
                    channel=instruction.channel,
                )
        else:
            raise ValueError(
                "SECRCalibrationBuilder only stretches/compresses GaussianSquare."
            )

    def get_calibration(
        self, node_op: CircuitInst, qubits: List
    ) -> Union[Schedule, ScheduleBlock]:
        """Builds the calibration schedule for the SECRGate(theta).

        Args:
            node_op: Instruction of the SECRGate(theta). I.e. params[0] is theta.
            qubits: List of qubits for which to get the schedules. The first qubit is
                the control and the second is the target.

        Returns:
            schedule: The calibration schedule for the SECRGate(theta).

        Raises:
            QiskitError: If the control and target
                qubits cannot be identified, or the backend does not support cx between
                the qubits.
            TranspilerError: If all Parameters are not bound.
        """
        try:
            theta = float(node_op.params[0])
        except TypeError as ex:
            raise TranspilerError(
                "This transpilation pass requires all Parameters to be bound and real."
            ) from ex

        q1, q2 = qubits[0], qubits[1]

        if not self._inst_map.has("cx", qubits):
            raise QiskitError(
                "This transpilation pass requires the backend to support cx "
                "between qubits %i and %i." % (q1, q2)
            )

        cx_sched = self._inst_map.get("cx", qubits=(q1, q2))
        secr_theta = Schedule(name="secr(%.3f)" % theta)
        secr_theta.metadata["publisher"] = CalibrationPublisher.QISKIT

        if theta == 0.0:
            return secr_theta

        crs, comp_tones = [], []
        control, target = None, None

        for time, inst in cx_sched.instructions:

            # Identify the CR pulses.
            if isinstance(inst, Play) and not isinstance(inst, ShiftPhase):
                if isinstance(inst.channel, ControlChannel):
                    crs.append((time, inst))

            # Identify the compensation tones.
            if isinstance(inst.channel, DriveChannel) and not isinstance(
                inst, ShiftPhase
            ):
                if isinstance(inst.pulse, GaussianSquare):
                    comp_tones.append((time, inst))
                    target = inst.channel.index
                    control = q1 if target == q2 else q2

        if control is None:
            raise QiskitError("Control qubit is None.")
        if target is None:
            raise QiskitError("Target qubit is None.")

        echo_x = self._inst_map.get("x", qubits=control)

        # Build the schedule

        # Stretch/compress the CR gates and compensation tones
        cr1 = self.rescale_cr_inst(crs[0][1], theta)
        cr2 = self.rescale_cr_inst(crs[1][1], theta)

        if len(comp_tones) == 0:
            comp1, comp2 = None, None
        elif len(comp_tones) == 2:
            comp1 = self.rescale_cr_inst(comp_tones[0][1], theta)
            comp2 = self.rescale_cr_inst(comp_tones[1][1], theta)
        else:
            raise QiskitError(
                "CX must have either 0 or 2 rotary tones between qubits %i and %i "
                "but %i were found." % (control, target, len(comp_tones))
            )

        # Build the schedule for the SECRGate
        secr_theta = secr_theta.insert(0, cr1)

        if comp1 is not None:
            secr_theta = secr_theta.insert(0, comp1)

        secr_theta = secr_theta.insert(comp1.duration, echo_x)
        time = comp1.duration + echo_x.duration
        secr_theta = secr_theta.insert(time, cr2)

        if comp2 is not None:
            secr_theta = secr_theta.insert(time, comp2)

        return secr_theta
