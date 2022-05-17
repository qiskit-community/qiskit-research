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
from typing import Iterable, List, Optional, Union

import numpy as np
from qiskit.circuit import Instruction as CircuitInst
from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.exceptions import QiskitError
from qiskit.pulse import (
    ControlChannel,
    DriveChannel,
    GaussianSquare,
    InstructionScheduleMap,
    Play,
    Schedule,
    ScheduleBlock,
    ShiftPhase,
)
from qiskit.pulse.instruction_schedule_map import (
    CalibrationPublisher,
    InstructionScheduleMap,
)
from qiskit.transpiler.basepasses import BasePass, TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes import (
    CXCancellation,
    Optimize1qGatesDecomposition,
    TemplateOptimization,
)
from qiskit.transpiler.passes.calibration.builders import CalibrationBuilder
from qiskit_research.utils.gate_decompositions import RZXtoEchoedCR
from qiskit_research.utils.gates import SECRGate

BASIS_GATES = ["sx", "rz", "rzx", "cx"]


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
                for i in range(len(grun) - 1):
                    chunk.append(grun[i])

                    qargs0 = grun[i].qargs
                    qargs1 = grun[i + 1].qargs

                    if qargs0 != qargs1:
                        partition.append(chunk)
                        chunk = []

                chunk.append(grun[-1])
                partition.append(chunk)

                # simplify each chunk in the partition
                for chunk in partition:
                    theta = 0
                    for i in range(len(chunk)):
                        theta += chunk[i].op.params[0]

                    # set the first chunk to sum of params
                    chunk[0].op.params[0] = theta

                    # remove remaining chunks if any
                    if len(chunk) > 1:
                        for node in chunk[1:]:
                            dag.remove_op_node(node)
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
                duration = round((width + n_sigmas * sigma) / sample_mult) * sample_mult
                return Play(
                    GaussianSquare(
                        amp=sign * amp, width=width, sigma=sigma, duration=duration
                    ),
                    channel=instruction.channel,
                )
            else:
                amp_scale = sign * target_area / gaussian_area
                duration = round(n_sigmas * sigma / sample_mult) * sample_mult
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
                f"between qubits {q1} and {q2}."
            )

        cx_sched = self._inst_map.get("cx", qubits=(q1, q2))
        secr_theta = Schedule(name=f"secr({theta:.3f})")
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


def cr_scaling_passes(
    inst_sched_map: InstructionScheduleMap,
    channel_map: List[List[str]],
    templates: List[QuantumCircuit],
    param_bind: Optional[dict] = None,
) -> Iterable[BasePass]:
    """Yields transpilation passes for CR pulse scaling."""

    yield TemplateOptimization(**templates)
    yield CombineRuns(["rzx"])
    # pauli twirl here
    yield RZXtoEchoedCR(inst_sched_map)
    yield Optimize1qGatesDecomposition(BASIS_GATES)
    yield CXCancellation()
    yield CombineRuns(["rz"])
    if param_bind is not None:
        yield BindParameters(param_bind)
        yield Optimize1qGatesDecomposition(BASIS_GATES)
        yield CXCancellation()
        yield SECRCalibrationBuilder(inst_sched_map, channel_map)
