# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Dynamical decoupling."""

from __future__ import annotations

from enum import Enum
from math import pi
from typing import cast, Iterable, List, Optional, Sequence, Union

import numpy as np
from qiskit import QuantumCircuit, pulse
from qiskit.circuit import Gate, Parameter, Qubit
from qiskit.circuit.delay import Delay
from qiskit.circuit.library import U3Gate, UGate, XGate, YGate
from qiskit.circuit.reset import Reset
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit, DAGInNode, DAGNode, DAGOpNode
from qiskit.providers.backend import Backend
from qiskit.pulse import Drag, Waveform
from qiskit.synthesis import OneQubitEulerDecomposer
from qiskit.transpiler import InstructionDurations, Target
from qiskit.transpiler.basepasses import BasePass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes import Optimize1qGates, PadDynamicalDecoupling
from qiskit.transpiler.passes.scheduling import ALAPScheduleAnalysis
from qiskit.transpiler.passes.scheduling.scheduling.base_scheduler import BaseScheduler

from qiskit_research.utils.gates import PiPhiGate, XmGate, XpGate, YmGate, YpGate
from qiskit_research.utils.periodic_dynamical_decoupling import (
    PeriodicDynamicalDecoupling,
)

X = XGate()
Xp = XpGate()
Xm = XmGate()
Y = YGate()
Yp = YpGate()
Ym = YmGate()


DD_SEQUENCE = {
    "X2": (X, X),
    "X2pm": (Xp, Xm),
    "XY4": (X, Y, X, Y),
    "XY4pm": (Xp, Yp, Xm, Ym),
    "XY8": (X, Y, X, Y, Y, X, Y, X),
    "XY8pm": (Xp, Yp, Xm, Ym, Ym, Xm, Yp, Xp),
}


class PulseMethod(Enum):
    """Class for enumerating way of implementing custom gates."""

    GATEBASED = "decompose operation into standard gates"
    PHASESHIFT = "pulse implementation with phase-offset pulses"
    AMPLITUDEINVERT = "pulse implementation with positive/negative amplitudes"
    IQSUM = "complex sum of signals with no phase offset"


def dynamical_decoupling_passes(
    target: Target,
    dd_str: str,
    scheduler: BaseScheduler = ALAPScheduleAnalysis,
    urdd_pulse_num: int = 4,
) -> Iterable[BasePass]:
    """
    Yield transpilation passes for dynamical decoupling.

    Args:
        target (Target): Backend to run on; gate timing is required for this method.
        dd_str (str): String describing DD sequence to use.
        scheduler (BaseScheduler, optional): Scheduler, defaults to ALAPScheduleAnalysis.
        urdd_pulse_num (int, optional): URDD pulse number must be even and at least 4.
            Defaults to 4.

    Yields:
        Iterator[Iterable[BasePass]]: Transpiler passes used for adding DD sequences.
    """
    for new_gate in [cast(Gate, gate) for gate in [Xp, Xm, Yp, Ym]]:
        if new_gate.name not in target:
            target.add_instruction(new_gate, target["x"])

    if dd_str in DD_SEQUENCE:
        sequence = DD_SEQUENCE[dd_str]
    elif dd_str == "URDD":
        phis = get_urdd_angles(urdd_pulse_num)
        sequence = tuple(PiPhiGate(phi) for phi in phis)
        if "pi_phi" not in target:
            phi = Parameter("φ")
            target.add_instruction(PiPhiGate(phi), target["x"])
    else:
        raise AttributeError("No DD sequence specified")

    yield scheduler(target.durations())
    yield PadDynamicalDecoupling(target.durations(), list(sequence))


def periodic_dynamical_decoupling(
    backend: Backend,
    base_dd_sequence: Optional[List[Gate]] = None,
    base_spacing: Optional[List[float]] = None,
    avg_min_delay: int = None,
    max_repeats: int = 1,
    scheduler: BaseScheduler = ALAPScheduleAnalysis,
) -> Iterable[BasePass]:
    """Yields transpilation passes for periodic dynamical decoupling."""
    target = backend.target
    durations = target.durations()
    pulse_alignment = backend.configuration().timing_constraints["pulse_alignment"]

    if base_dd_sequence is None:
        base_dd_sequence = [XGate(), XGate()]

    yield scheduler(durations)
    yield PeriodicDynamicalDecoupling(
        durations,
        base_dd_sequence,
        base_spacing=base_spacing,
        avg_min_delay=avg_min_delay,
        max_repeats=max_repeats,
        pulse_alignment=pulse_alignment,
    )


# TODO refactor this as a CalibrationBuilder transpilation pass
def add_pulse_calibrations(
    circuits: Union[QuantumCircuit, List[QuantumCircuit]],
    target: Target,
    pulse_method: PulseMethod = PulseMethod.PHASESHIFT,
) -> None:
    """
    Add pulse calibrations for custom gates to circuits in-place.

    Args:
        circuits (Union[QuantumCircuit, List[QuantumCircuit]]): Circuits which need pulse schedules
            attached to the non-basis gates.
        backend (Backend): Backend from which pulse information is obtained.
        pulse_method (PulseMethod, optional): Exact method of implemeting pulse schedules given by
            PulseMethod enumeration. These should all be equivalent but in practice they may differ.
            Defaults to PulseMethod.PHASESHIFT.

    Raises:
        ValueError: Not a defined method for implementing pulse schedules for URDD gates.
    """
    inst_sched_map = target.instruction_schedule_map()
    num_qubits = target.num_qubits

    if isinstance(circuits, QuantumCircuit):
        circuits = [circuits]

    for qubit in range(num_qubits):
        # get XGate pulse to define the others
        x_sched = inst_sched_map.get("x", qubits=[qubit])
        _, x_instruction = x_sched.instructions[0]

        # XpGate has the same pulse
        with pulse.build(f"xp gate for qubit {qubit}") as sched:
            pulse.play(x_instruction.pulse, x_instruction.channel)
            for circ in circuits:
                circ.add_calibration("xp", [qubit], sched)

        # XmGate has amplitude inverted
        with pulse.build(f"xm gate for qubit {qubit}") as sched:
            inverted_pulse = Drag(
                duration=x_instruction.pulse.duration,
                amp=-x_instruction.pulse.amp,
                sigma=x_instruction.pulse.sigma,
                beta=x_instruction.pulse.beta,
            )
            pulse.play(inverted_pulse, x_instruction.channel)
            for circ in circuits:
                circ.add_calibration("xm", [qubit], sched)

        # YGate and YpGate have phase shifted
        with pulse.build(f"y gate for qubit {qubit}") as sched:
            with pulse.phase_offset(pi / 2, x_instruction.channel):
                pulse.play(x_instruction.pulse, x_instruction.channel)
            for circ in circuits:
                circ.add_calibration("y", [qubit], sched)
                circ.add_calibration("yp", [qubit], sched)

        # YmGate has phase shifted in opposite direction and amplitude inverted
        with pulse.build(f"ym gate for qubit {qubit}") as sched:
            with pulse.phase_offset(-pi / 2, x_instruction.channel):
                inverted_pulse = Drag(
                    duration=x_instruction.pulse.duration,
                    amp=-x_instruction.pulse.amp,
                    sigma=x_instruction.pulse.sigma,
                    beta=x_instruction.pulse.beta,
                )
                pulse.play(inverted_pulse, x_instruction.channel)
            for circ in circuits:
                circ.add_calibration("ym", [qubit], sched)

    for circuit in circuits:
        dag = circuit_to_dag(circuit)
        for run in dag.collect_runs(["pi_phi"]):
            for node in run:
                qubit, _ = dag.find_bit(node.qargs[0])
                phi = node.op.params[0]
                x_sched = inst_sched_map.get("x", qubits=[qubit])
                _, x_instruction = x_sched.instructions[0]

                with pulse.build(f"PiPhi gate for qubit {qubit}") as sched:
                    if pulse_method == PulseMethod.PHASESHIFT:
                        with pulse.phase_offset(phi, x_instruction.channel):
                            pulse.play(x_instruction.pulse, x_instruction.channel)
                    elif pulse_method == PulseMethod.AMPLITUDEINVERT:
                        amp_flip_pulse = Drag(
                            duration=x_instruction.pulse.duration,
                            amp=(-1) ** (phi // (pi / 2)) * x_instruction.pulse.amp,
                            sigma=x_instruction.pulse.sigma,
                            beta=(-1) ** (phi // (pi / 2)) * x_instruction.pulse.beta,
                        )
                        phi %= pi / 2
                        with pulse.phase_offset(phi, x_instruction.channel):
                            pulse.play(amp_flip_pulse, x_instruction.channel)
                    elif pulse_method == PulseMethod.IQSUM:
                        wf_array = x_instruction.pulse.get_waveform().samples
                        iq_waveform = Waveform(
                            wf_array * (np.cos(phi) + 1j * np.sin(phi))
                        )
                        pulse.play(iq_waveform, x_instruction.channel)
                    else:
                        raise ValueError(
                            f"{pulse_method} not a valid URDD pulse calibration type."
                        )

                circuit.add_calibration("pi_phi", [qubit], sched, params=[phi])


def get_urdd_angles(num_pulses: int = 4) -> Sequence[float]:
    """Gets \\phi_k values for n pulse UR sequence"""
    if num_pulses % 2 == 1:
        raise ValueError("num_pulses must be even")
    if num_pulses < 4:
        raise ValueError("num_pulses must be >= 4")

    # get capital Phi value
    if num_pulses % 4 == 0:
        m_divisor = int(num_pulses / 4)
        big_phi = np.pi / m_divisor
    else:
        m_divisor = int((num_pulses - 2) / 4)
        big_phi = (2 * m_divisor * np.pi) / (2 * m_divisor + 1)

    # keep track of unique phi added; we choose phi2 = big_phi by convention--
    # only real requirement is (n * big_phi = 2pi * j for j int)
    unique_phi = [0, big_phi]
    # map each phi in [phis] to location (by index) of corresponding [unique_phi]
    phi_indices = [0, 1]
    # populate remaining phi values
    for kk in range(3, num_pulses + 1):
        phi_k = (kk * (kk - 1) * big_phi) / 2
        # values only matter modulo 2 pi
        phi_k = (phi_k) % (2 * np.pi)
        if np.isclose(phi_k, 0):
            phi_k = 0
        elif np.isclose(phi_k, 2 * np.pi):
            phi_k = 0

        added_new = False
        for idx, u_phi in enumerate(unique_phi):
            if np.isclose(u_phi, phi_k, atol=0.001):
                added_new = True
                phi_indices.append(idx)

        if added_new is False:
            unique_phi.append(phi_k)
            phi_indices.append(len(unique_phi) - 1)

    # construct phi list
    phis: list[float] = []
    for idx in phi_indices:
        phis.append(unique_phi[idx])

    return phis


class URDDSequenceStrategy(PadDynamicalDecoupling):
    """URDD strategic timing dynamical decoupling insertion pass.

    This pass acts the same as PadDynamicalDecoupling, but only inserts
    a number of URDD pulses specified by a sequence of num_pulses only when there
    is a delay greater than a sequence of min_delay_time. Each delay will be
    considered and the large number of sequence will be given by that delay
    specificied max(min_delay_time) < delay.
    """

    def __init__(
        self,
        durations: InstructionDurations = None,
        num_pulses: List[int] = None,
        min_delay_times: List[int] = None,
        qubits: Optional[List[int]] = None,
        spacing: Optional[List[float]] = None,
        skip_reset_qubits: bool = True,
        pulse_alignment: int = 1,
        extra_slack_distribution: str = "middle",
    ):
        """URDD strategic timing initializer.

        Args:
            durations: Durations of instructions to be used in scheduling.
            num_pulses Sequence[int]: Number of pulses to use corresponding
                to min_delay_time. Defaults to 4.
            min_delay_time Sequence[int]: Minimum delay for a given sequence length. In units of dt.
            qubits: Physical qubits on which to apply DD.
                If None, all qubits will undergo DD (when possible).
            spacing: A list of spacings between the DD gates.
                The available slack will be divided according to this.
                The list length must be one more than the length of dd_sequence,
                and the elements must sum to 1. If None, a balanced spacing
                will be used [d/2, d, d, ..., d, d, d/2].
            skip_reset_qubits: If True, does not insert DD on idle periods that
                immediately follow initialized/reset qubits
                (as qubits in the ground state are less susceptile to decoherence).
            pulse_alignment: The hardware constraints for gate timing allocation.
                This is usually provided from ``backend.configuration().timing_constraints``.
                If provided, the delay length, i.e. ``spacing``, is implicitly adjusted to
                satisfy this constraint.
            extra_slack_distribution: The option to control the behavior of DD sequence generation.
                The duration of the DD sequence should be identical to an idle time in the
                scheduled quantum circuit, however, the delay in between gates comprising the
                sequence should be integer number in units of dt, and it might be further
                truncated when ``pulse_alignment`` is specified. This sometimes results in
                the duration of the created sequence being shorter than the idle time
                that you want to fill with the sequence, i.e. `extra slack`.
                This option takes following values.

                    - "middle": Put the extra slack to the interval at the middle of the sequence.
                    - "edges": Divide the extra slack as evenly as possible into
                      intervals at beginning and end of the sequence.

        Raises:
            TranspilerError: When invalid DD sequence is specified.
            TranspilerError: When pulse gate with the duration which is
                non-multiple of the alignment constraint value is found.
        """
        if num_pulses is None:
            num_pulses = [4]
        if min_delay_times is None:
            min_delay_times = [0]

        phis = get_urdd_angles(min(num_pulses))
        dd_sequence = tuple(PiPhiGate(phi) for phi in phis)

        super().__init__(
            durations=durations,
            dd_sequence=dd_sequence,
            qubits=qubits,
            spacing=spacing,
            skip_reset_qubits=skip_reset_qubits,
            pulse_alignment=pulse_alignment,
            extra_slack_distribution=extra_slack_distribution,
        )

        self._num_pulses = num_pulses
        self._min_delay_times = np.array(min_delay_times)

    def __is_dd_qubit(self, qubit_index: int) -> bool:
        """DD can be inserted in the qubit or not."""
        if self._qubits and qubit_index not in self._qubits:
            return False
        return True

    def _compute_spacing(self, num_pulses):
        mid = 1 / num_pulses
        end = mid / 2
        self._spacing = [end] + [mid] * (num_pulses - 1) + [end]

    def _compute_dd_sequence_lengths(self, dd_sequence, dag: DAGCircuit) -> dict:
        # Precompute qubit-wise DD sequence length for performance
        dd_sequence_lengths = {}
        for physical_index, qubit in enumerate(dag.qubits):
            if not self.__is_dd_qubit(physical_index):
                continue

            sequence_lengths = []
            for gate in dd_sequence:
                try:
                    # Check calibration.
                    gate_length = dag.calibrations[gate.name][
                        (physical_index, gate.params)
                    ]
                    if gate_length % self._alignment != 0:
                        # This is necessary to implement lightweight scheduling logic for this pass.
                        # Usually the pulse alignment constraint and pulse data chunk size take
                        # the same value, however, we can intentionally violate this pattern
                        # at the gate level. For example, we can create a schedule consisting of
                        # a pi-pulse of 32 dt followed by a post buffer, i.e. delay, of 4 dt
                        # on the device with 16 dt constraint. Note that the pi-pulse length
                        # is multiple of 16 dt but the gate length of 36 is not multiple of it.
                        # Such pulse gate should be excluded.
                        raise TranspilerError(
                            f"Pulse gate {gate.name} with length non-multiple of {self._alignment} "
                            f"is not acceptable in {self.__class__.__name__} pass."
                        )
                except KeyError:
                    gate_length = self._durations.get(gate, physical_index)
                sequence_lengths.append(gate_length)
                # Update gate duration. This is necessary for current timeline drawer,
                # i.e. scheduled.
                gate.duration = gate_length
            dd_sequence_lengths[qubit] = sequence_lengths

        return dd_sequence_lengths

    def _pad(
        self,
        dag: DAGCircuit,
        qubit: Qubit,
        t_start: int,
        t_end: int,
        next_node: DAGNode,
        prev_node: DAGNode,
    ):
        # This routine takes care of the pulse alignment constraint for the URDD sequence.
        # The only difference is that it will only execute for time_intervals larger than
        # those specified by the internal property self._min_delay_time which is defined
        # at initialization.
        time_interval = t_end - t_start
        dd_indices = np.where(self._min_delay_times < time_interval)[0]
        if len(dd_indices) > 0:
            dd_idx = np.where(
                self._min_delay_times[dd_indices]
                == max(self._min_delay_times[dd_indices])
            )[0][0]
            urdd_num = self._num_pulses[dd_idx]
            phis = get_urdd_angles(urdd_num)
            dd_sequence = tuple(PiPhiGate(phi) for phi in phis)
            dd_sequence_lengths = self._compute_dd_sequence_lengths(dd_sequence, dag)
            self._compute_spacing(urdd_num)

            if time_interval % self._alignment != 0:
                raise TranspilerError(
                    f"Time interval {time_interval} is not divisible by alignment "
                    f"{self._alignment} between DAGNode {prev_node.name} on qargs "
                    f"{prev_node.qargs} and {next_node.name} on qargs {next_node.qargs}."
                )

            if not self.__is_dd_qubit(dag.qubits.index(qubit)):
                # Target physical qubit is not the target of this DD sequence.
                self._apply_scheduled_op(
                    dag, t_start, Delay(time_interval, dag.unit), qubit
                )
                return

            if self._skip_reset_qubits and (
                isinstance(prev_node, DAGInNode) or isinstance(prev_node.op, Reset)
            ):
                # Previous node is the start edge or reset, i.e. qubit is ground state.
                self._apply_scheduled_op(
                    dag, t_start, Delay(time_interval, dag.unit), qubit
                )
                return

            slack = time_interval - np.sum(dd_sequence_lengths[qubit])
            sequence_gphase = self._sequence_phase

            if slack <= 0:
                # Interval too short.
                self._apply_scheduled_op(
                    dag, t_start, Delay(time_interval, dag.unit), qubit
                )
                return

            if len(dd_sequence) == 1:
                # Special case of using a single gate for DD
                u_inv = dd_sequence[0].inverse().to_matrix()
                theta, phi, lam, phase = OneQubitEulerDecomposer().angles_and_phase(
                    u_inv
                )
                if isinstance(next_node, DAGOpNode) and isinstance(
                    next_node.op, (UGate, U3Gate)
                ):
                    # Absorb the inverse into the successor (from left in circuit)
                    theta_r, phi_r, lam_r = next_node.op.params
                    next_node.op.params = Optimize1qGates.compose_u3(
                        theta_r, phi_r, lam_r, theta, phi, lam
                    )
                    sequence_gphase += phase
                elif isinstance(prev_node, DAGOpNode) and isinstance(
                    prev_node.op, (UGate, U3Gate)
                ):
                    # Absorb the inverse into the predecessor (from right in circuit)
                    theta_l, phi_l, lam_l = prev_node.op.params
                    prev_node.op.params = Optimize1qGates.compose_u3(
                        theta, phi, lam, theta_l, phi_l, lam_l
                    )
                    sequence_gphase += phase
                else:
                    # Don't do anything if there's no single-qubit gate to absorb the inverse
                    self._apply_scheduled_op(
                        dag, t_start, Delay(time_interval, dag.unit), qubit
                    )
                    return

            def _constrained_length(values):
                return self._alignment * np.floor(values / self._alignment)

            # (1) Compute DD intervals satisfying the constraint
            taus = _constrained_length(slack * np.asarray(self._spacing))
            extra_slack = slack - np.sum(taus)

            # (2) Distribute extra slack
            if self._extra_slack_distribution == "middle":
                mid_ind = int((len(taus) - 1) / 2)
                to_middle = _constrained_length(extra_slack)
                taus[mid_ind] += to_middle
                if extra_slack - to_middle:
                    # If to_middle is not a multiple value of the pulse alignment,
                    # it is truncated to the nearlest multiple value and
                    # the rest of slack is added to the end.
                    taus[-1] += extra_slack - to_middle
            elif self._extra_slack_distribution == "edges":
                to_begin_edge = _constrained_length(extra_slack / 2)
                taus[0] += to_begin_edge
                taus[-1] += extra_slack - to_begin_edge
            else:
                raise TranspilerError(
                    f"Option extra_slack_distribution = {self._extra_slack_distribution} "
                    f"is invalid."
                )

            # (3) Construct DD sequence with delays
            num_elements = max(len(dd_sequence), len(taus))
            idle_after = t_start
            for dd_ind in range(num_elements):
                if dd_ind < len(taus):
                    tau = taus[dd_ind]
                    if tau > 0:
                        self._apply_scheduled_op(
                            dag, idle_after, Delay(tau, dag.unit), qubit
                        )
                        idle_after += tau
                if dd_ind < len(dd_sequence):
                    gate = dd_sequence[dd_ind]
                    gate_length = dd_sequence_lengths[qubit][dd_ind]
                    self._apply_scheduled_op(dag, idle_after, gate, qubit)
                    idle_after += gate_length

            dag.global_phase = self._mod_2pi(dag.global_phase + sequence_gphase)

    @staticmethod
    def _mod_2pi(angle: float, atol: float = 0):
        """Wrap angle into interval [-π,π). If within atol of the endpoint, clamp to -π"""
        wrapped = (angle + np.pi) % (2 * np.pi) - np.pi
        if abs(wrapped - np.pi) < atol:
            wrapped = -np.pi
        return wrapped
