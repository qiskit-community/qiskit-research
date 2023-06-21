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

from typing import Iterable, List, Optional, Sequence, Union

from qiskit import QuantumCircuit, pulse
from qiskit.circuit import Gate
from qiskit.circuit.library import XGate, YGate
from qiskit.converters import circuit_to_dag
from qiskit.providers.backend import Backend
from qiskit.pulse import Drag, Waveform
from qiskit.qasm import pi
from qiskit.transpiler import InstructionDurations
from qiskit.transpiler.basepasses import BasePass
from qiskit.transpiler.instruction_durations import InstructionDurationsType
from qiskit.transpiler.passes import PadDynamicalDecoupling
from qiskit.transpiler.passes.scheduling import ALAPScheduleAnalysis
from qiskit.transpiler.passes.scheduling.scheduling.base_scheduler import BaseScheduler

import numpy as np

from qiskit_research.utils.gates import XmGate, XpGate, YmGate, YpGate, PiPhiGate
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


def dynamical_decoupling_passes(
    backend: Backend,
    dd_str: str,
    scheduler: BaseScheduler = ALAPScheduleAnalysis,
    urdd_pulse_num: int = 4,
) -> Iterable[BasePass]:
    """
    Yield transpilation passes for dynamical decoupling.

    Args:
        backend (Backend): Backend to run on; gate timing is required for this method.
        dd_str (str): String describing DD sequence to use.
        scheduler (BaseScheduler, optional): Scheduler, defaults to ALAPScheduleAnalysis.
        urdd_pulse_num (int, optional): URDD pulse number must be even and at least 4.
            Defaults to 4.

    Yields:
        Iterator[Iterable[BasePass]]: Transpiler passes used for adding DD sequences.
    """
    durations = get_instruction_durations(backend)
    pulse_alignment = backend.configuration().timing_constraints["pulse_alignment"]

    if dd_str in DD_SEQUENCE:
        sequence = DD_SEQUENCE[dd_str]
    elif dd_str == "URDD":
        phis = get_urdd_angles(urdd_pulse_num)
        sequence = tuple(PiPhiGate(phi) for phi in phis)
    yield scheduler(durations)
    yield PadDynamicalDecoupling(
        durations, list(sequence), pulse_alignment=pulse_alignment
    )


def periodic_dynamical_decoupling(
    backend: Backend,
    base_dd_sequence: Optional[List[Gate]] = None,
    base_spacing: Optional[List[float]] = None,
    avg_min_delay: int = None,
    max_repeats: int = 1,
    scheduler: BaseScheduler = ALAPScheduleAnalysis,
) -> Iterable[BasePass]:
    """Yields transpilation passes for periodic dynamical decoupling."""
    durations = get_instruction_durations(backend)
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


# TODO this should take instruction schedule map instead of backend
def get_instruction_durations(backend: Backend) -> InstructionDurations:
    """
    Retrieves gate timing information for the backend from the instruction
    schedule map, and returns the type InstructionDurations for use by
    Qiskit's scheduler (i.e., ALAP) and DynamicalDecoupling passes.

    This method relies on IBM backend knowledge such as

      - all single qubit gates durations are the same
      - the 'x' gate, used for echoed cross resonance, is also the basis for
        all othe dynamical decoupling gates (currently)
    """
    inst_durs: InstructionDurationsType = []
    inst_sched_map = backend.defaults().instruction_schedule_map
    num_qubits = backend.configuration().num_qubits

    # single qubit gates
    for qubit in range(num_qubits):
        for inst_str in inst_sched_map.qubit_instructions(qubits=[qubit]):
            inst = inst_sched_map.get(inst_str, qubits=[qubit])
            inst_durs.append((inst_str, qubit, inst.duration))

            # create DD pulses from CR echo 'x' pulse
            if inst_str == "x":
                for new_gate in ["xp", "xm", "y", "yp", "ym", "\\pi_{\\phi}"]:
                    inst_durs.append((new_gate, qubit, inst.duration))

    # two qubit gates
    for qc in range(num_qubits):
        for qt in range(num_qubits):
            for inst_str in inst_sched_map.qubit_instructions(qubits=[qc, qt]):
                inst = inst_sched_map.get(inst_str, qubits=[qc, qt])
                inst_durs.append((inst_str, [qc, qt], inst.duration))

    return InstructionDurations(inst_durs)


# TODO refactor this as a CalibrationBuilder transpilation pass
def add_pulse_calibrations(
    circuits: Union[QuantumCircuit, List[QuantumCircuit]],
    backend: Backend,
    urdd_pulse_method: str = "phase_shift",
) -> None:
    """Add pulse calibrations for custom gates to circuits in-place."""
    inst_sched_map = backend.defaults().instruction_schedule_map
    num_qubits = backend.configuration().num_qubits

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
        for run in dag.collect_runs(["\\pi_{\\phi}"]):
            for node in run:
                qubit = node.qargs[0].index
                phi = node.op.params[0]
                x_sched = inst_sched_map.get("x", qubits=[qubit])
                _, x_instruction = x_sched.instructions[0]

                with pulse.build(f"PiPhi gate for qubit {qubit}") as sched:
                    if urdd_pulse_method == "phase_shift":
                        with pulse.phase_offset(phi, x_instruction.channel):
                            pulse.play(x_instruction.pulse, x_instruction.channel)
                    elif urdd_pulse_method == "amp_flip":
                        amp_flip_pulse = Drag(
                            duration=x_instruction.pulse.duration,
                            amp=(-1) ** (phi // (pi / 2)) * x_instruction.pulse.amp,
                            sigma=x_instruction.pulse.sigma,
                            beta=(-1) ** (phi // (pi / 2)) * x_instruction.pulse.beta,
                        )
                        phi %= pi / 2
                        with pulse.phase_offset(phi, x_instruction.channel):
                            pulse.play(amp_flip_pulse, x_instruction.channel)
                    elif urdd_pulse_method == "complex_sum":
                        wf_array = x_instruction.pulse.get_waveform().samples
                        iq_waveform = Waveform(
                            wf_array * (np.cos(phi) + 1j * np.sin(phi))
                        )
                        pulse.play(iq_waveform, x_instruction.channel)
                    else:
                        raise ValueError(
                            f"{urdd_pulse_method} not a valid URDD pulse calibration type."
                        )

                circuit.add_calibration(r"\\pi_{\\phi}", [qubit], sched, params=[phi])


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
