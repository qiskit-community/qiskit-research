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

"""Dynamical decoupling."""

from __future__ import annotations

from typing import Iterable, List, Optional, Union

from qiskit import QuantumCircuit, pulse
from qiskit.circuit import Gate
from qiskit.circuit.library import XGate, YGate
from qiskit.providers.backend import Backend
from qiskit.pulse import Drag
from qiskit.qasm import pi
from qiskit.transpiler import InstructionDurations
from qiskit.transpiler.basepasses import BasePass
from qiskit.transpiler.instruction_durations import InstructionDurationsType
from qiskit.transpiler.passes import PadDynamicalDecoupling
from qiskit.transpiler.passes.scheduling import ALAPScheduleAnalysis
from qiskit.transpiler.passes.scheduling.scheduling.base_scheduler import BaseScheduler
from qiskit_research.utils.gates import XmGate, XpGate, YmGate, YpGate
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
    backend: Backend, dd_str: str, scheduler: BaseScheduler = ALAPScheduleAnalysis
) -> Iterable[BasePass]:
    """Yields transpilation passes for dynamical decoupling."""
    durations = get_instruction_durations(backend)
    pulse_alignment = backend.configuration().timing_constraints["pulse_alignment"]

    sequence = DD_SEQUENCE[dd_str]
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
                for new_gate in ["xp", "xm", "y", "yp", "ym"]:
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
