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

from typing import Iterable, Union

from qiskit import QuantumCircuit, pulse
from qiskit.circuit.library import XGate, YGate
from qiskit.providers.backend import Backend
from qiskit.pulse import DriveChannel
from qiskit.qasm import pi
from qiskit.transpiler import InstructionDurations
from qiskit.transpiler.basepasses import BasePass
from qiskit.transpiler.instruction_durations import InstructionDurationsType
from qiskit.transpiler.passes import ALAPSchedule, DynamicalDecoupling
from qiskit_research.utils.gates import XmGate, XpGate, YmGate, YpGate

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
    backend, dd_str: str, scheduler: BasePass = ALAPSchedule
) -> Iterable[BasePass]:
    """Yields transpilation passes for dynamical decoupling."""
    durations = get_instruction_durations(backend)
    sequence = DD_SEQUENCE[dd_str]
    yield scheduler(durations)
    yield DynamicalDecoupling(durations, list(sequence))


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
    circuits: Union[QuantumCircuit, List[QuantumCircuit], List[List[QuantumCircuit]]],
    backend: Backend,
) -> None:
    """Add pulse calibrations for custom gates to circuits in-place."""
    inst_sched_map = backend.defaults().instruction_schedule_map
    num_qubits = backend.configuration().num_qubits

    if isinstance(circuits, QuantumCircuit):
        circuits = [circuits]

    for qubit in range(num_qubits):
        with pulse.build(f"xp gate for qubit {qubit}") as sched:
            # def of XpGate() in terms of XGate()
            x_sched = inst_sched_map.get("x", qubits=[qubit])
            pulse.call(x_sched)

            # add calibrations to circuits
            for circs in circuits:
                if isinstance(circs, QuantumCircuit):
                    circs.add_calibration("xp", [qubit], sched)
                else:
                    [circ.add_calibration("xp", [qubit], sched) for circ in circs]

        with pulse.build(f"xm gate for qubit {qubit}") as sched:
            # def of XmGate() in terms of XGate() and amplitude inversion
            x_sched = inst_sched_map.get("x", qubits=[qubit])
            x_pulse = x_sched.instructions[0][1].pulse
            # HACK is there a better way?
            x_pulse._amp = -x_pulse.amp  # pylint: disable=protected-access
            pulse.play(x_pulse, DriveChannel(qubit))

            # add calibrations to circuits
            for circs in circuits:
                if isinstance(circs, QuantumCircuit):
                    circs.add_calibration("xm", [qubit], sched)
                else:
                    [circ.add_calibration("xm", [qubit], sched) for circ in circs]

        with pulse.build(f"y gate for qubit {qubit}") as sched:
            # def of YGate() in terms of XGate() and phase_offset
            with pulse.phase_offset(pi / 2, DriveChannel(qubit)):
                x_sched = inst_sched_map.get("x", qubits=[qubit])
                pulse.call(x_sched)

            # add calibrations to circuits
            for circs in circuits:
                if isinstance(circs, QuantumCircuit):
                    circs.add_calibration("y", [qubit], sched)
                else:
                    [circ.add_calibration("y", [qubit], sched) for circ in circs]

        with pulse.build(f"yp gate for qubit {qubit}") as sched:
            # def of YpGate() in terms of XGate() and phase_offset
            with pulse.phase_offset(pi / 2, DriveChannel(qubit)):
                x_sched = inst_sched_map.get("x", qubits=[qubit])
                pulse.call(x_sched)

            # add calibrations to circuits
            for circs in circuits:
                if isinstance(circs, QuantumCircuit):
                    circs.add_calibration("yp", [qubit], sched)
                else:
                    [circ.add_calibration("yp", [qubit], sched) for circ in circs]

        with pulse.build(f"ym gate for qubit {qubit}") as sched:
            # def of YGate() in terms of XGate() and phase_offset
            with pulse.phase_offset(-pi / 2, DriveChannel(qubit)):
                x_sched = inst_sched_map.get("x", qubits=[qubit])
                x_pulse = x_sched.instructions[0][1].pulse
                # HACK is there a better way?
                x_pulse._amp = -x_pulse.amp  # pylint:disable=protected-access
                pulse.play(x_pulse, DriveChannel(qubit))

            # add calibrations to circuits
            for circs in circuits:
                if isinstance(circs, QuantumCircuit):
                    circs.add_calibration("ym", [qubit], sched)
                else:
                    [circ.add_calibration("ym", [qubit], sched) for circ in circs]
