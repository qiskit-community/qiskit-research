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

from typing import Iterable, List, Union
from qiskit.qasm import pi

from qiskit import pulse
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.gate import Gate
from qiskit.circuit.library import XGate, YGate
from qiskit.providers.backend import Backend
from qiskit.pulse import DriveChannel
from qiskit.transpiler import InstructionDurations, PassManager
from qiskit.transpiler.basepasses import BasePass
from qiskit.transpiler.passes import ALAPSchedule, DynamicalDecoupling
from qiskit_research.utils.gates import XpGate, XmGate, YpGate, YmGate


DD_SEQUENCE = {
    "X2": [XGate(), XGate()],
    "X2pm": [XpGate(), XmGate()],
    "XY4": [XGate(), YGate(), XGate(), YGate()],
    "XY4pm": [XpGate(), YpGate(), XmGate(), YmGate()],
    "XY8": [XGate(), YGate(), XGate(), YGate(), YGate(), XGate(), YGate(), XGate()],
    "XY8pm": [
        XpGate(),
        YpGate(),
        XmGate(),
        YmGate(),
        YmGate(),
        XmGate(),
        YpGate(),
        XpGate(),
    ],
}


def add_dynamical_decoupling(
    circuits: Union[QuantumCircuit, List[QuantumCircuit]],
    backend: Backend,
    dd_str: str,
    scheduler: BasePass = ALAPSchedule,
) -> Union[QuantumCircuit, List[QuantumCircuit]]:
    """Add dynamical decoupling sequences and calibrations to circuits.

    Adds dynamical decoupling sequences and the calibrations necessary
    to run them on an IBM backend.
    """
    circuits_dd = add_dd_sequence(circuits, backend, dd_str, scheduler)
    add_dd_pulse_calibrations(circuits_dd, backend)
    return circuits_dd


def get_dd_sequence(dd_str: str) -> List[Gate]:
    """Return dynamical decoupling sequence based on input string."""
    if dd_str not in DD_SEQUENCE:
        raise ValueError(
            f'The string "{dd_str}" does not describe a valid '
            "dynamical decoupling sequence. Please use one of the following: "
            f"{list(DD_SEQUENCE.keys())}."
        )
    return DD_SEQUENCE[dd_str]


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

    inst_durs = []
    inst_sched_map = backend.defaults().instruction_schedule_map
    num_qubits = backend.configuration().num_qubits

    # single qubit gates
    for qubit in range(num_qubits):
        for inst_str in inst_sched_map.qubit_instructions(qubits=[qubit]):
            inst = inst_sched_map.get(inst_str, qubits=[qubit])
            inst_durs.append((inst_str, qubit, inst.duration))

            # create DD pulses from CR echo 'x' pulse
            if inst_str == "x":
                inst_durs.append(("xp", qubit, inst.duration))
                inst_durs.append(("xm", qubit, inst.duration))
                inst_durs.append(("y", qubit, inst.duration))
                inst_durs.append(("yp", qubit, inst.duration))
                inst_durs.append(("ym", qubit, inst.duration))

    # two qubit gates
    for qc in range(num_qubits):
        for qt in range(num_qubits):
            for inst_str in inst_sched_map.qubit_instructions(qubits=[qc, qt]):
                inst = inst_sched_map.get(inst_str, qubits=[qc, qt])
                inst_durs.append((inst_str, [qc, qt], inst.duration))

    return InstructionDurations(inst_durs)


def add_dd_sequence(
    circuits: Union[QuantumCircuit, List[QuantumCircuit]],
    backend: Backend,
    dd_str: str,
    scheduler: BasePass = ALAPSchedule,
) -> Union[QuantumCircuit, List[QuantumCircuit]]:
    """Add dynamical decoupling sequences to a circuit or circuits."""
    pass_manager = PassManager(
        list(dynamical_decoupling_passes(backend, dd_str, scheduler))
    )
    return pass_manager.run(circuits)


def dynamical_decoupling_passes(
    backend, dd_str: str, scheduler: BasePass = ALAPSchedule
) -> Iterable[BasePass]:
    """Yields transpilation passes for dynamical decoupling."""
    durations = get_instruction_durations(backend)
    sequence = get_dd_sequence(dd_str)
    yield scheduler(durations)
    yield DynamicalDecoupling(durations, sequence)


def add_dd_pulse_calibrations(
    circuits: Union[QuantumCircuit, List[QuantumCircuit]],
    backend: Backend,
):
    """
    Build the pulse schedule and attach as pulse gate to all the
    gates used for dynamical decoupling.
    """
    inst_sched_map = backend.defaults().instruction_schedule_map
    num_qubits = backend.configuration().num_qubits

    if isinstance(circuits, QuantumCircuit):
        circuits = [circuits]

    for qubit in range(num_qubits):
        with pulse.build("xp gate for qubit " + str(qubit)) as sched:
            # def of XpGate() in terms of XGate()
            x_sched = inst_sched_map.get("x", qubits=[qubit])
            pulse.call(x_sched)

            # for each DD sequence with a XpGate() in it
            for circ in circuits:
                circ.add_calibration("xp", [qubit], sched)

        with pulse.build("xm gate for qubit " + str(qubit)) as sched:
            # def of XmGate() in terms of XGate() and amplitude inversion
            x_sched = inst_sched_map.get("x", qubits=[qubit])
            x_pulse = x_sched.instructions[0][1].pulse
            x_pulse._amp = -x_pulse.amp  # bad form
            pulse.play(x_pulse, DriveChannel(qubit))

            # for each DD sequence with a XmGate() in it
            for circ in circuits:
                circ.add_calibration("xm", [qubit], sched)

        with pulse.build("y gate for qubit " + str(qubit)) as sched:
            # def of YGate() in terms of XGate() and phase_offset
            with pulse.phase_offset(pi / 2, DriveChannel(qubit)):
                x_sched = inst_sched_map.get("x", qubits=[qubit])
                pulse.call(x_sched)

            # for each DD sequence with a YGate() in it
            for circ in circuits:
                circ.add_calibration("y", [qubit], sched)

        with pulse.build("yp gate for qubit " + str(qubit)) as sched:
            # def of YpGate() in terms of XGate() and phase_offset
            with pulse.phase_offset(pi / 2, DriveChannel(qubit)):
                x_sched = inst_sched_map.get("x", qubits=[qubit])
                pulse.call(x_sched)

            # for each DD sequence with a YpGate() in it
            for circ in circuits:
                circ.add_calibration("yp", [qubit], sched)

        with pulse.build("ym gate for qubit " + str(qubit)) as sched:
            # def of YGate() in terms of XGate() and phase_offset
            with pulse.phase_offset(-pi / 2, DriveChannel(qubit)):
                x_sched = inst_sched_map.get("x", qubits=[qubit])
                x_pulse = x_sched.instructions[0][1].pulse
                x_pulse._amp = -x_pulse.amp  # bad form
                pulse.play(x_pulse, DriveChannel(qubit))

            # for each DD sequence with a YmGate() in it
            for circ in circuits:
                circ.add_calibration("ym", [qubit], sched)
