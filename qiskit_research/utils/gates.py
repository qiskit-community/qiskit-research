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

from typing import Optional, Union

import numpy
from qiskit import QuantumCircuit, QuantumRegister, pulse
from qiskit.circuit.gate import Gate
from qiskit.circuit.library import U3Gate
from qiskit.providers.backend import Backend
from qiskit.pulse import DriveChannel
from qiskit.qasm import pi


class XpGate(Gate):
    r"""The single-qubit Pauli-X gate (:math:`\sigma_x`), implemented
    via RX(\pi).
    """

    def __init__(self, label: Optional[str] = None):
        """Create new Xp gate."""
        super().__init__("xp", 1, [], label=label)

    def _define(self):
        """
        gate xp a { u3(pi,0,pi) a; }
        """
        q = QuantumRegister(1, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [(U3Gate(pi, 0, pi), [q[0]], [])]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def inverse(self):
        r"""Return inverted Xp gate (Xm)."""
        return XmGate()  # self-inverse

    def __array__(self, dtype=None):
        """Return a numpy.array for the Xp gate."""
        return numpy.array([[0, 1], [1, 0]], dtype=dtype)


class XmGate(Gate):
    r"""The single-qubit Pauli-X gate (:math:`\sigma_x`), implemented
    via RX(-\pi).
    """

    def __init__(self, label: Optional[str] = None):
        """Create new Xm gate."""
        super().__init__("xm", 1, [], label=label)

    def _define(self):
        """
        gate xm a { u3(pi,0,pi) a; }
        """
        q = QuantumRegister(1, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [(U3Gate(pi, 0, pi), [q[0]], [])]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def inverse(self):
        r"""Return inverted Xm gate (Xp)."""
        return XpGate()  # self-inverse

    def __array__(self, dtype=None):
        """Return a numpy.array for the X gate."""
        return numpy.array([[0, 1], [1, 0]], dtype=dtype)


class YpGate(Gate):
    r"""The single-qubit Pauli-Y gate (:math:`\sigma_y`), implemented
    via RY(\pi).
    """

    def __init__(self, label: Optional[str] = None):
        """Create new Yp gate."""
        super().__init__("yp", 1, [], label=label)

    def _define(self):
        q = QuantumRegister(1, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [(U3Gate(pi, pi / 2, pi / 2), [q[0]], [])]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def inverse(self):
        r"""Return inverted Yp gate (:math:`Y{\dagger} = Y`)"""
        return YmGate()  # self-inverse

    def __array__(self, dtype=None):
        """Return a numpy.array for the Yp gate."""
        return numpy.array([[0, -1j], [1j, 0]], dtype=dtype)


class YmGate(Gate):
    r"""The single-qubit Pauli-Y gate (:math:`\sigma_y`), implemented
    via RY(-\pi).
    """

    def __init__(self, label: Optional[str] = None):
        """Create new Ym gate."""
        super().__init__("ym", 1, [], label=label)

    def _define(self):
        q = QuantumRegister(1, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [(U3Gate(pi, pi / 2, pi / 2), [q[0]], [])]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def inverse(self):
        r"""Return inverted Ym gate (:math:`Y{\dagger} = Y`)"""
        return YpGate()  # self-inverse

    def __array__(self, dtype=None):
        """Return a numpy.array for the Ym gate."""
        return numpy.array([[0, -1j], [1j, 0]], dtype=dtype)


def add_pulse_calibrations(
    circuits: Union[QuantumCircuit, list[QuantumCircuit]],
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
            for circ in circuits:
                circ.add_calibration("xp", [qubit], sched)

        with pulse.build(f"xm gate for qubit {qubit}") as sched:
            # def of XmGate() in terms of XGate() and amplitude inversion
            x_sched = inst_sched_map.get("x", qubits=[qubit])
            x_pulse = x_sched.instructions[0][1].pulse
            # HACK is there a better way?
            x_pulse._amp = -x_pulse.amp
            pulse.play(x_pulse, DriveChannel(qubit))

            # add calibrations to circuits
            for circ in circuits:
                circ.add_calibration("xm", [qubit], sched)

        with pulse.build(f"y gate for qubit {qubit}") as sched:
            # def of YGate() in terms of XGate() and phase_offset
            with pulse.phase_offset(pi / 2, DriveChannel(qubit)):
                x_sched = inst_sched_map.get("x", qubits=[qubit])
                pulse.call(x_sched)

            # add calibrations to circuits
            for circ in circuits:
                circ.add_calibration("y", [qubit], sched)

        with pulse.build(f"yp gate for qubit {qubit}") as sched:
            # def of YpGate() in terms of XGate() and phase_offset
            with pulse.phase_offset(pi / 2, DriveChannel(qubit)):
                x_sched = inst_sched_map.get("x", qubits=[qubit])
                pulse.call(x_sched)

            # add calibrations to circuits
            for circ in circuits:
                circ.add_calibration("yp", [qubit], sched)

        with pulse.build(f"ym gate for qubit {qubit}") as sched:
            # def of YGate() in terms of XGate() and phase_offset
            with pulse.phase_offset(-pi / 2, DriveChannel(qubit)):
                x_sched = inst_sched_map.get("x", qubits=[qubit])
                x_pulse = x_sched.instructions[0][1].pulse
                # HACK is there a better way?
                x_pulse._amp = -x_pulse.amp
                pulse.play(x_pulse, DriveChannel(qubit))

            # add calibrations to circuits
            for circ in circuits:
                circ.add_calibration("ym", [qubit], sched)
