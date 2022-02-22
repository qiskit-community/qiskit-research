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

from typing import Optional

import numpy
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.gate import Gate
from qiskit.circuit.library import U3Gate
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
