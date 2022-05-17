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

"""Gates."""

from __future__ import annotations

from typing import Optional

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.gate import Gate
from qiskit.circuit.library import RZXGate, U3Gate, XGate
from qiskit.circuit.parameterexpression import ParameterValueType
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
            qc.append(instr, qargs, cargs)

        self.definition = qc

    def inverse(self):
        r"""Gate inverse."""
        return XmGate()

    def __array__(self, dtype=None):
        """Gate matrix."""
        return np.array([[0, 1], [1, 0]], dtype=dtype)


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
            qc.append(instr, qargs, cargs)

        self.definition = qc

    def inverse(self):
        r"""Gate inverse."""
        return XpGate()

    def __array__(self, dtype=None):
        """Gate matrix"""
        return np.array([[0, 1], [1, 0]], dtype=dtype)


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
            qc.append(instr, qargs, cargs)

        self.definition = qc

    def inverse(self):
        r"""Return inverted Yp gate (:math:`Y{\dagger} = Y`)"""
        return YmGate()  # self-inverse

    def __array__(self, dtype=None):
        """Gate matrix."""
        return np.array([[0, -1j], [1j, 0]], dtype=dtype)


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
            qc.append(instr, qargs, cargs)

        self.definition = qc

    def inverse(self):
        r"""Return inverted Ym gate (:math:`Y{\dagger} = Y`)"""
        return YpGate()  # self-inverse

    def __array__(self, dtype=None):
        """Gate matrix."""
        return np.array([[0, -1j], [1j, 0]], dtype=dtype)


class SECRGate(Gate):
    r"""The scaled echoed cross resonance gate, as detailed in
    https://arxiv.org/abs/1603.04821 and
    https://arxiv.org/abs/2202.12910.

    Definitions are derived by appending an XGate() to q0
    of an RZXGate.
    """

    def __init__(self, theta: ParameterValueType, label: Optional[str] = None):
        """Create new SECR gate."""
        super().__init__("secr", 2, [theta], label=label)

    def _define(self):
        """
        gate secr(theta) a, b { h b; cx a, b; u1(theta) b; cx a, b; h b; x a}
        """
        theta = self.params[0]
        q = QuantumRegister(2, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [(RZXGate(theta), [q[0], q[1]], []), (XGate(), [q[0]], [])]
        for instr, qargs, cargs in rules:
            qc.append(instr, qargs, cargs)

        self.definition = qc

    def inverse(self):
        r"""Gate inverse."""
        return SECRGate(-self.params[0])

    def __array__(self, dtype=None):
        """Gate matrix."""
        half_theta = float(self.params[0]) / 2
        cos = np.cos(half_theta)
        isin = 1j * np.sin(half_theta)
        return np.array(
            [
                [0, cos, 0, isin],
                [cos, 0, -isin, 0],
                [0, isin, 0, cos],
                [-isin, 0, cos, 0],
            ],
            dtype=dtype,
        )
