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

"""Two-qubit phased XX - YY interaction gate."""

from typing import Optional

import numpy as np
from qiskit.circuit.gate import Gate
from qiskit.circuit.library import (
    CXGate,
    RYGate,
    RZGate,
    SdgGate,
    SGate,
    SXdgGate,
    SXGate,
)
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.qasm import pi


class PhasedXXMinusYYGate(Gate):
    r"""Phased XX - YY interaction gate.

    A 2-qubit parameterized XX-YY interaction. Its action is to induce
    a coherent rotation by some angle between :math:`|00\rangle` and :math:`|11\rangle`.

    **Circuit Symbol:**

    .. parsed-literal::

             ┌──────────────┐
        q_0: ┤0             ├
             │  Rxx-yy(θ,β) │
        q_1: ┤1             ├
             └──────────────┘

    **Matrix Representation:**

    .. math::

        \newcommand{\th}{\frac{\theta}{2}}

        R_{XX-YY}((\theta, \beta)\ q_0, q_1 =
          RZ_1(\beta) \cdot exp(-i \frac{\theta}{2} \frac{XX-YY}{2}) \cdot RZ_1(-\beta) =
            \begin{pmatrix}
                \cos(\th)             & 0 & 0 & i\sin(\th)e^{i\beta}  \\
                0                     & 1 & 0 & 0  \\
                0                     & 0 & 1 & 0  \\
                i\sin(\th)e^{-i\beta} & 0 & 0 & \cos(\th)
            \end{pmatrix}

    .. note::

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In the above example we apply the gate
        on (q_0, q_1) which results in adding the (optional) phase defined
        by :math:`beta` on q_1. Instead, if we apply it on (q_1, q_0), the
        phase is added on q_0. If :math:`beta` is set to its default value
        of :math:`0`, the gate is equivalent in big and little endian.

        .. parsed-literal::

                 ┌──────────────┐
            q_0: ┤1             ├
                 │  Rxx-yy(θ,β) │
            q_1: ┤0             ├
                 └──────────────┘

        .. math::

            \newcommand{\th}{\frac{\theta}{2}}

            R_{XX-YY}(\theta, \beta)\ q_1, q_0 =
            RZ_0(\beta) \cdot exp(-i \frac{\theta}{2} \frac{XX-YY}{2}) \cdot RZ_0(-\beta) =
                \begin{pmatrix}
                    \cos(\th)             & 0 & 0 & i\sin(\th)e^{-i\beta}  \\
                    0                     & 1 & 0 & 0  \\
                    0                     & 0 & 1 & 0  \\
                    i\sin(\th)e^{i\beta} & 0 & 0 & \cos(\th)
                \end{pmatrix}

    """

    def __init__(
        self,
        theta: ParameterValueType,
        beta: ParameterValueType = 0,
        label: Optional[str] = None,
    ):
        """Create new PhasedXXMinusYY gate."""
        super().__init__("phased_xx_minus_yy", 2, [theta, beta], label=label)

    def _define(self):
        """Gate decomposition."""
        theta, beta = self.params
        q = QuantumRegister(2, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [
            (RZGate(-beta), [q[1]], []),
            (RZGate(-pi / 2), [q[0]], []),
            (SXGate(), [q[0]], []),
            (RZGate(pi / 2), [q[0]], []),
            (SGate(), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
            (RYGate(theta / 2), [q[0]], []),
            (RYGate(-theta / 2), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
            (SdgGate(), [q[1]], []),
            (RZGate(-pi / 2), [q[0]], []),
            (SXdgGate(), [q[0]], []),
            (RZGate(pi / 2), [q[0]], []),
            (RZGate(beta), [q[1]], []),
        ]
        for instr, qargs, cargs in rules:
            qc.append(instr, qargs, cargs)

        self.definition = qc

    def inverse(self):
        """Inverse gate."""
        theta, beta = self.params
        return PhasedXXMinusYYGate(-theta, beta)

    def __array__(self, dtype=None):
        """Gate matrix."""
        theta, beta = self.params
        cos = np.cos(theta / 2)
        sin = np.sin(theta / 2)
        return np.array(
            [
                [cos, 0, 0, -1j * sin * np.exp(-1j * beta)],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [-1j * sin * np.exp(1j * beta), 0, 0, cos],
            ],
            dtype=dtype,
        )
