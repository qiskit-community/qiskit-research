"""Two-qubit YX + XY interaction gate."""

from typing import Optional

import numpy as np
from qiskit.circuit.gate import Gate
from qiskit.circuit.library import CXGate, HGate, RYGate, SdgGate, SGate
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister


class YXPlusXYInteractionGate(Gate):
    r"""A parametric 2-qubit :math:`Y \otimes X + X \otimes Y` interaction.

    **Circuit Symbol:**

    .. parsed-literal::

             ┌────────────┐
        q_0: ┤0           ├
             │  Ryx+xy(θ) │
        q_1: ┤1           ├
             └────────────┘

    **Matrix Representation:**

    .. math::

        R_{YX+XY}(\theta)\ q_0, q_1 = \exp(-i \theta \frac{X{\otimes}Y + Y{\otimes}X}{2}) =
            \begin{pmatrix}
                \cos(\theta) & 0 & 0 & -\sin(\theta) \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 1 & 0 \\
                \sin(\theta) & 0 & 0 & \cos(\theta)
            \end{pmatrix}

    **Examples:**

        .. math::

            R_{YX+XY}(\theta = 0) = I

        .. math::

            R_{YX+XY}(\theta = \frac{\pi}{4}) =
                \begin{pmatrix}
                    \frac{1}{\sqrt{2}} & 0 & 0 & -\frac{1}{\sqrt{2}} \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & 1 & 0 \\
                    \frac{1}{\sqrt{2}} & 0 & 0  & \frac{1}{\sqrt{2}}
                \end{pmatrix}

        .. math::

            R_{YX+XY}(\theta = \frac{\pi}{2}) =
                \begin{pmatrix}
                    0 & 0 & 0 & -1 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & 1 & 0 \\
                    1 & 0 & 0 & 0
                \end{pmatrix}
    """

    def __init__(self, theta: ParameterValueType, label: Optional[str] = None):
        """Create new YXPlusXYInteractionGate gate."""
        super().__init__("ryxxy", 2, [theta], label=label)

    def __array__(self, dtype=None):
        """Return a numpy array for the YXPlusXYInteractionGate gate."""
        (theta,) = self.params
        cos = np.cos(theta)
        sin = np.sin(theta)
        return np.array(
            [[cos, 0, 0, -sin], [0, 1, 0, 0], [0, 0, 1, 0], [sin, 0, 0, cos]],
            dtype=dtype,
        )

    def _define(self):
        """Decomposition of the gate."""
        (theta,) = self.params
        register = QuantumRegister(2, "q")
        circuit = QuantumCircuit(register, name=self.name)
        a, b = register
        rules = [
            (SGate(), [a], []),
            (SGate(), [b], []),
            (HGate(), [b], []),
            (CXGate(), [b, a], []),
            (RYGate(-theta), [a], []),
            (RYGate(theta), [b], []),
            (CXGate(), [b, a], []),
            (HGate(), [b], []),
            (SdgGate(), [b], []),
            (SdgGate(), [a], []),
        ]
        for instr, qargs, cargs in rules:
            circuit.append(instr, qargs, cargs)

        self.definition = circuit

    def inverse(self):
        """Return inverse YXPlusXYInteractionGate gate."""
        (theta,) = self.params
        return YXPlusXYInteractionGate(-theta)
