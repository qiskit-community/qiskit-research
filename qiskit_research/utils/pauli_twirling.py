# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Pauli twirling."""

from typing import Any, Iterable, Optional

import numpy as np
from qiskit.circuit import QuantumRegister
from qiskit.circuit.library import (
    IGate,
    XGate,
    YGate,
    ZGate,
)
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import BasePass, TransformationPass
from qiskit.transpiler.passes import (
    CXCancellation,
    Optimize1qGatesDecomposition,
)
from qiskit.quantum_info import Pauli, pauli_basis
from qiskit_research.utils.pulse_scaling import BASIS_GATES

I = IGate()
X = XGate()
Y = YGate()
Z = ZGate()

# this list consists of the 2-qubit rotation gates
TWO_QUBIT_PAULI_GENERATORS = {
    "rxx": Pauli("XX"),
    "ryy": Pauli("YY"),
    "rzx": Pauli("XZ"),
    "rzz": Pauli("ZZ"),
    "secr": Pauli("XZ"),
}

# this dictionary stores the twirl sets for each supported gate
# each key is the name of a supported gate
# each value is a tuple that represents the twirl set for the gate
# the twirl set is a list of (before, after) pairs describing twirl gates
# "before" and "after" are tuples of single-qubit gates to be applied
# before and after the gate to be twirled
TWIRL_GATES = {
    "cx": (
        ((I, I), (I, I)),
        ((I, X), (I, X)),
        ((I, Y), (Z, Y)),
        ((I, Z), (Z, Z)),
        ((X, I), (X, X)),
        ((X, X), (X, I)),
        ((X, Y), (Y, Z)),
        ((X, Z), (Y, Y)),
        ((Y, I), (Y, X)),
        ((Y, X), (Y, I)),
        ((Y, Y), (X, Z)),
        ((Y, Z), (X, Y)),
        ((Z, I), (Z, I)),
        ((Z, X), (Z, X)),
        ((Z, Y), (I, Y)),
        ((Z, Z), (I, Z)),
    ),
}


def parse_random_seed(seed: Any) -> np.random.Generator:
    """Parse a random number generator seed and return a Generator."""
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


class PauliTwirl(TransformationPass):
    """Add Pauli twirls."""

    def __init__(
        self,
        gates_to_twirl: Optional[Iterable[str]] = None,
        seed: Any = None,
    ):
        """
        Args:
            gates_to_twirl: Names of gates to twirl. The default behavior is to twirl all
                supported gates.
            seed: Seed for the pseudorandom number generator.
        """
        if gates_to_twirl is None:
            gates_to_twirl = TWIRL_GATES.keys() | TWO_QUBIT_PAULI_GENERATORS.keys()
        self.gates_to_twirl = gates_to_twirl
        self.rng = parse_random_seed(seed)
        super().__init__()

    def run(
        self,
        dag: DAGCircuit,
    ) -> DAGCircuit:
        for run in dag.collect_runs(list(self.gates_to_twirl)):
            for node in run:
                if node.op.name in TWO_QUBIT_PAULI_GENERATORS:
                    mini_dag = DAGCircuit()
                    q0, q1 = node.qargs
                    # mini_dag.add_qreg(q0.register)
                    mini_dag.add_qreg(dag.find_bit(q0).registers[0][0])

                    theta = node.op.params[0]
                    this_pauli = Pauli(
                        self.rng.choice(pauli_basis(2).to_labels())
                    ).to_instruction()
                    if TWO_QUBIT_PAULI_GENERATORS[node.op.name].anticommutes(
                        this_pauli
                    ):
                        theta *= -1

                    new_op = node.op.copy()
                    new_op.params[0] = theta

                    mini_dag.apply_operation_back(this_pauli, [q0, q1])
                    mini_dag.apply_operation_back(new_op, [q0, q1])
                    if node.op.name == "secr":
                        mini_dag.apply_operation_back(X, [q0])
                    mini_dag.apply_operation_back(this_pauli, [q0, q1])
                    if node.op.name == "secr":
                        mini_dag.apply_operation_back(X, [q0])

                    dag.substitute_node_with_dag(node, mini_dag, wires=[q0, q1])

                elif node.op.name in TWIRL_GATES:
                    twirl_gates = TWIRL_GATES[node.op.name]
                    (before0, before1), (after0, after1) = twirl_gates[
                        self.rng.integers(len(twirl_gates))
                    ]
                    mini_dag = DAGCircuit()
                    register = QuantumRegister(2)
                    mini_dag.add_qreg(register)
                    mini_dag.apply_operation_back(before0, [register[0]])
                    mini_dag.apply_operation_back(before1, [register[1]])
                    mini_dag.apply_operation_back(node.op, [register[0], register[1]])
                    mini_dag.apply_operation_back(after0, [register[0]])
                    mini_dag.apply_operation_back(after1, [register[1]])
                    dag.substitute_node_with_dag(node, mini_dag)
                else:
                    raise TypeError(f"Unknown how to twirl Instruction {node.op}.")
        return dag


def pauli_transpilation_passes() -> Iterable[BasePass]:
    "Yield simple transpilation steps after addition of Pauli gates."
    yield Optimize1qGatesDecomposition(BASIS_GATES)
    yield CXCancellation()
