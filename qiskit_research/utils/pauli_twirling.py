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
import itertools
import cmath

import numpy as np
from qiskit.circuit import QuantumRegister, QuantumCircuit
from qiskit.circuit.library import (
    IGate,
    XGate,
    YGate,
    ZGate,
    CXGate,
    CYGate,
    CZGate,
    CHGate,
    CSGate,
    DCXGate,
    CSXGate,
    CSdgGate,
    ECRGate,
    iSwapGate,
    SwapGate,
)
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import BasePass, TransformationPass
from qiskit.transpiler.passes import (
    CXCancellation,
    Optimize1qGatesDecomposition,
)
from qiskit.quantum_info import Pauli, Operator, pauli_basis
from qiskit_research.utils.pulse_scaling import BASIS_GATES

# Single qubit Pauli gates
I = IGate()
X = XGate()
Y = YGate()
Z = ZGate()

# 2Q entangling gates
CX = CXGate()  # cnot; controlled-X
CY = CYGate()  # controlled-Y
CZ = CZGate()  # controlled-Z
CH = CHGate()  # controlled-Hadamard
CS = CSGate()  # controlled-S
DCX = DCXGate()  # double cnot
CSX = CSXGate()  # controlled sqrt X
CSdg = CSdgGate()  # controlled S^dagger
ECR = ECRGate()  # echoed cross-resonance
Swap = SwapGate()  # swap
iSwap = iSwapGate()  # imaginary swap

# this list consists of the 2-qubit rotation gates
TWO_QUBIT_PAULI_GENERATORS = {
    "rxx": Pauli("XX"),
    "ryy": Pauli("YY"),
    "rzx": Pauli("XZ"),
    "rzz": Pauli("ZZ"),
    "secr": Pauli("XZ"),
}


def match_global_phase(a, b):
    """Phase the given arrays so that their phases match at one entry.

    Args:
        a: A Numpy array.
        b: Another Numpy array.

    Returns:
        A pair of arrays (a', b') that are equal if and only if a == b * exp(i phi)
        for some real number phi.
    """
    if a.shape != b.shape:
        return a, b
    # use the largest entry of one of the matrices to maximize precision
    index = max(np.ndindex(*a.shape), key=lambda i: abs(b[i]))
    phase_a = cmath.phase(a[index])
    phase_b = cmath.phase(b[index])
    return a * cmath.rect(1, -phase_a), b * cmath.rect(1, -phase_b)


def allclose_up_to_global_phase(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    """Check if two operators are close up to a global phase."""
    # Phase both operators to match their phases
    phased_op1, phased_op2 = match_global_phase(a, b)
    return np.allclose(phased_op1, phased_op2, rtol, atol, equal_nan)


def create_pauli_twirling_sets(two_qubit_gate):
    """Generate the Pauli twirling sets for a given 2Q gate.

    Sets are ordered such that gate[0] and gate[1] are pre-rotations
    applied to control and target, respectively. gate[2] and gate[3]
    are post-rotations for control and target, respectively.

    Parameters:
        two_qubit_gate (Gate): Input two-qubit gate

    Returns:
        tuple: Tuple of all twirling gate sets
    """

    target_unitary = np.array(two_qubit_gate)
    twirling_sets = []

    # Generate combinations of 4 gates from the operator list
    for gates in itertools.product(itertools.product([I, X, Y, Z], repeat=2), repeat=2):
        qc = _build_twirl_circuit(gates, two_qubit_gate)
        qc_array = Operator.from_circuit(qc).to_matrix()
        if allclose_up_to_global_phase(qc_array, target_unitary):
            twirling_sets.append(gates)

    return tuple(twirling_sets)


def _build_twirl_circuit(gates, two_qubit_gate):
    """Build the twirled quantum circuit with specified gates."""
    qc = QuantumCircuit(2)

    qc.append(gates[0][0], [0])
    qc.append(gates[0][1], [1])
    qc.append(two_qubit_gate, [0, 1])
    qc.append(gates[1][0], [0])
    qc.append(gates[1][1], [1])

    return qc


# this dictionary stores the twirl sets for each supported gate
# each key is the name of a supported gate
# each value is a tuple that represents the twirl set for the gate
# the twirl set is a list of (before, after) pairs describing twirl gates
# "before" and "after" are tuples of single-qubit gates to be applied
# before and after the gate to be twirled
TWIRL_GATES = {
    "cx": create_pauli_twirling_sets(CX),
    "cy": create_pauli_twirling_sets(CY),
    "cz": create_pauli_twirling_sets(CZ),
    "ch": create_pauli_twirling_sets(CH),
    "cs": create_pauli_twirling_sets(CS),
    "dcx": create_pauli_twirling_sets(DCX),
    "csx": create_pauli_twirling_sets(CSX),
    "csdg": create_pauli_twirling_sets(CSdg),
    "ecr": create_pauli_twirling_sets(ECR),
    "swap": create_pauli_twirling_sets(Swap),
    "iswap": create_pauli_twirling_sets(iSwap),
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
                    mini_dag.add_qubits([q0, q1])

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
