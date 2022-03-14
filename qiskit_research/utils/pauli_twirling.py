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

from typing import Any, Iterable, List, Optional, Union, cast

import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import IGate, XGate, YGate, ZGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit

I = IGate()
X = XGate()
Y = YGate()
Z = ZGate()

# this dictionary stores the twirl sets for each supported gate
# each key is the name of a supported gate
# each value is a tuple that represents the twirl set for the gate
# the twirl set is a list of (before, after) pairs describing twirl gates
# "before" and "after" are tuples of single-qubit gates to be applied
#   before and after the gate to be twirled
TWIRL_GATES = {
    "rzx": (
        ((I, I), (I, I)),
        ((X, Z), (X, Z)),
        ((Y, Y), (Y, Y)),
        ((Z, X), (Z, X)),
    ),
    "rzz": (
        ((I, I), (I, I)),
        ((X, X), (X, X)),
        ((Y, Y), (Y, Y)),
        ((Z, Z), (Z, Z)),
    ),
}


def parse_random_seed(seed: Any) -> np.random.Generator:
    """Parse a random number generator seed and return a Generator."""
    if isinstance(seed, np.random.Generator):
        return cast(np.random.Generator, seed)
    return np.random.default_rng(seed)


def add_pauli_twirls(
    circuits: Union[QuantumCircuit, List[QuantumCircuit]],
    num_twirled_circuits: int = 1,
    gates_to_twirl: Optional[Iterable[str]] = None,
    seed: Any = None,
) -> Union[List[QuantumCircuit], List[List[QuantumCircuit]]]:
    """Add Pauli twirls to circuits.

    Args:
        circuits: Circuit or list of circuits to be twirled.
        num_twirled_circuits: Number of twirled circuits to return for each input circuit.
        seed: Seed for the pseudorandom number generator.

    Returns:
        If the input is a single circuit, then a list of circuits is returned.
        If the input is a list of circuit, then a list of lists of circuits is returned.
    """
    if gates_to_twirl is None:
        gates_to_twirl = TWIRL_GATES.keys()
    rng = parse_random_seed(seed)
    if isinstance(circuits, QuantumCircuit):
        return [
            _twirl_circuit(circuits, gates_to_twirl, rng)
            for _ in range(num_twirled_circuits)
        ]
    return [
        [
            _twirl_circuit(circuit, gates_to_twirl, rng)
            for _ in range(num_twirled_circuits)
        ]
        for circuit in circuits
    ]


def _twirl_circuit(
    circuit: QuantumCircuit, gates_to_twirl: Iterable[str], rng: np.random.Generator
) -> QuantumCircuit:
    dag = circuit_to_dag(circuit)
    for run in dag.collect_runs(list(gates_to_twirl)):
        for node in run:
            twirl_gates = TWIRL_GATES[node.op.name]
            (before0, before1), (after0, after1) = twirl_gates[
                rng.integers(len(twirl_gates))
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
    return dag_to_circuit(dag)
