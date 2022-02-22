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

from copy import deepcopy
from typing import Iterable, List, Optional, TYPE_CHECKING, Union
from qiskit.qasm import pi
import numpy as np

from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.gate import Gate
from qiskit.circuit.library import IGate, XGate, YGate, ZGate, RZXGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.quantum_info import Operator

if TYPE_CHECKING:
    from qiskit.transpiler.basepasses import BasePass

def add_pauli_twirls(
    circuits: Union[QuantumCircuit, List[QuantumCircuit]],
    entangler_str: str,
    num_seeds: int,
    verify=False,
) -> Union[QuantumCircuit, List[QuantumCircuit]]:
    """
    Add pairs of gates before/after entangling gate randomly
    such that they commute. This helps turn coherent error into stochastic
    error.
    """
    if isinstance(circuits, QuantumCircuit):
        circuits = [circuits]

    twirl_op, twirl_gates = get_twirl_gates_list(entangler_str)

    all_twirled_circs = []
    for circ in circuits:
        dag = circuit_to_dag(circ)
        twirled_circs = []
        for seed in range(num_seeds):
            this_dag = deepcopy(dag)
            runs = this_dag.collect_runs([entangler_str])
            np.random.seed(seed)
            twirl_idxs = np.random.randint(0, len(twirl_gates), size=len(runs))
            for twirl_idx, run in enumerate(runs):
                mini_dag = DAGCircuit()
                p = QuantumRegister(2, 'p')
                mini_dag.add_qreg(p)
                mini_dag.apply_operation_back(twirl_gates[twirl_idxs[twirl_idx]][0], qargs=[p[0]])
                mini_dag.apply_operation_back(twirl_gates[twirl_idxs[twirl_idx]][1], qargs=[p[1]])
                mini_dag.apply_operation_back(run[0].op, qargs=[p[0], p[1]])
                mini_dag.apply_operation_back(twirl_gates[twirl_idxs[twirl_idx]][0], qargs=[p[0]])
                mini_dag.apply_operation_back(twirl_gates[twirl_idxs[twirl_idx]][1], qargs=[p[1]])

                twirl_node = this_dag.op_nodes(op=twirl_op).pop()
                this_dag.substitute_node_with_dag(node=run[0], input_dag=mini_dag)

            twirled_circs.append(dag_to_circuit(this_dag))

        all_twirled_circs.append(twirled_circs)

    if verify:
        if not verify_equiv_circuits(circuits, all_twirled_circs):
            print("Twirled circuits are not equivalent!")

    return all_twirled_circs


def get_twirl_gates_list(entangler_str: str) -> List[Gate]:
    """
    Return information based on the entrangling gate to be twirled,
    including the Gate operator and list of commuting pairs of gates.
    """

    if entangler_str == 'rzx':
        return (RZXGate, [[IGate(), IGate()], [XGate(), ZGate()],
                            [YGate(), YGate()], [ZGate(), XGate()]])
    else:
        print("Twirling gates not defined for entangler "+entangler_str)

def verify_equiv_circuits(circuits, all_twirled_circs):
    if isinstance(circuits, QuantumCircuit):
        circuits = [circuits]
    if isinstance(all_twirled_circs, QuantumCircuit):
        all_twirled_circs = [all_twirled_circs]

    all_equiv_circuits = True
    for cidx, circ in enumerate(circuits):
        for t_circ in all_twirled_circs[cidx]:
            param_bind = {}
            for param in circ.parameters:
                param_bind[param] = np.random.random()

            all_equiv_circuits = all_equiv_circuits and Operator(
                circ.bind_parameters(param_bind)).equiv(Operator(t_circ.bind_parameters(param_bind))
            )

    return all_equiv_circuits
