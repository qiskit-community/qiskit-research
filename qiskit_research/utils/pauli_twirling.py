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

from qiskit import pulse
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.gate import Gate
from qiskit.circuit.library import IGate, XGate, YGate, ZGate, RZXGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.providers.ibmq import IBMQBackend
from qiskit.pulse import DriveChannel
from qiskit.quantum_info import Operator

if TYPE_CHECKING:
    from qiskit.transpiler.basepasses import BasePass

def add_pauli_twirls(
    circuits: Union[QuantumCircuit, List[QuantumCircuit]],
    backend: IBMQBackend,
    entangler_str: str,
    seeds: Union[int, List[int]],
    verify=False,
) -> Union[QuantumCircuit, List[QuantumCircuit]]:
    """
    Add pairs of gates before/after entangling gate randomly
    such that they commute. This helps turn coherent error into stochastic
    error.

    Args:
        circuits (:class:`QuantumCircuit` or list(:class:`QuantumCircuit`)):
                  Circuits to be twirled.
        backend: (:class:`IBMQBackend`): backend the circtui will be run on, so
                 that non-basis gates have appropriate calibrations added.
        entangler_str (str): name of the entangling gate which will be twirled.
        seeds (int or list(int)): number of seeds or range of randome seeds to use
        verify (bool): verify Pauli twirls construct equivalent unitaries
    """
    if isinstance(circuits, QuantumCircuit):
        circuits = [circuits]

    twirl_op, twirl_gates = get_twirl_gates_list(entangler_str)
    inst_sched_map = backend.defaults().instruction_schedule_map

    # find non-basis gates which are not defined by backend
    extra_gate_defs = []
    for twirl_pair in twirl_gates:
        for twirl_gate in twirl_pair:
            if twirl_gate.name not in inst_sched_map.instructions:
                if twirl_gate.name not in extra_gate_defs:
                    extra_gate_defs.append(twirl_gate.name)

    all_twirled_circs = []
    for circ in circuits:
        dag = circuit_to_dag(circ)
        twirled_circs = []
        if isinstance(seeds, int):
            seed_array = range(seeds)
        else:
            seed_array = seeds

        for seed in seed_array:
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

                this_dag.substitute_node_with_dag(node=run[0], input_dag=mini_dag)

            twirled_circs.append(dag_to_circuit(this_dag))

            # add pulse gates (calibrations) for non-basis gates
            if 'y' in extra_gate_defs:
                for qubit in range(backend.configuration().num_qubits):
                    with pulse.build('y gate for qubit '+str(qubit)) as sched:
                        # def of YGate() in terms of XGate() and phase_offset
                        with pulse.phase_offset(np.pi/2, DriveChannel(qubit)):
                            x_gate = inst_sched_map.get('x', qubits=[qubit])
                            pulse.call(x_gate)

                    # for each Y twirl
                    for t_circ in twirled_circs:
                        t_circ.add_calibration('y', [qubit], sched)

            if 'z' in extra_gate_defs:
                continue # qiskit knows how to handle these

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
                circ.bind_parameters(param_bind)).equiv(Operator(
                    t_circ.bind_parameters(param_bind)
                )
            )

    return all_equiv_circuits
