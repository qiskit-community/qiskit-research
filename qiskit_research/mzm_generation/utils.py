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

import functools
from typing import Dict, FrozenSet, Tuple

import mthree
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import XYGate
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.operators.second_quantization import (
    FermionicOp,
    QuadraticHamiltonian,
)
from qiskit_research.mzm_generation.phased_xx_minus_yy import PhasedXXMinusYYGate


def majorana_op(index: int, action: int) -> FermionicOp:
    if action == 0:
        return FermionicOp(f"-_{index}") + FermionicOp(f"+_{index}")
    return -1j * (FermionicOp(f"-_{index}") - FermionicOp(f"+_{index}"))


def edge_correlation_op(n_modes: int) -> FermionicOp:
    return -1j * majorana_op(0, 0) @ majorana_op(n_modes - 1, 1)


def parity_op(n_modes: int) -> FermionicOp:
    op = FermionicOp.one(n_modes)
    for i in range(n_modes):
        op @= FermionicOp.one(n_modes) - 2 * FermionicOp(f"N_{i}")
    return op


def number_op(n_modes: int) -> FermionicOp:
    return sum(FermionicOp(f"N_{i}") for i in range(n_modes))


# TODO operator could be scipy sparse matrix
def expectation(operator: np.ndarray, state: np.ndarray) -> complex:
    """Expectation value of operator with state."""
    return np.vdot(state, operator @ state)


# TODO operator could be scipy sparse matrix
def variance(operator: np.ndarray, state: np.ndarray) -> complex:
    """Variance of operator with state."""
    return expectation(operator ** 2, state) - expectation(operator, state) ** 2


@functools.lru_cache
def kitaev_hamiltonian(
    n_modes: int, tunneling: float, superconducting: float, chemical_potential: float
) -> QuadraticHamiltonian:
    """Create Kitaev model Hamiltonian."""
    eye = np.eye(n_modes)
    upper_diag = np.diag(np.ones(n_modes - 1), k=1)
    lower_diag = np.diag(np.ones(n_modes - 1), k=-1)
    hermitian_part = -tunneling * (upper_diag + lower_diag) + chemical_potential * eye
    antisymmetric_part = superconducting * (upper_diag - lower_diag)
    return QuadraticHamiltonian(hermitian_part, antisymmetric_part)


def bdg_hamiltonian(hamiltonian: QuadraticHamiltonian) -> np.ndarray:
    return np.block(
        [
            [
                -hamiltonian.hermitian_part.conj(),
                -hamiltonian.antisymmetric_part.conj(),
            ],
            [hamiltonian.antisymmetric_part, hamiltonian.hermitian_part],
        ]
    )


def measure_pauli_string(circuit: QuantumCircuit, pauli_string: str) -> QuantumCircuit:
    """Measure a Pauli string."""
    circuit = circuit.copy()
    for q, pauli in zip(circuit.qubits, pauli_string):
        if pauli.lower() == "x":
            circuit.h(q)
        elif pauli.lower() == "y":
            circuit.rx(np.pi / 2, q)
    circuit.measure_all()
    return circuit


def measure_interaction_op(circuit: QuantumCircuit, label: str) -> QuantumCircuit:
    """Measure fermionic interaction with a circuit that preserves parity."""
    if label.startswith("tunneling_plus"):
        # this gate transforms a^\dagger_i a_j + h.c into (IZ - ZI) / 2
        gate = XYGate(np.pi / 2, -np.pi / 2)
    elif label.startswith("tunneling_minus"):
        # this gate transforms -i * (a^\dagger_i a_j - h.c) into (IZ - ZI) / 2
        gate = XYGate(np.pi / 2, -np.pi)
    elif label.startswith("superconducting_plus"):
        # this gate transforms a^\dagger_i a^_\dagger_j + h.c into (IZ + ZI) / 2
        gate = PhasedXXMinusYYGate(np.pi / 2, -np.pi / 2)
    else:  # label.startswith("superconducting_minus")
        # this gate transforms -i * (a^\dagger_i a^_\dagger_j - h.c) into (IZ + ZI) / 2
        gate = PhasedXXMinusYYGate(np.pi / 2, -np.pi)

    if label.endswith("even"):
        start_index = 0
    else:  # label.endswith('odd')
        start_index = 1

    circuit = circuit.copy()
    for i in range(start_index, circuit.num_qubits - 1, 2):
        # measure interaction between qubits i and i + 1
        circuit.append(gate, [i, i + 1])
    circuit.measure_all()

    return circuit


def compute_energy_pauli_basis(
    quasis: Dict[str, Dict[str, float]], hamiltonian: SparsePauliOp
) -> Tuple[float, float]:
    """Compute energy given quasiprobabilities of Pauli strings."""
    # TODO standard deviation estimate needs to include covariances
    quasis_x = quasis["pauli", "x" * hamiltonian.num_qubits]
    quasis_y = quasis["pauli", "y" * hamiltonian.num_qubits]
    quasis_z = quasis["pauli", "z" * hamiltonian.num_qubits]
    hamiltonian_expectation = 0.0
    hamiltonian_var = 0.0
    for term, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
        term_y = np.logical_and(term.z, term.x)
        if np.any(term_y):
            quasi_dist = quasis_y
            pauli_string = term_y
        elif np.any(term.z):
            quasi_dist = quasis_z
            pauli_string = term.z
        elif np.any(term.x):
            quasi_dist = quasis_x
            pauli_string = term.x
        else:
            hamiltonian_expectation += coeff
            continue
        # don't reverse pauli string because M3 does it internally!
        operator = "".join("Z" if b else "I" for b in pauli_string)
        term_expectation, term_stddev = quasi_dist.expval_and_stddev(operator)
        hamiltonian_expectation += coeff * term_expectation
        hamiltonian_var += abs(coeff) ** 2 * term_stddev ** 2
    return np.real(hamiltonian_expectation), np.sqrt(hamiltonian_var)


def compute_interaction_matrix(
    quasis: Dict[str, Dict[str, float]], label: str
) -> np.ndarray:
    """Compute interaction operator given quasiprobabilities."""
    n_qubits = len(next(iter(next(iter(quasis.values())))))

    if label == "tunneling_plus":
        sign = -1
        symmetry = 1
    elif label == "tunneling_minus":
        sign = -1
        symmetry = -1
    elif label == "superconducting_plus":
        sign = 1
        symmetry = -1
    else:  # label == "superconducting_minus"
        sign = 1
        symmetry = -1

    even_quasis = quasis["parity", f"{label}_even"]
    odd_quasis = quasis["parity", f"{label}_odd"]
    z_quasis = quasis["pauli", "z" * n_qubits]

    mat = np.zeros((n_qubits, n_qubits))
    # diagonal terms
    if label == "tunneling_plus":
        for i in range(n_qubits):
            # don't reverse pauli string because M3 does it internally!
            num = "I" * i + "1" + "I" * (n_qubits - i - 1)
            expval, stddev = z_quasis.expval_and_stddev(num)
            mat[i, i] = 2 * expval
    # off-diagonal terms
    for start_index in [0, 1]:
        quasi_dist = odd_quasis if start_index else even_quasis
        for i in range(start_index, n_qubits - 1, 2):
            # don't reverse pauli string because M3 does it internally!
            z0 = "I" * i + "Z" + "I" * (n_qubits - i - 1)
            z1 = "I" * (i + 1) + "Z" + "I" * (n_qubits - i - 2)
            z0_expval, z0_stddev = quasi_dist.expval_and_stddev(z0)
            z1_expval, z1_stddev = quasi_dist.expval_and_stddev(z1)
            val = 0.5 * (z1_expval + sign * z0_expval)
            mat[i, i + 1] = val
            mat[i + 1, i] = symmetry * val

    return mat


def compute_energy_parity_basis(
    quasis: Dict[str, Dict[str, float]], hamiltonian: QuadraticHamiltonian
) -> float:
    """Compute energy given quasiprobabilities of interaction operator measurements."""
    tunneling_plus = compute_interaction_matrix(quasis, "tunneling_plus")
    superconducting_plus = compute_interaction_matrix(quasis, "superconducting_plus")
    return (
        0.5
        * np.sum(
            hamiltonian.hermitian_part * tunneling_plus
            + hamiltonian.antisymmetric_part * superconducting_plus
        )
        + hamiltonian.constant
    )


def compute_edge_correlation(quasis: Dict[str, Dict[str, float]]) -> float:
    # TODO estimate standard deviation
    n_qubits = len(next(iter(next(iter(quasis.values())))))
    quasi_dist = quasis["pauli", "y" + "z" * (n_qubits - 2) + "y"]
    correlation_expectation = -quasi_dist.expval()
    return correlation_expectation


def compute_parity(quasis: Dict[str, Dict[str, float]]) -> float:
    # TODO estimate standard deviation
    n_qubits = len(next(iter(next(iter(quasis.values())))))
    quasi_dist = quasis["pauli", "z" * n_qubits]
    parity_expectation = quasi_dist.expval()
    return parity_expectation


def compute_number(quasis: Dict[str, Dict[str, float]]) -> float:
    # TODO estimate standard deviation
    n_qubits = len(next(iter(next(iter(quasis.values())))))
    quasi_dist = quasis["pauli", "z" * n_qubits]
    projectors = ["I" * k + "1" + "I" * (n_qubits - k - 1) for k in range(n_qubits)]
    number_expectation = np.sum(quasi_dist.expval(projectors))
    return number_expectation


def post_select_quasis(
    quasis: mthree.classes.QuasiDistribution, parity: int
) -> Tuple[mthree.classes.QuasiDistribution, float]:
    new_quasis = quasis.copy()
    removed_mass = 0.0
    # set bitstrings with wrong parity to zero
    for bitstring in new_quasis:
        if (-1) ** sum(1 for b in bitstring if b == "1") != parity:
            removed_mass += abs(new_quasis[bitstring])
            new_quasis[bitstring] = 0.0
    # normalize
    normalization = sum(new_quasis.values())
    for bitstring in new_quasis:
        new_quasis[bitstring] /= normalization
    return (
        mthree.classes.QuasiDistribution(
            new_quasis,
            shots=quasis.shots,
            mitigation_overhead=quasis.mitigation_overhead,
        ),
        removed_mass,
    )


def counts_to_quasis(counts: Dict[str, int]) -> mthree.classes.QuasiDistribution:
    shots = sum(counts.values())
    data = {bitstring: count / shots for bitstring, count in counts.items()}
    return mthree.classes.QuasiDistribution(data, shots=shots, mitigation_overhead=1.0)
