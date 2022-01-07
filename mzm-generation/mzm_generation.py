import itertools
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg
from qiskit import IBMQ, Aer, QuantumCircuit, transpile
from qiskit.providers.ibmq import IBMQJob
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.circuit.library import FermionicGaussianState
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit_nature.operators.second_quantization import (
    FermionicOp,
    QuadraticHamiltonian,
)


def majorana_op(index: int, action: int) -> FermionicOp:
    if action == 0:
        return FermionicOp(f"-_{index}") + FermionicOp(f"+_{index}")
    return 1j * (FermionicOp(f"-_{index}") - FermionicOp(f"+_{index}"))


def edge_correlation_op(n_modes: int) -> FermionicOp:
    return 1j * majorana_op(0, 0) @ majorana_op(n_modes - 1, 1)


def parity_op(n_modes: int) -> FermionicOp:
    op = FermionicOp.one(n_modes)
    for i in range(n_modes):
        op @= FermionicOp.one(n_modes) - 2 * FermionicOp(f"N_{i}")
    return op


def number_op(n_modes: int) -> FermionicOp:
    return sum(FermionicOp(f"N_{i}") for i in range(n_modes))


def expectation(operator: np.ndarray, state: np.ndarray) -> complex:
    return np.vdot(state, operator @ state)


def kitaev_hamiltonian(
    n_modes: int, tunneling: float, superconducting: float, chemical_potential: float
) -> QuadraticHamiltonian:
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


def measure_z(circuit: QuantumCircuit) -> QuantumCircuit:
    circuit = circuit.copy()
    circuit.measure_all()
    return circuit


def measure_x(circuit: QuantumCircuit) -> QuantumCircuit:
    circuit = circuit.copy()
    circuit.h(circuit.qubits)
    circuit.measure_all()
    return circuit


def measure_edge_correlation(circuit: QuantumCircuit) -> QuantumCircuit:
    circuit = circuit.copy()
    circuit.rx(np.pi / 2, circuit.qubits[0])
    circuit.rx(np.pi / 2, circuit.qubits[-1])
    circuit.measure_all()
    return circuit


def run_job(backend, circuit: QuantumCircuit, shots: int) -> IBMQJob:
    z_circuit = transpile(measure_z(circuit), backend)
    x_circuit = transpile(measure_x(circuit), backend)
    edge_correlation_circuit = transpile(measure_edge_correlation(circuit), backend)
    return backend.run([z_circuit, x_circuit, edge_correlation_circuit], shots=shots)


def compute_measure_edge_correlation(job: IBMQJob) -> float:
    counts = job.result().get_counts(2)
    shots = sum(counts.values())
    correlation_expectation = 0.0
    for bitstring, count in counts.items():
        parity = sum(1 for b in bitstring if b == "1")
        correlation_expectation -= (-1) ** parity * count
    correlation_expectation /= shots
    return np.real(correlation_expectation)


def compute_measure_hamiltonian(job: IBMQJob, hamiltonian: SparsePauliOp) -> float:
    # Assumes Hamiltonian only has X strings and Z strings
    counts_z = job.result().get_counts(0)
    counts_x = job.result().get_counts(1)
    shots_z = sum(counts_z.values())
    shots_x = sum(counts_x.values())
    hamiltonian_expectation = 0.0
    for term, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
        if np.any(term.z):
            counts = counts_z
            shots = shots_z
            pauli_string = term.z
        elif np.any(term.x):
            counts = counts_x
            shots = shots_x
            pauli_string = term.x
        else:
            hamiltonian_expectation += coeff
            continue
        term_expectation = 0.0
        for bitstring, count in counts.items():
            parity = sum(
                1 for i, b in enumerate(bitstring) if pauli_string[i] and b == "1"
            )
            term_expectation += (-1) ** parity * count
        term_expectation /= shots
        hamiltonian_expectation += coeff * term_expectation
    return np.real(hamiltonian_expectation)
