import dataclasses
import json
import os
from typing import Any, Dict, Iterable, List, Tuple, Union

import mthree
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Gate, Qubit
from qiskit.circuit.library import XYGate
from qiskit.providers.ibmq import IBMQBackend, IBMQJob
from qiskit.providers.aer import AerJob
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.circuit.library import FermionicGaussianState
from qiskit_nature.operators.second_quantization import (
    FermionicOp,
    QuadraticHamiltonian,
)
from yx_plus_xy_interaction import YXPlusXYInteractionGate


def save(task, data, base_dir: str = "data/", mode="x"):
    filename = os.path.join(base_dir, f"{task.filename}.json")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, mode) as f:
        json.dump(data, f)


def load(task, base_dir: str = "data/"):
    filename = os.path.join(base_dir, f"{task.filename}.json")
    with open(filename) as f:
        data = json.load(f)
    return data


@dataclasses.dataclass
class FermionicGaussianStateParameters:
    n_modes: int
    tunneling: float
    superconducting: float
    chemical_potential: float
    occupied_orbitals: Tuple[int]


@dataclasses.dataclass
class FermionicGaussianStateTask:
    experiment_id: str
    shots: Tuple[int]
    state_params: FermionicGaussianStateParameters

    @property
    def filename(self) -> str:
        return os.path.join(
            self.experiment_id,
            f"shots{self.shots}",
            f"n{self.state_params.n_modes}",
            f"t{self.state_params.tunneling:.2f}"
            f"_Delta{self.state_params.superconducting:.2f}"
            f"_mu{self.state_params.chemical_potential:.2f}",
            str(self.state_params.occupied_orbitals),
        )


@dataclasses.dataclass
class MeasurementErrorCalibrationTask:
    experiment_id: str
    shots: int

    @property
    def filename(self) -> str:
        return os.path.join(
            self.experiment_id,
            "measurement_error_calibration",
            f"shots{self.shots}",
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


def variance(operator: np.ndarray, state: np.ndarray) -> complex:
    return expectation(operator ** 2, state) - expectation(operator, state) ** 2


def kitaev_hamiltonian(
    n_modes: int, tunneling: float, superconducting: float, chemical_potential: float
) -> QuadraticHamiltonian:
    eye = np.eye(n_modes)
    upper_diag = np.diag(np.ones(n_modes - 1), k=1)
    lower_diag = np.diag(np.ones(n_modes - 1), k=-1)
    hermitian_part = -tunneling * (upper_diag + lower_diag) + chemical_potential * eye
    antisymmetric_part = superconducting * (upper_diag - lower_diag)
    return QuadraticHamiltonian(hermitian_part, antisymmetric_part)


def state_preparation_circuit(
    params: FermionicGaussianStateParameters,
) -> FermionicGaussianState:
    hamiltonian = kitaev_hamiltonian(
        params.n_modes,
        tunneling=params.tunneling,
        superconducting=params.superconducting,
        chemical_potential=params.chemical_potential,
    )
    transformation_matrix, _, _ = hamiltonian.diagonalizing_bogoliubov_transform()
    return FermionicGaussianState(
        transformation_matrix, params.occupied_orbitals, name=repr(params)
    )


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


def measurement_pauli_strings(n_qubits: int) -> Iterable[str]:
    # NOTE these strings are in big endian order (opposite of qiskit)
    yield "x" * n_qubits
    yield "y" * n_qubits
    yield "z" * n_qubits
    for i in range(n_qubits - 1):
        yield "y" + "z" * i + "x"
        yield "y" + "z" * i + "y"


def measure_pauli_string(circuit: QuantumCircuit, pauli_string: str) -> QuantumCircuit:
    circuit = circuit.copy(name=pauli_string)
    for q, pauli in zip(circuit.qubits, pauli_string):
        if pauli.lower() == "x":
            circuit.h(q)
        elif pauli.lower() == "y":
            circuit.rx(np.pi / 2, q)
    circuit.measure_all()
    return circuit


def measure_pauli_strings(
    circuit: QuantumCircuit, pauli_strings: Iterable[str]
) -> Iterable[QuantumCircuit]:
    for pauli in pauli_strings:
        yield measure_pauli_string(circuit, pauli)


def measure_tunneling_ops(circuit: QuantumCircuit) -> Iterable[QuantumCircuit]:
    # TODO add circuit name or metadata
    for start_index in (0, 1):
        for i in range(start_index, circuit.num_qubits - 1, 2):
            # measure tunneling op between modes i and i + 1
            circ = circuit.copy(name=f"tunneling_{i}_{i + 1}")
            circ.append(XYGate(np.pi / 2, -np.pi / 2), (i, i + 1))
            circ.measure_all()
            yield circ


def measure_superconducting_ops(circuit: QuantumCircuit) -> Iterable[QuantumCircuit]:
    for start_index in (0, 1):
        for i in range(start_index, circuit.num_qubits - 1, 2):
            # measure superconducting op between modes i and i + 1
            circ = circuit.copy(name=f"superconducting_{i}_{i + 1}")
            circ.append(YXPlusXYInteractionGate(-np.pi / 4), (i, i + 1))
            circ.measure_all()
            yield circ


# TODO saving should be done separately but mthree does not support that
# TODO calibration date needs to be handled better
# (see https://github.com/Qiskit-Partners/mthree/issues/71)
def run_measurement_error_calibration_task(
    task: MeasurementErrorCalibrationTask,
    backend,
    qubits: List[int],
    base_dir: str = "data/",
) -> None:
    filename = os.path.join(base_dir, f"{task.filename}.json")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    mit = mthree.M3Mitigation(backend)
    mit.cals_from_system(qubits, shots=task.shots, cals_file=filename)


def run_fermionic_gaussian_state_task(
    task: FermionicGaussianStateTask, backend, qubits: List[int]
) -> Union[AerJob, IBMQJob]:
    circuit = state_preparation_circuit(task.state_params)
    pauli_measurement_circuits = measure_pauli_strings(
        circuit, measurement_pauli_strings(circuit.num_qubits)
    )
    tunneling_measurement_circuits = measure_tunneling_ops(circuit)
    superconducting_measurement_circuits = measure_superconducting_ops(circuit)
    all_measurement_circuits = [
        *pauli_measurement_circuits,
        *tunneling_measurement_circuits,
        *superconducting_measurement_circuits,
    ]
    all_measurement_circuits_transpiled = [
        transpile(circ, backend, initial_layout=qubits)
        for circ in all_measurement_circuits
    ]
    job = backend.run(all_measurement_circuits_transpiled, shots=task.shots)
    # HACK AerJob has no circuits method
    # See https://github.com/Qiskit/qiskit-aer/issues/1431
    if isinstance(job, AerJob):
        job.circuits = lambda: all_measurement_circuits_transpiled
    return job


def run_measurement_error_correction(
    measurement_error_calibration_task: MeasurementErrorCalibrationTask,
    fermionic_gaussian_state_task: FermionicGaussianStateTask,
    qubits: List[int],
    base_dir: str = "data/",
) -> None:
    mit = mthree.M3Mitigation()
    mit.cals_from_file(
        os.path.join(base_dir, f"{measurement_error_calibration_task.filename}.json")
    )
    data = load(fermionic_gaussian_state_task)
    measurements = data["measurements"]
    # TODO save mitigation overhead
    quasis = {
        pauli_string: mit.apply_correction(
            counts, qubits, return_mitigation_overhead=True
        )
        for pauli_string, counts in measurements.items()
    }
    data["quasis"] = quasis
    save(fermionic_gaussian_state_task, data, mode="w")


def compute_energy(
    measurements: Dict["str", Dict["str", int]],
    hamiltonian: SparsePauliOp,
) -> Tuple[float, float]:
    # TODO standard deviation estimate needs to include covariances
    # Assumes Hamiltonian only has X strings, Y strings, and Z strings
    counts_x = measurements["x" * hamiltonian.num_qubits]
    counts_y = measurements["y" * hamiltonian.num_qubits]
    counts_z = measurements["z" * hamiltonian.num_qubits]
    shots_x = sum(counts_x.values())
    shots_y = sum(counts_y.values())
    shots_z = sum(counts_z.values())
    hamiltonian_expectation = 0.0
    hamiltonian_var = 0.0
    for term, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
        term_y = np.logical_and(term.z, term.x)
        if np.any(term_y):
            counts = counts_y
            shots = shots_y
            pauli_string = term_y
        elif np.any(term.z):
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
        parities = {1: 0.0, -1: 0.0}
        for bitstring, count in counts.items():
            parity = (-1) ** sum(
                1 for i, b in enumerate(bitstring) if pauli_string[i] and b == "1"
            )
            parities[parity] += count
        term_expectation = sum(p * count for p, count in parities.items()) / shots
        term_var = sum(
            (p - term_expectation) ** 2 * count for p, count in parities.items()
        ) / (shots - 1)
        hamiltonian_expectation += coeff * term_expectation
        hamiltonian_var += abs(coeff) ** 2 * term_var / shots
    return np.real(hamiltonian_expectation), np.sqrt(hamiltonian_var)


def compute_energy_measurement_corrected(
    quasis: Dict["str", Dict["str", float]], hamiltonian: SparsePauliOp
) -> Tuple[float, float]:
    # TODO standard deviation estimate needs to include covariances
    quasis_x = quasis["x" * hamiltonian.num_qubits]
    quasis_y = quasis["y" * hamiltonian.num_qubits]
    quasis_z = quasis["z" * hamiltonian.num_qubits]
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
        operator = "".join("Z" if b else "I" for b in pauli_string)
        term_expectation, term_stddev = quasi_dist.expval_and_stddev(operator)
        hamiltonian_expectation += coeff * term_expectation
        hamiltonian_var += abs(coeff) ** 2 * term_stddev ** 2
    return np.real(hamiltonian_expectation), np.sqrt(hamiltonian_var)


def compute_edge_correlation(measurements: Dict["str", Dict["str", int]]) -> float:
    # TODO estimate standard deviation
    n_qubits = len(next(iter(measurements)))
    counts = measurements["y" + "z" * (n_qubits - 2) + "y"]
    shots = sum(counts.values())
    correlation_expectation = 0.0
    for bitstring, count in counts.items():
        parity = sum(1 for b in bitstring if b == "1")
        correlation_expectation -= (-1) ** parity * count
    correlation_expectation /= shots
    return np.real(correlation_expectation)


def compute_edge_correlation_measurement_corrected(
    quasis: Dict["str", Dict["str", float]],
) -> float:
    # TODO estimate standard deviation
    n_qubits = len(next(iter(quasis)))
    quasi_dist = quasis["y" + "z" * (n_qubits - 2) + "y"]
    correlation_expectation = -quasi_dist.expval()
    return correlation_expectation


def compute_parity(measurements: Dict["str", Dict["str", int]]) -> float:
    # TODO estimate standard deviation
    n_qubits = len(next(iter(measurements)))
    counts = measurements["z" * n_qubits]
    shots = sum(counts.values())
    parity_expectation = 0.0
    for bitstring, count in counts.items():
        parity = sum(1 for b in bitstring if b == "1")
        parity_expectation += (-1) ** parity * count
    parity_expectation /= shots
    return parity_expectation


def compute_parity_measurement_corrected(
    quasis: Dict["str", Dict["str", float]],
) -> float:
    # TODO estimate standard deviation
    n_qubits = len(next(iter(quasis)))
    quasi_dist = quasis["z" * n_qubits]
    parity_expectation = quasi_dist.expval()
    return parity_expectation


def compute_number(measurements: Dict["str", Dict["str", int]]) -> float:
    # TODO estimate standard deviation
    n_qubits = len(next(iter(measurements)))
    counts = measurements["z" * n_qubits]
    shots = sum(counts.values())
    number_expectation = 0.0
    for bitstring, count in counts.items():
        number = sum(1 for b in bitstring if b == "1")
        number_expectation += number * count
    number_expectation /= shots
    return number_expectation


def compute_number_measurement_corrected(
    quasis: Dict["str", Dict["str", float]],
) -> float:
    # TODO estimate standard deviation
    n_qubits = len(next(iter(quasis)))
    quasi_dist = quasis["z" * n_qubits]
    projectors = ["I" * k + "1" + "I" * (n_qubits - k - 1) for k in range(n_qubits)]
    number_expectation = np.sum(quasi_dist.expval(projectors))
    return number_expectation
