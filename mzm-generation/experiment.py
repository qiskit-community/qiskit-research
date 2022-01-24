import functools
import itertools
import os
from collections import defaultdict, namedtuple
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import mthree
import numpy as np
from matplotlib.figure import Figure
from qiskit import Aer, QuantumCircuit
from qiskit.providers import Backend
from qiskit_experiments.framework import (
    AnalysisResultData,
    BaseAnalysis,
    BaseExperiment,
    ExperimentData,
)
from qiskit_nature.circuit.library import FermionicGaussianState
from qiskit_nature.converters.second_quantization.qubit_converter import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper

from mzm_generation import (
    compute_edge_correlation,
    compute_edge_correlation_measurement_corrected,
    compute_energy_pauli,
    compute_energy_pauli_measurement_corrected,
    compute_number,
    compute_number_measurement_corrected,
    compute_parity,
    compute_parity_measurement_corrected,
    edge_correlation_op,
    expectation,
    kitaev_hamiltonian,
    measure_pauli_string,
    measurement_pauli_strings,
    number_op,
)

CircuitParameters = namedtuple(
    "CircuitParameters",
    [
        "tunneling",
        "superconducting",
        "chemical_potential",
        "occupied_orbitals",
        "measurement_basis",
        "measurement_label",
    ],
)


class KitaevHamiltonianExperiment(BaseExperiment):
    """Prepare and measure eigenstates of the Kitaev Hamiltonian."""

    def __init__(
        self,
        experiment_id: str,
        qubits: Sequence[int],
        tunneling_values: float,
        superconducting_values: float,
        chemical_potential_values: float,
        occupied_orbitals_list: Sequence[Tuple[int]],
        backend: Optional[Backend] = None,
    ) -> None:
        self.experiment_id = experiment_id
        self.n_modes = len(qubits)
        self.tunneling_values = tunneling_values
        self.superconducting_values = superconducting_values
        self.chemical_potential_values = chemical_potential_values
        self.occupied_orbitals_list = occupied_orbitals_list
        super().__init__(qubits=qubits, backend=backend)

    def circuits(self) -> List[QuantumCircuit]:
        return list(self._circuits())

    def _circuits(self) -> Iterable[QuantumCircuit]:
        for circuit_params in self.circuit_parameters():
            yield self._generate_circuit(circuit_params)

    def _generate_circuit(self, circuit_params: CircuitParameters) -> QuantumCircuit:
        base_circuit = self._base_circuit(
            circuit_params.tunneling,
            circuit_params.superconducting,
            circuit_params.chemical_potential,
            circuit_params.occupied_orbitals,
        )
        if circuit_params.measurement_basis == "pauli":
            circuit = measure_pauli_string(
                base_circuit, circuit_params.measurement_label
            )
            circuit.metadata = {"params": circuit_params}
            return circuit

    @functools.lru_cache
    def _base_circuit(
        self,
        tunneling: float,
        superconducting: float,
        chemical_potential: float,
        occupied_orbitals: Tuple[int],
    ) -> QuantumCircuit:
        hamiltonian = kitaev_hamiltonian(
            self.n_modes,
            tunneling=tunneling,
            superconducting=superconducting,
            chemical_potential=chemical_potential,
        )
        transformation_matrix, _, _ = hamiltonian.diagonalizing_bogoliubov_transform()
        return FermionicGaussianState(transformation_matrix, occupied_orbitals)

    def circuit_parameters(self) -> Iterable[CircuitParameters]:
        for (
            tunneling,
            superconducting,
            chemical_potential,
            occupied_orbitals,
        ) in itertools.product(
            self.tunneling_values,
            self.superconducting_values,
            self.chemical_potential_values,
            self.occupied_orbitals_list,
        ):
            for pauli_string in measurement_pauli_strings(self.n_modes):
                yield CircuitParameters(
                    tunneling=tunneling,
                    superconducting=superconducting,
                    chemical_potential=chemical_potential,
                    occupied_orbitals=occupied_orbitals,
                    measurement_basis="pauli",
                    measurement_label=pauli_string,
                )


class KitaevHamiltonianAnalysis(BaseAnalysis):
    "Analyze Kitaev Hamiltonian experimental data."

    def _run_analysis(
        self, experiment_data: ExperimentData
    ) -> Tuple[List[AnalysisResultData], List[Figure]]:
        experiment = experiment_data.experiment
        experiment_id = experiment.experiment_id

        # put data into dictionary for easier handling
        data = dict(zip(experiment.circuit_parameters(), experiment_data.data()))

        # load readout calibration
        mit = mthree.M3Mitigation()
        mit.cals_from_file(
            os.path.join("data", experiment_id, f"readout_calibration.json")
        )

        results = list(self._compute_analysis_results(experiment, data, mit))
        return results, []

    def _compute_analysis_results(
        self,
        experiment: KitaevHamiltonianExperiment,
        data: Dict[CircuitParameters, Dict],
        mit: mthree.M3Mitigation,
    ) -> Iterable[AnalysisResultData]:
        simulator_backend = Aer.get_backend("statevector_simulator")
        converter = QubitConverter(mapper=JordanWignerMapper())

        # fix tunneling and superconducting
        tunneling = -1.0
        superconducting = 1.0

        # construct operators
        edge_correlation = edge_correlation_op(experiment.n_modes)
        edge_correlation_jw = converter.convert(edge_correlation)
        # TODO should probably use sparse matrix here
        edge_correlation_dense = edge_correlation_jw.to_matrix()
        number = number_op(experiment.n_modes)
        number_jw = converter.convert(number)
        # TODO should probably use sparse matrix here
        number_dense = number_jw.to_matrix()

        # data storage objects
        energy_exact = defaultdict(list)  # Dict[Tuple[int, ...], List[float]]
        energy_raw = defaultdict(
            list
        )  # Dict[Tuple[int, ...], List[Tuple[float, float]]]
        energy_mem = defaultdict(
            list
        )  # Dict[Tuple[int, ...], List[Tuple[float, float]]]
        # TODO add stddev and type annotations
        edge_correlation_exact = defaultdict(list)  # Dict[Tuple[int, ...], List[float]]
        edge_correlation_raw = defaultdict(list)
        edge_correlation_mem = defaultdict(list)
        parity_exact = defaultdict(list)  # Dict[Tuple[int, ...], List[float]]
        parity_raw = defaultdict(list)
        parity_mem = defaultdict(list)
        number_exact = defaultdict(list)
        number_raw = defaultdict(list)
        number_mem = defaultdict(list)

        for chemical_potential in experiment.chemical_potential_values:
            # create Hamiltonian
            hamiltonian_quad = kitaev_hamiltonian(
                experiment.n_modes,
                tunneling=tunneling,
                superconducting=superconducting,
                chemical_potential=chemical_potential,
            )
            hamiltonian = hamiltonian_quad._fermionic_op()
            hamiltonian_jw = converter.convert(hamiltonian).primitive
            # compute energy
            (
                transformation_matrix,
                orbital_energies,
                constant,
            ) = hamiltonian_quad.diagonalizing_bogoliubov_transform()
            energy_shift = -0.5 * np.sum(orbital_energies) - constant
            # compute parity
            W1 = transformation_matrix[:, : experiment.n_modes]
            W2 = transformation_matrix[:, experiment.n_modes :]
            full_transformation = np.block([[W1, W2], [W2.conj(), W1.conj()]])
            hamiltonian_parity = np.real(np.sign(np.linalg.det(full_transformation)))
            # compute exact and experimental values
            for occupied_orbitals in experiment.occupied_orbitals_list:
                measurements = {
                    pauli_string: data[
                        CircuitParameters(
                            tunneling,
                            superconducting,
                            chemical_potential,
                            occupied_orbitals,
                            "pauli",
                            pauli_string,
                        )
                    ]["counts"]
                    for pauli_string in measurement_pauli_strings(experiment.n_modes)
                }
                quasis = {
                    pauli_string: mit.apply_correction(
                        counts,
                        experiment.physical_qubits,
                        return_mitigation_overhead=True,
                    )
                    for pauli_string, counts in measurements.items()
                }
                # exact values
                circuit = FermionicGaussianState(
                    transformation_matrix, occupied_orbitals
                )
                transpiled_circuit = circuit.decompose()
                state = (
                    simulator_backend.run(transpiled_circuit)
                    .result()
                    .get_statevector()
                    .data
                )
                exact_energy = (
                    np.sum(orbital_energies[list(occupied_orbitals)]) + constant
                )
                exact_correlation = np.real(expectation(edge_correlation_dense, state))
                exact_parity = (-1) ** len(occupied_orbitals) * hamiltonian_parity
                exact_number = np.real(expectation(number_dense, state))
                # raw values
                raw_energy, raw_energy_stddev = compute_energy_pauli(
                    measurements, hamiltonian_jw
                )
                raw_edge_correlation = compute_edge_correlation(measurements)
                raw_parity = compute_parity(measurements)
                raw_number = compute_number(measurements)
                # measurement error corrected values
                (
                    mem_energy,
                    mem_energy_stddev,
                ) = compute_energy_pauli_measurement_corrected(quasis, hamiltonian_jw)
                mem_correlation = compute_edge_correlation_measurement_corrected(quasis)
                mem_parity = compute_parity_measurement_corrected(quasis)
                mem_number = compute_number_measurement_corrected(quasis)
                # add computed values to data storage objects
                energy_exact[occupied_orbitals].append(exact_energy + energy_shift)
                energy_raw[occupied_orbitals].append(
                    (raw_energy + energy_shift, raw_energy_stddev)
                )
                energy_mem[occupied_orbitals].append(
                    (
                        mem_energy + energy_shift,
                        mem_energy_stddev,
                    )
                )
                edge_correlation_exact[occupied_orbitals].append(exact_correlation)
                edge_correlation_raw[occupied_orbitals].append(raw_edge_correlation)
                edge_correlation_mem[occupied_orbitals].append(mem_correlation)
                parity_exact[occupied_orbitals].append(exact_parity)
                parity_raw[occupied_orbitals].append(raw_parity)
                parity_mem[occupied_orbitals].append(mem_parity)
                number_exact[occupied_orbitals].append(exact_number)
                number_raw[occupied_orbitals].append(raw_number)
                number_mem[occupied_orbitals].append(mem_number)

        yield AnalysisResultData("energy_exact", energy_exact)
        yield AnalysisResultData("energy_raw", energy_raw)
        yield AnalysisResultData("energy_mem", energy_mem)
        yield AnalysisResultData("edge_correlation_exact", edge_correlation_exact)
        yield AnalysisResultData("edge_correlation_raw", edge_correlation_raw)
        yield AnalysisResultData("edge_correlation_mem", edge_correlation_mem)
        yield AnalysisResultData("parity_exact", parity_exact)
        yield AnalysisResultData("parity_raw", parity_raw)
        yield AnalysisResultData("parity_mem", parity_mem)
        yield AnalysisResultData("number_exact", number_exact)
        yield AnalysisResultData("number_raw", number_raw)
        yield AnalysisResultData("number_mem", number_mem)
