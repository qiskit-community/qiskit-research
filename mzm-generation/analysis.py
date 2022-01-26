import os
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import mthree
import numpy as np
from matplotlib.figure import Figure
from qiskit import Aer
from qiskit_experiments.framework import (
    AnalysisResultData,
    BaseAnalysis,
    ExperimentData,
)
from qiskit_nature.circuit.library import FermionicGaussianState
from qiskit_nature.converters.second_quantization.qubit_converter import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper

from experiment import CircuitParameters, KitaevHamiltonianExperiment
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
    measurement_pauli_strings,
    number_op,
)


class KitaevHamiltonianAnalysis(BaseAnalysis):
    "Analyze Kitaev Hamiltonian experimental data."

    def _run_analysis(
        self, experiment_data: ExperimentData
    ) -> Tuple[List[AnalysisResultData], List[Figure]]:
        experiment = experiment_data.experiment
        experiment_id = experiment.experiment_id

        # put data into dictionary for easier handling
        data = {}
        for result in experiment_data.data():
            params = CircuitParameters(
                *(
                    tuple(p) if isinstance(p, list) else p
                    for p in result["metadata"]["params"]
                )
            )
            data[params] = result

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
