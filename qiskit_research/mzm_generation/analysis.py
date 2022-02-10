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

import os
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, Iterable, List, Tuple

import mthree
import numpy as np
from matplotlib.figure import Figure
from qiskit_experiments.framework import (
    AnalysisResultData,
    BaseAnalysis,
    ExperimentData,
)
from qiskit_research.mzm_generation.experiment import (
    CircuitParameters,
    KitaevHamiltonianExperiment,
)
from qiskit_research.mzm_generation.utils import (
    compute_correlation_matrix,
    compute_parity,
    counts_to_quasis,
    edge_correlation_op,
    expectation_from_correlation_matrix,
    kitaev_hamiltonian,
    number_op,
    post_select_quasis,
    purify_idempotent_matrix,
)

if TYPE_CHECKING:
    from mthree.classes import QuasiDistribution


class KitaevHamiltonianAnalysis(BaseAnalysis):
    "Analyze Kitaev Hamiltonian experimental data."

    def _run_analysis(
        self, experiment_data: ExperimentData
    ) -> Tuple[List[AnalysisResultData], List[Figure]]:
        # reconstruct experiment
        experiment_id = experiment_data.metadata["experiment_id"]
        qubits = experiment_data.metadata["qubits"]
        tunneling_values = experiment_data.metadata["tunneling_values"]
        superconducting_values = experiment_data.metadata["superconducting_values"]
        chemical_potential_values = experiment_data.metadata[
            "chemical_potential_values"
        ]
        occupied_orbitals_list = [
            tuple(occupied_orbitals)
            for occupied_orbitals in experiment_data.metadata["occupied_orbitals_list"]
        ]
        experiment = KitaevHamiltonianExperiment(
            experiment_id=experiment_id,
            qubits=qubits,
            tunneling_values=tunneling_values,
            superconducting_values=superconducting_values,
            chemical_potential_values=chemical_potential_values,
            occupied_orbitals_list=occupied_orbitals_list,
        )

        # put data into dictionary for easier handling
        data = {}
        for result in experiment_data.data():
            (
                tunneling,
                superconducting,
                chemical_potential,
                occupied_orbitals,
                permutation,
                measurement_label,
            ) = result["metadata"]["params"]
            params = CircuitParameters(
                tunneling=tunneling,
                superconducting=superconducting
                if isinstance(superconducting, float)
                else complex(*superconducting),
                chemical_potential=chemical_potential,
                occupied_orbitals=tuple(occupied_orbitals),
                permutation=tuple(permutation),
                measurement_label=measurement_label,
            )
            data[params] = result

        # load readout calibration
        mit = mthree.M3Mitigation()
        mit.cals_from_file(
            os.path.join("data", experiment.experiment_id, f"readout_calibration.json")
        )

        # get results
        results = list(self._compute_analysis_results(experiment, data, mit))
        return results, []

    def _compute_analysis_results(
        self,
        experiment: KitaevHamiltonianExperiment,
        data: Dict[CircuitParameters, Dict],
        mit: mthree.M3Mitigation,
    ) -> Iterable[AnalysisResultData]:
        # fix tunneling and superconducting
        tunneling = -1.0
        superconducting = 1.0

        # construct operators
        edge_correlation = edge_correlation_op(experiment.n_modes)
        number = number_op(experiment.n_modes)

        # create data storage objects
        energy_exact = defaultdict(list)  # Dict[Tuple[int, ...], List[float]]
        energy_raw = defaultdict(
            list
        )  # Dict[Tuple[int, ...], List[Tuple[float, float]]]
        energy_mem = defaultdict(
            list
        )  # Dict[Tuple[int, ...], List[Tuple[float, float]]]
        energy_ps = defaultdict(
            list
        )  # Dict[Tuple[int, ...], List[Tuple[float, float]]]
        energy_pur = defaultdict(
            list
        )  # Dict[Tuple[int, ...], List[Tuple[float, float]]]
        edge_correlation_exact = defaultdict(list)  # Dict[Tuple[int, ...], List[float]]
        edge_correlation_raw = defaultdict(
            list
        )  # Dict[Tuple[int, ...], List[Tuple[float, float]]]
        edge_correlation_mem = defaultdict(
            list
        )  # Dict[Tuple[int, ...], List[Tuple[float, float]]]
        edge_correlation_ps = defaultdict(
            list
        )  # Dict[Tuple[int, ...], List[Tuple[float, float]]]
        edge_correlation_pur = defaultdict(
            list
        )  # Dict[Tuple[int, ...], List[Tuple[float, float]]]
        parity_exact = defaultdict(list)  # Dict[Tuple[int, ...], List[float]]
        parity_raw = defaultdict(
            list
        )  # Dict[Tuple[int, ...], List[Tuple[float, float]]]
        parity_mem = defaultdict(
            list
        )  # Dict[Tuple[int, ...], List[Tuple[float, float]]]
        parity_ps = defaultdict(
            list
        )  # Dict[Tuple[int, ...], List[Tuple[float, float]]]
        parity_pur = defaultdict(
            list
        )  # Dict[Tuple[int, ...], List[Tuple[float, float]]]
        number_exact = defaultdict(list)  # Dict[Tuple[int, ...], List[float]]
        number_raw = defaultdict(
            list
        )  # Dict[Tuple[int, ...], List[Tuple[float, float]]]
        number_mem = defaultdict(
            list
        )  # Dict[Tuple[int, ...], List[Tuple[float, float]]]
        number_ps = defaultdict(
            list
        )  # Dict[Tuple[int, ...], List[Tuple[float, float]]]
        number_pur = defaultdict(
            list
        )  # Dict[Tuple[int, ...], List[Tuple[float, float]]]

        # calculate results
        for chemical_potential in experiment.chemical_potential_values:
            # create Hamiltonian
            hamiltonian_quad = kitaev_hamiltonian(
                experiment.n_modes,
                tunneling=tunneling,
                superconducting=superconducting,
                chemical_potential=chemical_potential,
            )
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
            full_transformation_matrix = np.block([[W1, W2], [W2.conj(), W1.conj()]])
            hamiltonian_parity = np.sign(
                np.real(np.linalg.det(full_transformation_matrix))
            )
            # compute results
            for occupied_orbitals in experiment.occupied_orbitals_list:
                exact_parity = (-1) ** len(occupied_orbitals) * hamiltonian_parity
                quasis_raw = {}  # Dict[Tuple[str, str], QuasiDistribution]
                quasis_mem = {}  # Dict[Tuple[str, str], QuasiDistribution]
                quasis_ps = {}  # Dict[Tuple[str, str], QuasiDistribution]
                ps_removed_mass = {}  # Dict[Tuple[str, str], float]
                for permutation, label in experiment.measurement_labels():
                    params = CircuitParameters(
                        tunneling,
                        superconducting,
                        chemical_potential,
                        occupied_orbitals,
                        permutation,
                        label,
                    )
                    counts = data[params]["counts"]
                    # raw quasis
                    quasis_raw[permutation, label] = counts_to_quasis(counts)
                    # measurement error mitigation
                    quasis_mem[permutation, label] = mit.apply_correction(
                        counts,
                        experiment.qubits,
                        return_mitigation_overhead=True,
                    )
                    # post-selection
                    new_quasis, removed_mass = post_select_quasis(
                        quasis_mem[permutation, label],
                        lambda bitstring: (-1) ** sum(1 for b in bitstring if b == "1")
                        == exact_parity,
                    )
                    quasis_ps[permutation, label] = new_quasis
                    ps_removed_mass[permutation, label] = removed_mass

                # compute exact correlation matrix
                occupation = np.zeros(experiment.n_modes)
                occupation[list(occupied_orbitals)] = 1.0
                corr_diag = np.diag(np.concatenate([occupation, 1 - occupation]))
                corr_exact = (
                    full_transformation_matrix.T.conj()
                    @ corr_diag
                    @ full_transformation_matrix
                )
                # compute correlation matrices
                corr_raw, cov_raw = compute_correlation_matrix(quasis_raw, experiment)
                corr_mem, cov_mem = compute_correlation_matrix(quasis_mem, experiment)
                corr_ps, cov_ps = compute_correlation_matrix(quasis_ps, experiment)
                corr_pur = purify_idempotent_matrix(corr_ps)
                cov_pur = cov_ps
                # exact values
                exact_energy = (
                    np.sum(orbital_energies[list(occupied_orbitals)]) + constant
                )
                exact_edge_correlation, _ = np.real(
                    expectation_from_correlation_matrix(edge_correlation, corr_exact)
                )
                exact_number = np.real(
                    np.sum(np.diag(corr_exact)[: experiment.n_modes])
                )
                # raw values
                raw_energy, raw_energy_stddev = np.real(
                    expectation_from_correlation_matrix(
                        hamiltonian_quad, corr_raw, cov_raw
                    )
                )
                raw_edge_correlation, raw_edge_correlation_stddev = np.real(
                    expectation_from_correlation_matrix(
                        edge_correlation, corr_raw, cov_raw
                    )
                )
                raw_parity, raw_parity_stddev = compute_parity(quasis_raw)
                raw_number, raw_number_stddev = np.real(
                    expectation_from_correlation_matrix(number, corr_raw, cov_raw)
                )
                # measurement error corrected values
                mem_energy, mem_energy_stddev = np.real(
                    expectation_from_correlation_matrix(
                        hamiltonian_quad, corr_mem, cov_mem
                    )
                )
                mem_edge_correlation, mem_edge_correlation_stddev = np.real(
                    expectation_from_correlation_matrix(
                        edge_correlation, corr_mem, cov_mem
                    )
                )
                mem_parity, mem_parity_stddev = compute_parity(quasis_mem)
                mem_number, mem_number_stddev = np.real(
                    expectation_from_correlation_matrix(number, corr_mem, cov_mem)
                )
                # post-selected values
                ps_energy, ps_energy_stddev = np.real(
                    expectation_from_correlation_matrix(
                        hamiltonian_quad, corr_ps, cov_ps
                    )
                )
                ps_edge_correlation, ps_edge_correlation_stddev = np.real(
                    expectation_from_correlation_matrix(
                        edge_correlation, corr_ps, cov_ps
                    )
                )
                # purified values
                pur_energy, pur_energy_stddev = np.real(
                    expectation_from_correlation_matrix(
                        hamiltonian_quad, corr_pur, cov_pur
                    )
                )
                pur_edge_correlation, pur_edge_correlation_stddev = np.real(
                    expectation_from_correlation_matrix(
                        edge_correlation, corr_pur, cov_pur
                    )
                )
                # add computed values to data storage objects
                energy_exact[occupied_orbitals].append(exact_energy + energy_shift)
                energy_raw[occupied_orbitals].append(
                    (
                        raw_energy + energy_shift,
                        raw_energy_stddev,
                    )
                )
                energy_mem[occupied_orbitals].append(
                    (
                        mem_energy + energy_shift,
                        mem_energy_stddev,
                    )
                )
                energy_ps[occupied_orbitals].append(
                    (ps_energy + energy_shift, ps_energy_stddev)
                )
                energy_pur[occupied_orbitals].append(
                    (pur_energy + energy_shift, pur_energy_stddev)
                )
                edge_correlation_exact[occupied_orbitals].append(exact_edge_correlation)
                edge_correlation_raw[occupied_orbitals].append(
                    (raw_edge_correlation, raw_edge_correlation_stddev)
                )
                edge_correlation_mem[occupied_orbitals].append(
                    (mem_edge_correlation, mem_edge_correlation_stddev)
                )
                edge_correlation_ps[occupied_orbitals].append(
                    (ps_edge_correlation, ps_edge_correlation_stddev)
                )
                edge_correlation_pur[occupied_orbitals].append(
                    (pur_edge_correlation, pur_edge_correlation_stddev)
                )
                parity_exact[occupied_orbitals].append(exact_parity)
                parity_raw[occupied_orbitals].append((raw_parity, raw_parity_stddev))
                parity_mem[occupied_orbitals].append((mem_parity, mem_parity_stddev))
                number_exact[occupied_orbitals].append(exact_number)
                number_raw[occupied_orbitals].append((raw_number, raw_number_stddev))
                number_mem[occupied_orbitals].append((mem_number, mem_number_stddev))

        def zip_dict(d):
            val = next(iter(d.values()))
            if isinstance(val[0], Iterable):
                return {k: tuple(np.array(a) for a in zip(*v)) for k, v in d.items()}
            return {k: np.array(v) for k, v in d.items()}

        yield AnalysisResultData("energy_exact", zip_dict(energy_exact))
        yield AnalysisResultData("energy_raw", zip_dict(energy_raw))
        yield AnalysisResultData("energy_mem", zip_dict(energy_mem))
        yield AnalysisResultData("energy_ps", zip_dict(energy_ps))
        yield AnalysisResultData("energy_pur", zip_dict(energy_pur))
        yield AnalysisResultData(
            "edge_correlation_exact", zip_dict(edge_correlation_exact)
        )
        yield AnalysisResultData("edge_correlation_raw", zip_dict(edge_correlation_raw))
        yield AnalysisResultData("edge_correlation_mem", zip_dict(edge_correlation_mem))
        yield AnalysisResultData("edge_correlation_ps", zip_dict(edge_correlation_ps))
        yield AnalysisResultData("edge_correlation_pur", zip_dict(edge_correlation_pur))
        yield AnalysisResultData("parity_exact", zip_dict(parity_exact))
        yield AnalysisResultData("parity_raw", zip_dict(parity_raw))
        yield AnalysisResultData("parity_mem", zip_dict(parity_mem))
        yield AnalysisResultData("number_exact", zip_dict(number_exact))
        yield AnalysisResultData("number_raw", zip_dict(number_raw))
        yield AnalysisResultData("number_mem", zip_dict(number_mem))

        # error analysis
        energy_error_raw = np.zeros(len(experiment.chemical_potential_values))
        energy_error_stddev_raw = np.zeros(len(experiment.chemical_potential_values))
        energy_error_mem = np.zeros(len(experiment.chemical_potential_values))
        energy_error_stddev_mem = np.zeros(len(experiment.chemical_potential_values))
        energy_error_ps = np.zeros(len(experiment.chemical_potential_values))
        energy_error_stddev_ps = np.zeros(len(experiment.chemical_potential_values))
        energy_error_pur = np.zeros(len(experiment.chemical_potential_values))
        energy_error_stddev_pur = np.zeros(len(experiment.chemical_potential_values))

        for occupied_orbitals in experiment.occupied_orbitals_list:
            exact = np.array(energy_exact[occupied_orbitals])

            raw, raw_stddev = zip(*energy_raw[occupied_orbitals])
            raw = np.array(raw)
            energy_error_raw += np.abs(raw - exact)
            energy_error_stddev_raw += np.array(raw_stddev) ** 2

            mem, mem_stddev = zip(*energy_mem[occupied_orbitals])
            mem = np.array(mem)
            energy_error_mem += np.abs(mem - exact)
            energy_error_stddev_mem += np.array(mem_stddev) ** 2

            ps, ps_stddev = zip(*energy_ps[occupied_orbitals])
            ps = np.array(ps)
            energy_error_ps += np.abs(ps - exact)
            energy_error_stddev_ps += np.array(ps_stddev) ** 2

            pur, pur_stddev = zip(*energy_pur[occupied_orbitals])
            pur = np.array(pur)
            energy_error_pur += np.abs(pur - exact)
            energy_error_stddev_pur += np.array(pur_stddev) ** 2

        energy_error_raw /= len(experiment.occupied_orbitals_list)
        energy_error_stddev_raw = np.sqrt(energy_error_stddev_raw) / len(
            experiment.occupied_orbitals_list
        )
        energy_error_mem /= len(experiment.occupied_orbitals_list)
        energy_error_stddev_mem = np.sqrt(energy_error_stddev_mem) / len(
            experiment.occupied_orbitals_list
        )
        energy_error_ps /= len(experiment.occupied_orbitals_list)
        energy_error_stddev_ps = np.sqrt(energy_error_stddev_ps) / len(
            experiment.occupied_orbitals_list
        )
        energy_error_pur /= len(experiment.occupied_orbitals_list)
        energy_error_stddev_pur = np.sqrt(energy_error_stddev_pur) / len(
            experiment.occupied_orbitals_list
        )

        yield AnalysisResultData(
            "energy_error_raw", (energy_error_raw, energy_error_stddev_raw)
        )
        yield AnalysisResultData(
            "energy_error_mem", (energy_error_mem, energy_error_stddev_mem)
        )
        yield AnalysisResultData(
            "energy_error_ps", (energy_error_ps, energy_error_stddev_ps)
        )
        yield AnalysisResultData(
            "energy_error_pur", (energy_error_pur, energy_error_stddev_pur)
        )
