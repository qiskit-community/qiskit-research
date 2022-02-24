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
from typing import Dict, Iterable, List, Optional, Tuple, Union

import mthree
import numpy as np
from matplotlib.figure import Figure
from mthree.classes import QuasiDistribution
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
    _CovarianceDict,
    compute_correlation_matrix,
    compute_parity,
    counts_to_quasis,
    diagonalizing_bogoliubov_transform,
    edge_correlation_op,
    expectation_from_correlation_matrix,
    fidelity_witness,
    kitaev_hamiltonian,
    number_op,
    orbital_combinations,
    post_select_quasis,
    purify_idempotent_matrix,
)


class KitaevHamiltonianAnalysis(BaseAnalysis):
    "Analyze Kitaev Hamiltonian experimental data."

    def _run_analysis(
        self, experiment_data: ExperimentData
    ) -> Tuple[List[AnalysisResultData], List[Figure]]:
        # reconstruct experiment
        # TODO should be able to serialize and deserialize experiment automatically
        experiment_id = experiment_data.metadata["experiment_id"]
        backend = experiment_data.backend
        readout_calibration_date = experiment_data.metadata["readout_calibration_date"]
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
        dynamical_decoupling_sequences = experiment_data.metadata[
            "dynamical_decoupling_sequences"
        ]
        experiment = KitaevHamiltonianExperiment(
            experiment_id=experiment_id,
            backend=backend,
            readout_calibration_date=readout_calibration_date,
            qubits=qubits,
            tunneling_values=tunneling_values,
            superconducting_values=superconducting_values,
            chemical_potential_values=chemical_potential_values,
            occupied_orbitals_list=occupied_orbitals_list,
            dynamical_decoupling_sequences=dynamical_decoupling_sequences,
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
                dynamical_decoupling_sequence,
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
                dynamical_decoupling_sequence=dynamical_decoupling_sequence,
            )
            data[params] = result

        # load readout calibration
        mit = mthree.M3Mitigation()
        mit.cals_from_file(
            os.path.join(
                "data",
                "readout_calibration",
                backend.name(),
                f"{readout_calibration_date}.json",
            )
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

        # get simulation results
        yield from self._compute_simulation_results(
            tunneling, superconducting, experiment
        )

        # create data storage objects
        corr_raw = {}
        corr_mem = {}
        corr_ps = {}
        corr_pur = {}
        quasi_dists_raw = {}
        quasi_dists_mem = {}
        quasi_dists_ps = {}
        ps_removed_masses = {}

        # calculate results
        dd_sequences = [None]
        if experiment.dynamical_decoupling_sequences:
            dd_sequences += experiment.dynamical_decoupling_sequences
        for chemical_potential in experiment.chemical_potential_values:
            # diagonalize
            (transformation_matrix, _, _,) = diagonalizing_bogoliubov_transform(
                experiment.n_modes,
                tunneling=tunneling,
                superconducting=superconducting,
                chemical_potential=chemical_potential,
            )
            # compute parity
            W1 = transformation_matrix[:, : experiment.n_modes]
            W2 = transformation_matrix[:, experiment.n_modes :]
            full_transformation_matrix = np.block([[W1, W2], [W2.conj(), W1.conj()]])
            hamiltonian_parity = np.sign(
                np.real(np.linalg.det(full_transformation_matrix))
            )
            # compute quasis and correlation matrices
            for occupied_orbitals in experiment.occupied_orbitals_list:
                exact_parity = (-1) ** len(occupied_orbitals) * hamiltonian_parity
                for dd_sequence in dd_sequences:
                    quasis_raw = (
                        {}
                    )  # Dict[Tuple[Tuple[int, ...], str], QuasiDistribution]
                    quasis_mem = (
                        {}
                    )  # Dict[Tuple[Tuple[int, ...], str], QuasiDistribution]
                    quasis_ps = (
                        {}
                    )  # Dict[Tuple[Tuple[int, ...], str], QuasiDistribution]
                    ps_removed_mass = {}  # Dict[Tuple[Tuple[int, ...], str], float]
                    for permutation, label in experiment.measurement_labels():
                        params = CircuitParameters(
                            tunneling,
                            superconducting,
                            chemical_potential,
                            occupied_orbitals,
                            permutation,
                            label,
                            dynamical_decoupling_sequence=dd_sequence,
                        )
                        if params in data:
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
                                lambda bitstring: (-1)
                                ** sum(1 for b in bitstring if b == "1")
                                == exact_parity,
                            )
                            quasis_ps[permutation, label] = new_quasis
                            ps_removed_mass[permutation, label] = removed_mass
                    # save data
                    quasi_dists_raw[
                        tunneling,
                        superconducting,
                        chemical_potential,
                        occupied_orbitals,
                        dd_sequence,
                    ] = quasis_raw
                    quasi_dists_mem[
                        tunneling,
                        superconducting,
                        chemical_potential,
                        occupied_orbitals,
                        dd_sequence,
                    ] = quasis_mem
                    quasi_dists_ps[
                        tunneling,
                        superconducting,
                        chemical_potential,
                        occupied_orbitals,
                        dd_sequence,
                    ] = quasis_ps
                    ps_removed_masses[
                        tunneling,
                        superconducting,
                        chemical_potential,
                        occupied_orbitals,
                        dd_sequence,
                    ] = ps_removed_mass
                    # compute correlation matrices
                    corr_raw[
                        tunneling,
                        superconducting,
                        chemical_potential,
                        occupied_orbitals,
                        dd_sequence,
                    ] = compute_correlation_matrix(quasis_raw, experiment)
                    corr_mem[
                        tunneling,
                        superconducting,
                        chemical_potential,
                        occupied_orbitals,
                        dd_sequence,
                    ] = compute_correlation_matrix(quasis_mem, experiment)
                    corr_mat_ps, cov_ps = compute_correlation_matrix(
                        quasis_ps, experiment
                    )
                    corr_ps[
                        tunneling,
                        superconducting,
                        chemical_potential,
                        occupied_orbitals,
                        dd_sequence,
                    ] = (corr_mat_ps, cov_ps)
                    corr_pur[
                        tunneling,
                        superconducting,
                        chemical_potential,
                        occupied_orbitals,
                        dd_sequence,
                    ] = (purify_idempotent_matrix(corr_mat_ps), cov_ps)

        for dd_sequence in dd_sequences:
            yield from self._compute_fidelity_witness(
                "raw" + (f"_{dd_sequence}" if dd_sequence else ""),
                corr_raw,
                experiment.n_modes,
                tunneling,
                superconducting,
                experiment.chemical_potential_values,
                experiment.occupied_orbitals_list,
                dd_sequence,
            )
            yield from self._compute_fidelity_witness(
                "mem" + (f"_{dd_sequence}" if dd_sequence else ""),
                corr_mem,
                experiment.n_modes,
                tunneling,
                superconducting,
                experiment.chemical_potential_values,
                experiment.occupied_orbitals_list,
                dd_sequence,
            )
            yield from self._compute_fidelity_witness(
                "ps" + (f"_{dd_sequence}" if dd_sequence else ""),
                corr_ps,
                experiment.n_modes,
                tunneling,
                superconducting,
                experiment.chemical_potential_values,
                experiment.occupied_orbitals_list,
                dd_sequence,
            )
            yield from self._compute_fidelity_witness(
                "pur" + (f"_{dd_sequence}" if dd_sequence else ""),
                corr_pur,
                experiment.n_modes,
                tunneling,
                superconducting,
                experiment.chemical_potential_values,
                experiment.occupied_orbitals_list,
                dd_sequence,
            )
            yield from self._compute_energy(
                "raw" + (f"_{dd_sequence}" if dd_sequence else ""),
                corr_raw,
                experiment.n_modes,
                tunneling,
                superconducting,
                experiment.chemical_potential_values,
                experiment.occupied_orbitals_list,
                dd_sequence,
            )
            yield from self._compute_energy(
                "mem" + (f"_{dd_sequence}" if dd_sequence else ""),
                corr_mem,
                experiment.n_modes,
                tunneling,
                superconducting,
                experiment.chemical_potential_values,
                experiment.occupied_orbitals_list,
                dd_sequence,
            )
            yield from self._compute_energy(
                "ps" + (f"_{dd_sequence}" if dd_sequence else ""),
                corr_ps,
                experiment.n_modes,
                tunneling,
                superconducting,
                experiment.chemical_potential_values,
                experiment.occupied_orbitals_list,
                dd_sequence,
            )
            yield from self._compute_energy(
                "pur" + (f"_{dd_sequence}" if dd_sequence else ""),
                corr_pur,
                experiment.n_modes,
                tunneling,
                superconducting,
                experiment.chemical_potential_values,
                experiment.occupied_orbitals_list,
                dd_sequence,
            )
            yield from self._compute_edge_correlation(
                "raw" + (f"_{dd_sequence}" if dd_sequence else ""),
                corr_raw,
                experiment.n_modes,
                tunneling,
                superconducting,
                experiment.chemical_potential_values,
                experiment.occupied_orbitals_list,
                dd_sequence,
            )
            yield from self._compute_edge_correlation(
                "mem" + (f"_{dd_sequence}" if dd_sequence else ""),
                corr_mem,
                experiment.n_modes,
                tunneling,
                superconducting,
                experiment.chemical_potential_values,
                experiment.occupied_orbitals_list,
                dd_sequence,
            )
            yield from self._compute_edge_correlation(
                "ps" + (f"_{dd_sequence}" if dd_sequence else ""),
                corr_ps,
                experiment.n_modes,
                tunneling,
                superconducting,
                experiment.chemical_potential_values,
                experiment.occupied_orbitals_list,
                dd_sequence,
            )
            yield from self._compute_edge_correlation(
                "pur" + (f"_{dd_sequence}" if dd_sequence else ""),
                corr_pur,
                experiment.n_modes,
                tunneling,
                superconducting,
                experiment.chemical_potential_values,
                experiment.occupied_orbitals_list,
                dd_sequence,
            )
            yield from self._compute_number(
                "raw" + (f"_{dd_sequence}" if dd_sequence else ""),
                corr_raw,
                experiment.n_modes,
                tunneling,
                superconducting,
                experiment.chemical_potential_values,
                experiment.occupied_orbitals_list,
                dd_sequence,
            )
            yield from self._compute_number(
                "mem" + (f"_{dd_sequence}" if dd_sequence else ""),
                corr_mem,
                experiment.n_modes,
                tunneling,
                superconducting,
                experiment.chemical_potential_values,
                experiment.occupied_orbitals_list,
                dd_sequence,
            )
            yield from self._compute_number(
                "ps" + (f"_{dd_sequence}" if dd_sequence else ""),
                corr_ps,
                experiment.n_modes,
                tunneling,
                superconducting,
                experiment.chemical_potential_values,
                experiment.occupied_orbitals_list,
                dd_sequence,
            )
            yield from self._compute_number(
                "pur" + (f"_{dd_sequence}" if dd_sequence else ""),
                corr_pur,
                experiment.n_modes,
                tunneling,
                superconducting,
                experiment.chemical_potential_values,
                experiment.occupied_orbitals_list,
                dd_sequence,
            )
            yield from self._compute_parity(
                "raw" + (f"_{dd_sequence}" if dd_sequence else ""),
                quasi_dists_raw,
                tunneling,
                superconducting,
                experiment.chemical_potential_values,
                experiment.occupied_orbitals_list,
                dd_sequence,
            )
            yield from self._compute_parity(
                "mem" + (f"_{dd_sequence}" if dd_sequence else ""),
                quasi_dists_mem,
                tunneling,
                superconducting,
                experiment.chemical_potential_values,
                experiment.occupied_orbitals_list,
                dd_sequence,
            )
            yield from self._compute_parity(
                "ps" + (f"_{dd_sequence}" if dd_sequence else ""),
                quasi_dists_ps,
                tunneling,
                superconducting,
                experiment.chemical_potential_values,
                experiment.occupied_orbitals_list,
                dd_sequence,
            )

    def _compute_simulation_results(
        self,
        tunneling: float,
        superconducting: Union[float, complex],
        experiment: KitaevHamiltonianExperiment,
    ) -> Iterable[AnalysisResultData]:
        # set chemical potential values to the experiment range but with fixed resolution
        chemical_potential_values = np.linspace(
            experiment.chemical_potential_values[0],
            experiment.chemical_potential_values[-1],
            num=50,
        )

        # construct operators
        edge_correlation = edge_correlation_op(experiment.n_modes)
        number = number_op(experiment.n_modes)

        # create data storage objects
        energy_exact = defaultdict(list)  # Dict[Tuple[int, ...], List[float]]
        edge_correlation_exact = defaultdict(list)  # Dict[Tuple[int, ...], List[float]]
        parity_exact = defaultdict(list)  # Dict[Tuple[int, ...], List[float]]
        number_exact = defaultdict(list)  # Dict[Tuple[int, ...], List[float]]

        for chemical_potential in chemical_potential_values:
            # diagonalize Hamiltonian
            (
                transformation_matrix,
                orbital_energies,
                constant,
            ) = diagonalizing_bogoliubov_transform(
                experiment.n_modes,
                tunneling=tunneling,
                superconducting=superconducting,
                chemical_potential=chemical_potential,
            )
            # compute parity
            W1 = transformation_matrix[:, : experiment.n_modes]
            W2 = transformation_matrix[:, experiment.n_modes :]
            full_transformation_matrix = np.block([[W1, W2], [W2.conj(), W1.conj()]])
            hamiltonian_parity = np.sign(
                np.real(np.linalg.det(full_transformation_matrix))
            )
            # compute results
            for occupied_orbitals in experiment.occupied_orbitals_list:
                # compute exact correlation matrix
                occupation = np.zeros(experiment.n_modes)
                occupation[list(occupied_orbitals)] = 1.0
                corr_diag = np.diag(np.concatenate([occupation, 1 - occupation]))
                corr_exact = (
                    full_transformation_matrix.T.conj()
                    @ corr_diag
                    @ full_transformation_matrix
                )
                # exact values
                exact_energy = (
                    np.sum(orbital_energies[list(occupied_orbitals)]) + constant
                )
                exact_edge_correlation, _ = np.real(
                    expectation_from_correlation_matrix(edge_correlation, corr_exact)
                )
                exact_parity = (-1) ** len(occupied_orbitals) * hamiltonian_parity
                exact_number = np.real(
                    np.sum(np.diag(corr_exact)[: experiment.n_modes])
                )
                # add computed values to data storage objects
                energy_exact[occupied_orbitals].append(exact_energy)
                edge_correlation_exact[occupied_orbitals].append(exact_edge_correlation)
                parity_exact[occupied_orbitals].append(exact_parity)
                number_exact[occupied_orbitals].append(exact_number)

        def zip_dict(d):
            return {k: (np.array(v), chemical_potential_values) for k, v in d.items()}

        yield AnalysisResultData("energy_exact", zip_dict(energy_exact))
        yield AnalysisResultData(
            "edge_correlation_exact", zip_dict(edge_correlation_exact)
        )
        yield AnalysisResultData("parity_exact", zip_dict(parity_exact))
        yield AnalysisResultData("number_exact", zip_dict(number_exact))

        # BdG
        occupied_orbitals_set = set(experiment.occupied_orbitals_list)
        combs = list(orbital_combinations(experiment.n_modes))
        threshold = -1
        for i in range(0, len(combs), 2):
            if (
                combs[i] not in occupied_orbitals_set
                or combs[i + 1] not in occupied_orbitals_set
            ):
                break
            threshold += 1
        if threshold >= 0:
            bdg_energy = np.zeros((2 * threshold, len(chemical_potential_values)))
            low = np.array(energy_exact[()])
            high = np.array(energy_exact[tuple(range(experiment.n_modes))])
            for i in range(threshold):
                particle = np.array(energy_exact[combs[2 * i + 2]])
                hole = np.array(energy_exact[combs[2 * i + 3]])
                bdg_energy[i] = low - particle
                bdg_energy[threshold + i] = high - hole
            yield AnalysisResultData(
                f"bdg_energy_exact", (bdg_energy, chemical_potential_values)
            )

    def _compute_fidelity_witness(
        self,
        label: str,
        corr: Dict[
            Tuple[int, float, Union[float, complex], Tuple[int, ...], Optional[str]],
            Tuple[np.ndarray, _CovarianceDict],
        ],
        n_modes: int,
        tunneling: float,
        superconducting: Union[float, complex],
        chemical_potential_values: Iterable[float],
        occupied_orbitals_list: Iterable[Tuple[int, ...]],
        dynamical_decoupling_sequence: Optional[str],
    ) -> Iterable[AnalysisResultData]:
        data = defaultdict(list)  # Dict[Tuple[int, ...], List[Tuple[float, float]]]
        for chemical_potential in chemical_potential_values:
            # diagonalize Hamiltonian
            (transformation_matrix, _, _,) = diagonalizing_bogoliubov_transform(
                n_modes,
                tunneling=tunneling,
                superconducting=superconducting,
                chemical_potential=chemical_potential,
            )
            W1 = transformation_matrix[:, :n_modes]
            W2 = transformation_matrix[:, n_modes:]
            full_transformation_matrix = np.block([[W1, W2], [W2.conj(), W1.conj()]])
            for occupied_orbitals in occupied_orbitals_list:
                # compute exact correlation matrix
                occupation = np.zeros(n_modes)
                occupation[list(occupied_orbitals)] = 1.0
                corr_diag = np.diag(np.concatenate([occupation, 1 - occupation]))
                corr_exact = (
                    full_transformation_matrix.T.conj()
                    @ corr_diag
                    @ full_transformation_matrix
                )
                corr_mat, cov = corr[
                    tunneling,
                    superconducting,
                    chemical_potential,
                    occupied_orbitals,
                    dynamical_decoupling_sequence,
                ]
                fidelity_wit, stddev = fidelity_witness(corr_mat, corr_exact, cov)
                data[occupied_orbitals].append((fidelity_wit, stddev))
        data_zipped = {k: tuple(np.array(a) for a in zip(*v)) for k, v in data.items()}
        yield AnalysisResultData(f"fidelity_witness_{label}", data_zipped)

        fidelity_witness_avg = np.zeros(len(chemical_potential_values))
        fidelity_witness_stddev = np.zeros(len(chemical_potential_values))
        for occupied_orbitals in occupied_orbitals_list:
            values, stddevs = data_zipped[occupied_orbitals]
            fidelity_witness_avg += np.array(values)
            fidelity_witness_stddev += np.array(stddevs) ** 2
        fidelity_witness_avg /= len(occupied_orbitals_list)
        fidelity_witness_stddev = np.sqrt(fidelity_witness_stddev) / len(
            occupied_orbitals_list
        )
        yield AnalysisResultData(
            f"fidelity_witness_avg_{label}",
            (fidelity_witness_avg, fidelity_witness_stddev),
        )

    def _compute_energy(
        self,
        label: str,
        corr: Dict[
            Tuple[int, float, Union[float, complex], Tuple[int, ...], Optional[str]],
            Tuple[np.ndarray, _CovarianceDict],
        ],
        n_modes: int,
        tunneling: float,
        superconducting: Union[float, complex],
        chemical_potential_values: Iterable[float],
        occupied_orbitals_list: Iterable[Tuple[int, ...]],
        dynamical_decoupling_sequence: Optional[str],
    ) -> Iterable[AnalysisResultData]:
        energy_exact = defaultdict(list)  # Dict[Tuple[int, ...], List[float]]
        data = defaultdict(list)  # Dict[Tuple[int, ...], List[Tuple[float, float]]]
        for chemical_potential in chemical_potential_values:
            # create Hamiltonian
            hamiltonian_quad = kitaev_hamiltonian(
                n_modes,
                tunneling=tunneling,
                superconducting=superconducting,
                chemical_potential=chemical_potential,
            )
            # diagonalize
            (_, orbital_energies, constant,) = diagonalizing_bogoliubov_transform(
                n_modes,
                tunneling=tunneling,
                superconducting=superconducting,
                chemical_potential=chemical_potential,
            )
            for occupied_orbitals in occupied_orbitals_list:
                exact_energy = (
                    np.sum(orbital_energies[list(occupied_orbitals)]) + constant
                )
                energy_exact[occupied_orbitals].append(exact_energy)

                corr_mat, cov = corr[
                    tunneling,
                    superconducting,
                    chemical_potential,
                    occupied_orbitals,
                    dynamical_decoupling_sequence,
                ]
                energy, stddevs = np.real(
                    expectation_from_correlation_matrix(hamiltonian_quad, corr_mat, cov)
                )
                data[occupied_orbitals].append((energy, stddevs))
        data_zipped = {k: tuple(np.array(a) for a in zip(*v)) for k, v in data.items()}
        yield AnalysisResultData(f"energy_{label}", data_zipped)

        # error
        error = np.zeros(len(chemical_potential_values))
        error_stddev = np.zeros(len(chemical_potential_values))
        for occupied_orbitals in occupied_orbitals_list:
            exact = np.array(energy_exact[occupied_orbitals])
            values, stddevs = data_zipped[occupied_orbitals]
            values = np.array(values)
            error += np.abs(values - exact)
            error_stddev += np.array(stddevs) ** 2
        error /= len(occupied_orbitals_list)
        error_stddev = np.sqrt(error_stddev) / len(occupied_orbitals_list)
        yield AnalysisResultData(f"energy_error_{label}", (error, error_stddev))

        # BdG
        occupied_orbitals_set = set(occupied_orbitals_list)
        combs = list(orbital_combinations(n_modes))
        threshold = -1
        for i in range(0, len(combs), 2):
            if (
                combs[i] not in occupied_orbitals_set
                or combs[i + 1] not in occupied_orbitals_set
            ):
                break
            threshold += 1
        if threshold >= 0:
            bdg_energy = np.zeros((2 * threshold, len(chemical_potential_values)))
            bdg_stddev = np.zeros((2 * threshold, len(chemical_potential_values)))
            low, low_stddev = data_zipped[()]
            high, high_stddev = data_zipped[tuple(range(n_modes))]
            for i in range(threshold):
                particle, particle_stddev = data_zipped[combs[2 * i + 2]]
                hole, hole_stddev = data_zipped[combs[2 * i + 3]]
                bdg_energy[i] = low - particle
                bdg_energy[threshold + i] = high - hole
                bdg_stddev[i] = low_stddev ** 2 + particle_stddev ** 2
                bdg_stddev[threshold + i] = high_stddev ** 2 + hole_stddev ** 2
            bdg_stddev = np.sqrt(bdg_stddev)
            yield AnalysisResultData(f"bdg_energy_{label}", (bdg_energy, bdg_stddev))

    def _compute_edge_correlation(
        self,
        label: str,
        corr: Dict[
            Tuple[int, float, Union[float, complex], Tuple[int, ...], Optional[str]],
            Tuple[np.ndarray, _CovarianceDict],
        ],
        n_modes: int,
        tunneling: float,
        superconducting: Union[float, complex],
        chemical_potential_values: Iterable[float],
        occupied_orbitals_list: Iterable[Tuple[int, ...]],
        dynamical_decoupling_sequence: Optional[str],
    ) -> Iterable[AnalysisResultData]:
        edge_correlation = edge_correlation_op(n_modes)
        data = defaultdict(list)  # Dict[Tuple[int, ...], List[Tuple[float, float]]]
        for chemical_potential in chemical_potential_values:
            for occupied_orbitals in occupied_orbitals_list:
                corr_mat, cov = corr[
                    tunneling,
                    superconducting,
                    chemical_potential,
                    occupied_orbitals,
                    dynamical_decoupling_sequence,
                ]
                edge_correlation_val, stddev = np.real(
                    expectation_from_correlation_matrix(edge_correlation, corr_mat, cov)
                )
                data[occupied_orbitals].append((edge_correlation_val, stddev))
        data_zipped = {k: tuple(np.array(a) for a in zip(*v)) for k, v in data.items()}
        yield AnalysisResultData(f"edge_correlation_{label}", data_zipped)

    def _compute_number(
        self,
        label: str,
        corr: Dict[
            Tuple[int, float, Union[float, complex], Tuple[int, ...], Optional[str]],
            Tuple[np.ndarray, _CovarianceDict],
        ],
        n_modes: int,
        tunneling: float,
        superconducting: Union[float, complex],
        chemical_potential_values: Iterable[float],
        occupied_orbitals_list: Iterable[Tuple[int, ...]],
        dynamical_decoupling_sequence: Optional[str],
    ) -> Iterable[AnalysisResultData]:
        number = number_op(n_modes)
        data = defaultdict(list)  # Dict[Tuple[int, ...], List[Tuple[float, float]]]
        for chemical_potential in chemical_potential_values:
            for occupied_orbitals in occupied_orbitals_list:
                corr_mat, cov = corr[
                    tunneling,
                    superconducting,
                    chemical_potential,
                    occupied_orbitals,
                    dynamical_decoupling_sequence,
                ]
                number_val, stddev = np.real(
                    expectation_from_correlation_matrix(number, corr_mat, cov)
                )
                data[occupied_orbitals].append((number_val, stddev))
        data_zipped = {k: tuple(np.array(a) for a in zip(*v)) for k, v in data.items()}
        yield AnalysisResultData(f"number_{label}", data_zipped)

    def _compute_parity(
        self,
        label: str,
        quasi_dists: Dict[
            Tuple[int, float, Union[float, complex], Tuple[int, ...], Optional[str]],
            Dict[Tuple[Tuple[int, ...], str], QuasiDistribution],
        ],
        tunneling: float,
        superconducting: Union[float, complex],
        chemical_potential_values: Iterable[float],
        occupied_orbitals_list: Iterable[Tuple[int, ...]],
        dynamical_decoupling_sequence: Optional[str],
    ) -> Iterable[AnalysisResultData]:
        data = defaultdict(list)  # Dict[Tuple[int, ...], List[Tuple[float, float]]]
        for chemical_potential in chemical_potential_values:
            for occupied_orbitals in occupied_orbitals_list:
                quasis = quasi_dists[
                    tunneling,
                    superconducting,
                    chemical_potential,
                    occupied_orbitals,
                    dynamical_decoupling_sequence,
                ]
                parity, stddev = compute_parity(quasis)
                data[occupied_orbitals].append((parity, stddev))
        data_zipped = {k: tuple(np.array(a) for a in zip(*v)) for k, v in data.items()}
        yield AnalysisResultData(f"parity_{label}", data_zipped)
