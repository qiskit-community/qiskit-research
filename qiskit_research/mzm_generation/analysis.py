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
    KitaevHamiltonianExperimentParameters,
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
    measurement_labels,
    number_op,
    orbital_combinations,
    post_select_quasis,
    purify_idempotent_matrix,
    site_correlation_op,
)


class KitaevHamiltonianAnalysis(BaseAnalysis):
    "Analyze Kitaev Hamiltonian experimental data."

    def _run_analysis(
        self, experiment_data: ExperimentData
    ) -> Tuple[List[AnalysisResultData], List[Figure]]:
        experiment_params = experiment_data.metadata["params"]

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
            circuit_params = CircuitParameters(
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
            data[circuit_params] = result

        # load readout calibration
        mit = mthree.M3Mitigation()
        mit.cals_from_file(
            os.path.join(
                "data",
                "readout_calibration",
                experiment_params.backend_name,
                f"{experiment_params.timestamp}.json",
            )
        )

        # get results
        results = list(self._compute_analysis_results(experiment_params, data, mit))
        return results, []

    def _compute_analysis_results(
        self,
        params: KitaevHamiltonianExperimentParameters,
        data: Dict[CircuitParameters, Dict],
        mit: mthree.M3Mitigation,
    ) -> Iterable[AnalysisResultData]:
        # fix tunneling and superconducting
        tunneling = -1.0
        superconducting = 1.0

        # get simulation results
        yield from self._compute_simulation_results(tunneling, superconducting, params)

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
        dd_sequences = params.dynamical_decoupling_sequences or [None]
        for chemical_potential in params.chemical_potential_values:
            # diagonalize
            (transformation_matrix, _, _,) = diagonalizing_bogoliubov_transform(
                params.n_modes,
                tunneling=tunneling,
                superconducting=superconducting,
                chemical_potential=chemical_potential,
            )
            # compute parity
            W1 = transformation_matrix[:, : params.n_modes]
            W2 = transformation_matrix[:, params.n_modes :]
            full_transformation_matrix = np.block([[W1, W2], [W2.conj(), W1.conj()]])
            hamiltonian_parity = np.sign(
                np.real(np.linalg.det(full_transformation_matrix))
            )
            # compute quasis and correlation matrices
            for occupied_orbitals in params.occupied_orbitals_list:
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
                    for permutation, label in measurement_labels(params.n_modes):
                        circuit_params = CircuitParameters(
                            tunneling,
                            superconducting,
                            chemical_potential,
                            occupied_orbitals,
                            permutation,
                            label,
                            dynamical_decoupling_sequence=dd_sequence,
                        )
                        if circuit_params in data:
                            counts = data[circuit_params]["counts"]
                            # raw quasis
                            quasis_raw[permutation, label] = counts_to_quasis(counts)
                            # measurement error mitigation
                            quasis_mem[permutation, label] = mit.apply_correction(
                                counts,
                                params.qubits,
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
                    ] = compute_correlation_matrix(quasis_raw)
                    corr_mem[
                        tunneling,
                        superconducting,
                        chemical_potential,
                        occupied_orbitals,
                        dd_sequence,
                    ] = compute_correlation_matrix(quasis_mem)
                    corr_mat_ps, cov_ps = compute_correlation_matrix(quasis_ps)
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
                params.n_modes,
                tunneling,
                superconducting,
                params.chemical_potential_values,
                params.occupied_orbitals_list,
                dd_sequence,
            )
            yield from self._compute_fidelity_witness(
                "mem" + (f"_{dd_sequence}" if dd_sequence else ""),
                corr_mem,
                params.n_modes,
                tunneling,
                superconducting,
                params.chemical_potential_values,
                params.occupied_orbitals_list,
                dd_sequence,
            )
            yield from self._compute_fidelity_witness(
                "ps" + (f"_{dd_sequence}" if dd_sequence else ""),
                corr_ps,
                params.n_modes,
                tunneling,
                superconducting,
                params.chemical_potential_values,
                params.occupied_orbitals_list,
                dd_sequence,
            )
            yield from self._compute_fidelity_witness(
                "pur" + (f"_{dd_sequence}" if dd_sequence else ""),
                corr_pur,
                params.n_modes,
                tunneling,
                superconducting,
                params.chemical_potential_values,
                params.occupied_orbitals_list,
                dd_sequence,
            )
            yield from self._compute_energy(
                "raw" + (f"_{dd_sequence}" if dd_sequence else ""),
                corr_raw,
                params.n_modes,
                tunneling,
                superconducting,
                params.chemical_potential_values,
                params.occupied_orbitals_list,
                dd_sequence,
            )
            yield from self._compute_energy(
                "mem" + (f"_{dd_sequence}" if dd_sequence else ""),
                corr_mem,
                params.n_modes,
                tunneling,
                superconducting,
                params.chemical_potential_values,
                params.occupied_orbitals_list,
                dd_sequence,
            )
            yield from self._compute_energy(
                "ps" + (f"_{dd_sequence}" if dd_sequence else ""),
                corr_ps,
                params.n_modes,
                tunneling,
                superconducting,
                params.chemical_potential_values,
                params.occupied_orbitals_list,
                dd_sequence,
            )
            yield from self._compute_energy(
                "pur" + (f"_{dd_sequence}" if dd_sequence else ""),
                corr_pur,
                params.n_modes,
                tunneling,
                superconducting,
                params.chemical_potential_values,
                params.occupied_orbitals_list,
                dd_sequence,
            )
            yield from self._compute_edge_correlation(
                "raw" + (f"_{dd_sequence}" if dd_sequence else ""),
                corr_raw,
                params.n_modes,
                tunneling,
                superconducting,
                params.chemical_potential_values,
                params.occupied_orbitals_list,
                dd_sequence,
            )
            yield from self._compute_edge_correlation(
                "mem" + (f"_{dd_sequence}" if dd_sequence else ""),
                corr_mem,
                params.n_modes,
                tunneling,
                superconducting,
                params.chemical_potential_values,
                params.occupied_orbitals_list,
                dd_sequence,
            )
            yield from self._compute_edge_correlation(
                "ps" + (f"_{dd_sequence}" if dd_sequence else ""),
                corr_ps,
                params.n_modes,
                tunneling,
                superconducting,
                params.chemical_potential_values,
                params.occupied_orbitals_list,
                dd_sequence,
            )
            yield from self._compute_edge_correlation(
                "pur" + (f"_{dd_sequence}" if dd_sequence else ""),
                corr_pur,
                params.n_modes,
                tunneling,
                superconducting,
                params.chemical_potential_values,
                params.occupied_orbitals_list,
                dd_sequence,
            )
            yield from self._compute_number(
                "raw" + (f"_{dd_sequence}" if dd_sequence else ""),
                corr_raw,
                params.n_modes,
                tunneling,
                superconducting,
                params.chemical_potential_values,
                params.occupied_orbitals_list,
                dd_sequence,
            )
            yield from self._compute_number(
                "mem" + (f"_{dd_sequence}" if dd_sequence else ""),
                corr_mem,
                params.n_modes,
                tunneling,
                superconducting,
                params.chemical_potential_values,
                params.occupied_orbitals_list,
                dd_sequence,
            )
            yield from self._compute_number(
                "ps" + (f"_{dd_sequence}" if dd_sequence else ""),
                corr_ps,
                params.n_modes,
                tunneling,
                superconducting,
                params.chemical_potential_values,
                params.occupied_orbitals_list,
                dd_sequence,
            )
            yield from self._compute_number(
                "pur" + (f"_{dd_sequence}" if dd_sequence else ""),
                corr_pur,
                params.n_modes,
                tunneling,
                superconducting,
                params.chemical_potential_values,
                params.occupied_orbitals_list,
                dd_sequence,
            )
            yield from self._compute_parity(
                "raw" + (f"_{dd_sequence}" if dd_sequence else ""),
                quasi_dists_raw,
                tunneling,
                superconducting,
                params.chemical_potential_values,
                params.occupied_orbitals_list,
                dd_sequence,
            )
            yield from self._compute_parity(
                "mem" + (f"_{dd_sequence}" if dd_sequence else ""),
                quasi_dists_mem,
                tunneling,
                superconducting,
                params.chemical_potential_values,
                params.occupied_orbitals_list,
                dd_sequence,
            )
            yield from self._compute_parity(
                "ps" + (f"_{dd_sequence}" if dd_sequence else ""),
                quasi_dists_ps,
                tunneling,
                superconducting,
                params.chemical_potential_values,
                params.occupied_orbitals_list,
                dd_sequence,
            )
            yield from self._compute_site_correlation(
                "raw" + (f"_{dd_sequence}" if dd_sequence else ""),
                corr_raw,
                params.n_modes,
                tunneling,
                superconducting,
                params.chemical_potential_values,
                params.occupied_orbitals_list,
                dd_sequence,
            )
            yield from self._compute_site_correlation(
                "mem" + (f"_{dd_sequence}" if dd_sequence else ""),
                corr_mem,
                params.n_modes,
                tunneling,
                superconducting,
                params.chemical_potential_values,
                params.occupied_orbitals_list,
                dd_sequence,
            )
            yield from self._compute_site_correlation(
                "ps" + (f"_{dd_sequence}" if dd_sequence else ""),
                corr_ps,
                params.n_modes,
                tunneling,
                superconducting,
                params.chemical_potential_values,
                params.occupied_orbitals_list,
                dd_sequence,
            )
            yield from self._compute_site_correlation(
                "pur" + (f"_{dd_sequence}" if dd_sequence else ""),
                corr_pur,
                params.n_modes,
                tunneling,
                superconducting,
                params.chemical_potential_values,
                params.occupied_orbitals_list,
                dd_sequence,
            )

    def _compute_simulation_results(
        self,
        tunneling: float,
        superconducting: Union[float, complex],
        params: KitaevHamiltonianExperimentParameters,
    ) -> Iterable[AnalysisResultData]:
        # set chemical potential values to the experiment range but with fixed resolution
        start = params.chemical_potential_values[0]
        # avoid discontinuities at 0
        if start == 0:
            start = 1e-8
        chemical_potential_values = np.linspace(
            start,
            params.chemical_potential_values[-1],
            num=50,
        )

        # construct operators
        edge_correlation = edge_correlation_op(params.n_modes)

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
                params.n_modes,
                tunneling=tunneling,
                superconducting=superconducting,
                chemical_potential=chemical_potential,
            )
            # compute parity
            W1 = transformation_matrix[:, : params.n_modes]
            W2 = transformation_matrix[:, params.n_modes :]
            full_transformation_matrix = np.block([[W1, W2], [W2.conj(), W1.conj()]])
            hamiltonian_parity = np.sign(
                np.real(np.linalg.det(full_transformation_matrix))
            )
            # compute results
            for occupied_orbitals in params.occupied_orbitals_list:
                # compute exact correlation matrix
                occupation = np.zeros(params.n_modes)
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
                exact_number = np.real(np.sum(np.diag(corr_exact)[: params.n_modes]))
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
        occupied_orbitals_set = set(params.occupied_orbitals_list)
        combs = list(orbital_combinations(params.n_modes))
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
            high = np.array(energy_exact[tuple(range(params.n_modes))])
            for i in range(threshold):
                particle = np.array(energy_exact[combs[2 * i + 2]])
                hole = np.array(energy_exact[combs[2 * i + 3]])
                bdg_energy[i] = low - particle
                bdg_energy[threshold + i] = high - hole
            yield AnalysisResultData(
                f"bdg_energy_exact", (bdg_energy, chemical_potential_values)
            )

        # site correlation
        # construct operators
        site_correlation_ops = [
            site_correlation_op(i) for i in range(1, 2 * params.n_modes)
        ]

        # create data storage objects
        site_correlation_exact = defaultdict(list)  # Dict[Tuple[int, ...], List[float]]

        for chemical_potential in params.chemical_potential_values:
            # diagonalize Hamiltonian
            (
                transformation_matrix,
                orbital_energies,
                constant,
            ) = diagonalizing_bogoliubov_transform(
                params.n_modes,
                tunneling=tunneling,
                superconducting=superconducting,
                chemical_potential=chemical_potential,
            )
            W1 = transformation_matrix[:, : params.n_modes]
            W2 = transformation_matrix[:, params.n_modes :]
            full_transformation_matrix = np.block([[W1, W2], [W2.conj(), W1.conj()]])
            # compute results
            for occupied_orbitals in params.occupied_orbitals_list:
                # compute exact correlation matrix
                occupation = np.zeros(params.n_modes)
                occupation[list(occupied_orbitals)] = 1.0
                corr_diag = np.diag(np.concatenate([occupation, 1 - occupation]))
                corr_exact = (
                    full_transformation_matrix.T.conj()
                    @ corr_diag
                    @ full_transformation_matrix
                )
                for site_correlation in site_correlation_ops:
                    exact_site_correlation, _ = np.real(
                        expectation_from_correlation_matrix(
                            site_correlation, corr_exact
                        )
                    )
                    site_correlation_exact[
                        chemical_potential, occupied_orbitals
                    ].append(exact_site_correlation)

        yield AnalysisResultData(
            "site_correlation_exact",
            {k: np.array(v) for k, v in site_correlation_exact.items()},
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
            error = np.zeros(len(chemical_potential_values))
            error_stddev = np.zeros(len(chemical_potential_values))
            low_exact = energy_exact[()]
            high_exact = energy_exact[tuple(range(n_modes))]
            for i in range(threshold):
                # data
                particle, particle_stddev = data_zipped[combs[2 * i + 2]]
                hole, hole_stddev = data_zipped[combs[2 * i + 3]]
                # exact values
                particle_exact = np.array(energy_exact[combs[2 * i + 2]])
                hole_exact = np.array(energy_exact[combs[2 * i + 3]])
                # energy
                bdg_energy[i] = low - particle
                bdg_energy[threshold + i] = high - hole
                # stddev
                bdg_stddev[i] = low_stddev ** 2 + particle_stddev ** 2
                bdg_stddev[threshold + i] = high_stddev ** 2 + hole_stddev ** 2
                # error
                error += np.abs((low - particle) - (low_exact - particle_exact))
                error += np.abs((high - hole) - (high_exact - hole_exact))
                # error stddev
                error_stddev += (
                    np.array(low_stddev) ** 2
                    + np.array(particle_stddev) ** 2
                    + np.array(high_stddev) ** 2
                    + np.array(hole_stddev) ** 2
                )
            bdg_stddev = np.sqrt(bdg_stddev)
            error /= 2 * threshold
            error_stddev = np.sqrt(error_stddev) / (2 * threshold)
            yield AnalysisResultData(f"bdg_energy_{label}", (bdg_energy, bdg_stddev))
            yield AnalysisResultData(f"bdg_energy_error_{label}", (error, error_stddev))

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

    def _compute_site_correlation(
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
        site_correlation_ops = [site_correlation_op(i) for i in range(1, 2 * n_modes)]
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
                for site_correlation in site_correlation_ops:
                    site_correlation_val, stddev = np.real(
                        expectation_from_correlation_matrix(
                            site_correlation, corr_mat, cov
                        )
                    )
                    data[chemical_potential, occupied_orbitals].append(
                        (site_correlation_val, stddev)
                    )
        data_zipped = {k: tuple(np.array(a) for a in zip(*v)) for k, v in data.items()}
        yield AnalysisResultData(f"site_correlation_{label}", data_zipped)
